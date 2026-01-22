import os
import sys
import math
import random
import pickle
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchmetrics import Metric

# Ensure relative imports work regardless of CWD
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Set wandb directory (optional, can be set via environment variable)
if "WANDB_DIR" not in os.environ:
    os.environ["WANDB_DIR"] = os.path.expanduser("~/wandb")

from composer import Trainer, algorithms, ComposerModel, Evaluator
from composer.callbacks import LRMonitor, MemoryMonitor, OptimizerMonitor, RuntimeEstimator, SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
)
from composer.core import Callback, State, Event
from composer.core.types import Batch
from composer.utils import dist, reproducibility

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

import src.flex_bert as flex_bert_module
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
from src.scheduler import WarmupStableDecayScheduler
from src.arg_data import ARGDataset
from src.text_data import DistributedSamplerPCG64DXSM
from composer.core.types import Dataset
import transformers


def update_batch_size_info(cfg: DictConfig):
    """Copy of the helper from sequence_classification.py for consistency."""
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f"WARNING: device_train_microbatch_size > device_train_batch_size, "
                f"will be reduced from {device_microbatch_size} -> {device_train_batch_size}."
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size

    # Safely set `device_eval_microbatch_size` if not provided by user
    if "device_eval_microbatch_size" not in cfg:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_microbatch_size = 1
        else:
            cfg.device_eval_microbatch_size = cfg.device_train_microbatch_size

    global_eval_batch_size, device_eval_microbatch_size = (
        cfg.get("global_eval_batch_size", global_batch_size),
        cfg.device_eval_microbatch_size,
    )
    device_eval_batch_size = global_eval_batch_size // dist.get_world_size()
    if isinstance(device_eval_microbatch_size, int):
        if device_eval_microbatch_size > device_eval_batch_size:
            print(
                f"WARNING: device_eval_microbatch_size > device_eval_batch_size, "
                f"will be reduced from {device_eval_microbatch_size} -> {device_eval_batch_size}."
            )
            device_eval_microbatch_size = device_eval_batch_size
    cfg.device_eval_batch_size = device_eval_batch_size
    cfg.device_eval_microbatch_size = device_eval_microbatch_size
    return cfg


def setup_logging(run_name: str = "default"):
    import logging
    import sys as _sys
    console_handler = logging.StreamHandler(_sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    composer_logger = logging.getLogger('composer')
    composer_logger.setLevel(logging.DEBUG)


def log_config(cfg: DictConfig):
    import logging
    config_str = om.to_yaml(cfg)
    logging.info("Training configuration:")
    logging.info("\n" + config_str)
    if "wandb" in cfg.get("loggers", {}):
        try:
            import wandb
            if wandb.run:
                wandb.config.update(om.to_container(cfg, resolve=True))
        except ImportError as e:
            logging.warning(f"Failed to log config to wandb: {e}")


def build_algorithm(name, kwargs):
    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    elif name == "alibi":
        return algorithms.Alibi(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")


class MaxMetricTracker(Callback):
    """Callback to track and log the maximum value of a metric during training."""
    
    def __init__(self, metric_name: str = "metrics/eval/InBatchSameClassTop1"):
        self.metric_name = metric_name
        self.max_value = 0.0
        self.max_batch = 0
        self.max_epoch = 0
    
    def eval_end(self, state: State, logger):
        # Try to get the metric from the logged metrics
        # The metric name in composer is typically like "metrics/eval/InBatchSameClassTop1"
        current_value = None
        
        # Check state's eval metrics
        if hasattr(state, 'eval_metrics') and state.eval_metrics:
            for evaluator_name, metrics in state.eval_metrics.items():
                for metric_name, metric in metrics.items():
                    if 'InBatchSameClassTop1' in metric_name or 'InBatchSameClassTop1' in str(type(metric)):
                        try:
                            current_value = float(metric.compute())
                        except:
                            pass
        
        if current_value is not None and current_value > self.max_value:
            self.max_value = current_value
            self.max_batch = int(state.timestamp.batch)
            self.max_epoch = int(state.timestamp.epoch)
        
        # Log to wandb
        try:
            logger.log_metrics({
                "max/InBatchSameClassTop1": self.max_value,
                "max/InBatchSameClassTop1_batch": self.max_batch,
            })
        except:
            pass
    
    def fit_end(self, state: State, logger):
        # Final log
        try:
            logger.log_metrics({
                "final/max_InBatchSameClassTop1": self.max_value,
                "final/max_InBatchSameClassTop1_batch": self.max_batch,
            })
        except:
            pass


def build_callback(name, kwargs):
    if name == "lr_monitor":
        return LRMonitor()
    elif name == "memory_monitor":
        return MemoryMonitor()
    elif name == "speed_monitor":
        return SpeedMonitor(
            window_size=kwargs.get("window_size", 1), gpu_flops_available=kwargs.get("gpu_flops_available", None)
        )
    elif name == "runtime_estimator":
        return RuntimeEstimator()
    elif name == "optimizer_monitor":
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True))
    elif name == "max_metric_tracker":
        return MaxMetricTracker(metric_name=kwargs.get("metric_name", "metrics/eval/InBatchSameClassTop1"))
    else:
        raise ValueError(f"Not sure how to build callback: {name}")


def build_logger(name, kwargs):
    if name == "wandb":
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f"Not sure how to build logger: {name}")


def build_scheduler(cfg):
    if cfg.name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "warmup_stable_decay":
        return WarmupStableDecayScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def build_optimizer(cfg, model):
    if cfg.name == "decoupled_adamw":
        return DecoupledAdamW(
            model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Not sure how to build optimizer: {cfg.name}")


class ARGClassificationDataset(Dataset):
    """
    Minimal, local copy of the classification dataset to attach labels for contrastive sampling.
    Reuses the tokenization/padding logic from `src.arg_data.ARGDataset` via composition.
    """

    def __init__(
        self,
        local: str,
        split: str,
        max_seq_len: int,
        labels_file: Optional[str] = None,
        pad_sequences: bool = True,
        skip_extant_tokens: bool = True,
    ):
        super().__init__()
        self.base = ARGDataset(
            local=local,
            split=split,
            max_seq_len=max_seq_len,
            tokenizer=None,
            pad_sequences=pad_sequences,
            skip_extant_tokens=skip_extant_tokens,
        )
        if labels_file is None:
            raise RuntimeError("No labels file specified. Needed for contrastive sampling.")
        labels_path = os.path.join(local, split, labels_file)
        if not os.path.exists(labels_path):
            raise RuntimeError(f"Labels file {labels_path} not found.")
        with open(labels_path, 'rb') as f:
            self.labels: List[int] = pickle.load(f)
        if len(self.labels) != len(self.base):
            raise RuntimeError(
                f"Number of labels ({len(self.labels)}) must match number of sequences ({len(self.base)})."
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        item['label'] = int(self.labels[idx])
        return item


class ARGContrastivePairs(Dataset):
    """
    Yields (query, positive, [negatives]) tuples built from labeled ARG sequences.
    - Query and positive come from the same class.
    - Negatives (optional) are sampled from different classes.
    Uses in-batch negatives by default; explicit negatives further increase hardness/diversity.
    """

    def __init__(
        self,
        local: str,
        split: str,
        max_seq_len: int,
        labels_file: Optional[str],
        pad_sequences: bool = True,
        skip_extant_tokens: bool = True,
        num_hard_negatives: int = 0,
        negative_sampling_temperature: float = 1.0,
    ):
        super().__init__()
        self.ds = ARGClassificationDataset(
            local=local,
            split=split,
            max_seq_len=max_seq_len,
            labels_file=labels_file,
            pad_sequences=pad_sequences,
            skip_extant_tokens=skip_extant_tokens,
        )
        self.num_hard_negatives = int(max(0, num_hard_negatives))
        self.temperature_sampling = float(max(1e-6, negative_sampling_temperature))

        # Build label -> indices mapping
        self.label_to_indices: Dict[int, List[int]] = {}
        for i in range(len(self.ds)):
            lab = self.ds.labels[i]
            self.label_to_indices.setdefault(int(lab), []).append(i)

        # Precompute a flat list of (label, indices) for negatives
        self.labels_sorted = sorted(self.label_to_indices.keys())
        self.num_classes = len(self.labels_sorted)
        self.indices_per_label = [self.label_to_indices[l] for l in self.labels_sorted]
        self.class_sizes = torch.tensor([len(ixs) for ixs in self.indices_per_label], dtype=torch.float)
        self.class_probs = (self.class_sizes / self.class_sizes.sum()).tolist()

    def _sample_positive_index(self, label: int, exclude_idx: int) -> int:
        candidates = self.label_to_indices[label]
        if len(candidates) < 2:
            # Duplicate if only one example exists
            return exclude_idx
        while True:
            j = random.choice(candidates)
            if j != exclude_idx:
                return j

    def _sample_negative_indices(self, label: int, k: int) -> List[int]:
        if k <= 0:
            return []
        # Sample negative labels proportional to class frequency
        neg_labels = []
        while len(neg_labels) < k:
            lbl = random.choices(self.labels_sorted, weights=self.class_probs, k=1)[0]
            if lbl != label:
                neg_labels.append(lbl)
        neg_indices = [random.choice(self.label_to_indices[lbl]) for lbl in neg_labels]
        return neg_indices

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor = self.ds[idx]
        label = int(anchor['label'])
        pos_idx = self._sample_positive_index(label, exclude_idx=idx)
        positive = self.ds[pos_idx]

        sample: Dict[str, Any] = {
            # Query
            'q_input_ids': torch.tensor(anchor['input_ids'], dtype=torch.long),
            'q_attention_mask': torch.tensor(anchor['attention_mask'], dtype=torch.long),
            # Positive
            'p_input_ids': torch.tensor(positive['input_ids'], dtype=torch.long),
            'p_attention_mask': torch.tensor(positive['attention_mask'], dtype=torch.long),
        }

        if self.num_hard_negatives > 0:
            neg_indices = self._sample_negative_indices(label, self.num_hard_negatives)
            neg_items = [self.ds[nidx] for nidx in neg_indices]
            sample['n_input_ids'] = torch.stack(
                [torch.tensor(it['input_ids'], dtype=torch.long) for it in neg_items], dim=0
            )  # [K, L]
            sample['n_attention_mask'] = torch.stack(
                [torch.tensor(it['attention_mask'], dtype=torch.long) for it in neg_items], dim=0
            )  # [K, L]

        return sample


def contrastive_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    # Stack query and positive
    batch['q_input_ids'] = torch.stack([f['q_input_ids'] for f in features], dim=0)
    batch['q_attention_mask'] = torch.stack([f['q_attention_mask'] for f in features], dim=0)
    batch['p_input_ids'] = torch.stack([f['p_input_ids'] for f in features], dim=0)
    batch['p_attention_mask'] = torch.stack([f['p_attention_mask'] for f in features], dim=0)
    # Optional negatives
    if 'n_input_ids' in features[0]:
        # Pad to same K across batch if needed
        max_k = max(f['n_input_ids'].shape[0] for f in features)
        ids_list = []
        attn_list = []
        for f in features:
            cur_k = f['n_input_ids'].shape[0]
            if cur_k < max_k:
                pad_ids = f['n_input_ids'][0:1].repeat(max_k - cur_k, 1)
                pad_attn = f['n_attention_mask'][0:1].repeat(max_k - cur_k, 1)
                ids_list.append(torch.cat([f['n_input_ids'], pad_ids], dim=0))
                attn_list.append(torch.cat([f['n_attention_mask'], pad_attn], dim=0))
            else:
                ids_list.append(f['n_input_ids'])
                attn_list.append(f['n_attention_mask'])
        batch['n_input_ids'] = torch.stack(ids_list, dim=0)         # [B, K, L]
        batch['n_attention_mask'] = torch.stack(attn_list, dim=0)    # [B, K, L]
    return batch

def supervised_contrastive_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([torch.tensor(f['input_ids'], dtype=torch.long) for f in features], dim=0)
    attention_mask = torch.stack([torch.tensor(f['attention_mask'], dtype=torch.long) for f in features], dim=0)
    batch: Dict[str, torch.Tensor] = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': torch.tensor([int(f['label']) for f in features], dtype=torch.long),
    }
    if 'token_type_ids' in features[0]:
        batch['token_type_ids'] = torch.stack([torch.tensor(f['token_type_ids'], dtype=torch.long) for f in features], dim=0)
    return batch


class ContrastiveModel(nn.Module):
    """FlexBERT encoder+pooler with supervised (class-level) contrastive loss."""
    def __init__(
        self,
        hf_model_wrapper,
        normalize: bool = True,
        temperature: float = 0.05,
        similarity: str = "cosine",
        class_weights: Optional[Dict[int, float]] = None,
    ):
        super().__init__()
        # `hf_model_wrapper` is a composer HuggingFaceModel from create_flex_bert_classification
        self.backbone = hf_model_wrapper.model  # FlexBertForSequenceClassification
        self.normalize = normalize
        self.temperature = float(temperature)
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be 'cosine' or 'dot'")
        self.similarity = similarity
        # Class weights for weighted contrastive loss (None = no weighting)
        self.class_weights = class_weights

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if token_type_ids is not None:
            output = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            output = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.backbone.head(output)  # [B, H] embedding
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def _similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.similarity == "cosine":
            if not self.normalize:
                a = F.normalize(a, dim=-1)
                b = F.normalize(b, dim=-1)
            return a @ b.t()
        else:
            return a @ b.t()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Single backbone forward reused for both contrastive and classification
        bert_out = self.backbone.bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', None),
        )
        pooled = self.backbone.head(bert_out)  # [B, H]
        z = F.normalize(pooled, p=2, dim=-1) if self.normalize else pooled
        labels = batch['labels']  # [B]

        B = z.shape[0]
        sim = self._similarity(z, z) / self.temperature  # [B, B]

        if B <= 1:
            loss = sim.new_zeros(())
            return {'loss': loss, 'sim': sim.detach()}

        eye = torch.eye(B, dtype=torch.bool, device=sim.device)
        sim_no_self = sim.masked_fill(eye, float('-inf'))

        # Positive mask: same class, exclude self
        eq = labels.view(-1, 1).eq(labels.view(1, -1))
        pos_mask = eq & (~eye)  # [B, B]
        pos_counts = pos_mask.sum(dim=1)  # [B]

        # log-softmax over all non-self entries
        denom_lse = torch.logsumexp(sim_no_self, dim=1, keepdim=True)  # [B, 1]
        log_prob = sim - denom_lse  # [B, B]

        # Average over positives per anchor; ignore anchors with no positives in batch
        loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / torch.clamp(pos_counts.float(), min=1.0)
        valid = pos_counts > 0
        
        # Class-weighted loss: give more weight to minority class anchors
        if valid.any() and self.class_weights is not None:
            # Get class weights for valid anchors
            anchor_labels = labels[valid]
            weights = torch.tensor([self.class_weights.get(int(l), 1.0) for l in anchor_labels], 
                                 device=loss_per_anchor.device, dtype=loss_per_anchor.dtype)
            # Normalize weights to maintain scale
            weights = weights / weights.mean()
            loss = (loss_per_anchor[valid] * weights).mean()
        elif valid.any():
            loss = loss_per_anchor[valid].mean()
        else:
            loss = sim.new_zeros(())

        outputs: Dict[str, torch.Tensor] = {'loss': loss, 'sim': sim.detach()}

        return outputs


class InBatchSameClassTop1Accuracy(Metric):
    """Top-1 accuracy: nearest neighbor (excluding self) shares the same class label."""
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, sim: torch.Tensor, labels: torch.Tensor):  # type: ignore[override]
        B = sim.shape[0]
        if B <= 1:
            return
        eye = torch.eye(B, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(eye, float('-inf'))
        nn_idx = sim.argmax(dim=1)  # [B]
        correct = (labels[nn_idx] == labels).sum()
        self.correct += correct.to(self.correct.device)
        self.total += torch.tensor(B, dtype=torch.long, device=self.total.device)

    def compute(self):  # type: ignore[override]
        return (self.correct.float() / torch.clamp(self.total.float(), min=1.0))


class ContrastiveLoss(Metric):
    """Average contrastive loss across batches."""
    def __init__(self):
        super().__init__()
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_batches", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):  # type: ignore[override]
        # Accept scalar or vector; reduce to scalar
        if loss.numel() > 1:
            loss = loss.mean()
        self.sum_loss += loss.detach()
        self.num_batches += torch.tensor(1, dtype=torch.long, device=self.num_batches.device)

    def compute(self):  # type: ignore[override]
        return self.sum_loss / torch.clamp(self.num_batches.float(), min=1.0)


class ContrastiveComposerModel(ComposerModel):
    """ComposerModel wrapper around supervised ContrastiveModel."""
    def __init__(self, base: ContrastiveModel, num_labels: int):
        super().__init__()
        self.base = base
        # Contrastive-specific retrieval metric
        self._train_metrics = {"InBatchSameClassTop1": InBatchSameClassTop1Accuracy()}
        # Include eval loss for contrastive objective tracking
        self._eval_metrics = {
            "InBatchSameClassTop1": InBatchSameClassTop1Accuracy(),
            "ContrastiveLoss": ContrastiveLoss(),
        }

    def forward(self, batch: Batch):
        return self.base(batch)

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Batch) -> torch.Tensor:
        return outputs["loss"]

    def eval_forward(self, batch: Batch, outputs: Optional[Any] = None):
        return self.base(batch) if outputs is None else outputs

    def get_metrics(self, is_train: bool):
        return self._train_metrics if is_train else self._eval_metrics

    def update_metric(self, batch: Batch, outputs: Dict[str, torch.Tensor], metric):
        if not hasattr(metric, "update"):
            return {}
        name = metric.__class__.__name__
        if name == "InBatchSameClassTop1Accuracy":
            metric.update(outputs["sim"], batch['labels'])
        elif name == "ContrastiveLoss":
            metric.update(outputs["loss"])
        return {}


def build_contrastive_model(cfg: DictConfig):
    """Build a FlexBERT encoder+pooler via the classification constructor, reuse tokenizer setup."""
    # We use the classification builder to get a properly initialized HF model+tokenizer for ARG
    # (it handles ARG tokenizer, embedding resizing, checkpoint loading, etc.)
    hf_wrapper = flex_bert_module.create_flex_bert_classification(
        num_labels=max(2, int(cfg.get("num_labels", 2))),
        pretrained_model_name=cfg.pretrained_model_name,
        pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
        model_config=cfg.get("model_config", None),
        tokenizer_name=cfg.get("tokenizer_name", None),
        gradient_checkpointing=cfg.get("gradient_checkpointing", None),
    )

    contrastive_cfg = cfg.get("contrastive", {})
    temperature = float(contrastive_cfg.get("temperature", 0.05))
    normalize = bool(contrastive_cfg.get("normalize", True))
    similarity = str(contrastive_cfg.get("similarity", "cosine"))
    
    # Compute class weights if requested (inverse frequency weighting)
    class_weights = None
    if contrastive_cfg.get("use_class_weights", False):
        # Load dataset to get label distribution
        try:
            temp_ds = ARGClassificationDataset(
                local=cfg.train_loader.local,
                split=cfg.train_loader.split,
                max_seq_len=cfg.train_loader.max_seq_len,
                labels_file=cfg.train_loader.get("labels_file", "labels.pkl"),
                pad_sequences=True,
                skip_extant_tokens=cfg.train_loader.get("skip_extant_tokens", True),
            )
            from collections import Counter
            label_counts = Counter(temp_ds.labels)
            total = sum(label_counts.values())
            # Inverse frequency: weight = total / (num_classes * count)
            num_classes = len(label_counts)
            class_weights = {
                label: float(total) / (num_classes * count) 
                for label, count in label_counts.items()
            }
            print(f"Computed class weights: {class_weights}")
        except Exception as e:
            print(f"[WARN] Failed to compute class weights: {e}. Using uniform weighting.")

    base = ContrastiveModel(
        hf_model_wrapper=hf_wrapper,
        normalize=normalize,
        temperature=temperature,
        similarity=similarity,
        class_weights=class_weights,
    )
    # Expose tokenizer so we can access vocab or other info if needed
    base.tokenizer = hf_wrapper.tokenizer  # type: ignore[attr-defined]
    num_labels = int(getattr(hf_wrapper.model.config, "num_labels", 2))
    return ContrastiveComposerModel(base, num_labels=num_labels)


def build_contrastive_dataloader(cfg: DictConfig, device_batch_size: int) -> DataLoader:
    assert cfg.get("name") == "arg", "Only ARG data loader is supported for sequence_contrastive.py"
    ds = ARGClassificationDataset(
        local=cfg.local,
        split=cfg.split,
        max_seq_len=cfg.max_seq_len,
        labels_file=cfg.get("labels_file", None),
        pad_sequences=True,
        skip_extant_tokens=cfg.get("skip_extant_tokens", True),
    )

    # Use balanced sampling if requested
    use_balanced_sampling = cfg.get("balanced_sampling", False)
    
    if use_balanced_sampling:
        # Compute sample weights: each sample gets weight = 1 / (num_classes * class_frequency)
        from collections import Counter
        label_counts = Counter(ds.labels)
        total = len(ds.labels)
        num_classes = len(label_counts)
        
        # Weight for each sample = inverse of its class frequency
        sample_weights = [
            float(total) / (num_classes * label_counts[label]) 
            for label in ds.labels
        ]
        
        # Create weighted sampler
        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        
        # For distributed training, we need to wrap this, but for now use the weighted sampler
        # Note: This doesn't handle distributed training properly - would need a distributed version
        if dist.get_world_size() > 1:
            print("[WARN] Balanced sampling with distributed training not fully supported. "
                  "Consider using class_weights in loss instead.")
            sampler = DistributedSamplerPCG64DXSM(
                ds,
                num_replicas=dist.get_world_size(),
                rank=dist.get_global_rank(),
                shuffle=cfg.get("shuffle", True),
                seed=cfg.get("shuffle_seed", 9176),
                drop_last=cfg.drop_last,
            )
        else:
            sampler = weighted_sampler
        
        print(f"Using balanced sampling. Class distribution: {dict(label_counts)}")
    else:
        sampler = DistributedSamplerPCG64DXSM(
            ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_global_rank(),
            shuffle=cfg.get("shuffle", True),
            seed=cfg.get("shuffle_seed", 9176),
            drop_last=cfg.drop_last,
        )

    return DataLoader(
        ds,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2) if cfg.num_workers > 0 else None,
        persistent_workers=cfg.get("persistent_workers", True) if cfg.num_workers > 0 else False,
        timeout=cfg.get("timeout", 0),
        sampler=sampler,
        collate_fn=supervised_contrastive_collate_fn,
    )


def build_model(cfg: DictConfig):
    """Build a classification model (for use with embeddings.py and other classification tasks).
    
    Supports hf_bert, mosaic_bert, and flex_bert model types.
    """
    if cfg.name == "hf_bert":
        return hf_bert_module.create_hf_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get("use_pretrained", False),
            model_config=cfg.get("model_config"),
            tokenizer_name=cfg.get("tokenizer_name"),
            gradient_checkpointing=cfg.get("gradient_checkpointing"),
        )
    elif cfg.name == "mosaic_bert":
        return mosaic_bert_module.create_mosaic_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint"),
            model_config=cfg.get("model_config"),
            tokenizer_name=cfg.get("tokenizer_name"),
            gradient_checkpointing=cfg.get("gradient_checkpointing"),
        )
    elif cfg.name == "flex_bert":
        return flex_bert_module.create_flex_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint"),
            model_config=cfg.get("model_config"),
            tokenizer_name=cfg.get("tokenizer_name"),
            gradient_checkpointing=cfg.get("gradient_checkpointing"),
        )
    else:
        raise ValueError(f"Not sure how to build model with name={cfg.name}")


def build_my_dataloader(cfg: DictConfig, device_batch_size: int):
    """Create a dataloader for classification tasks.
    
    Supports ARG datasets and GLUE datasets (for backward compatibility).
    """
    # Check if we're using ARG data
    if cfg.get("name") == "arg":
        return_metadata = cfg.get("return_metadata", True)
        # Create ARG classification dataset
        dataset = ARGClassificationDataset(
            local=cfg.local,
            split=cfg.split,
            max_seq_len=cfg.max_seq_len,
            labels_file=cfg.get("labels_file", None),
            pad_sequences=True,
            skip_extant_tokens=cfg.get("skip_extant_tokens", True),
        )
        
        # Create a simple collate function for ARG data
        def arg_collate_fn(batch):
            # Convert lists to tensors
            input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
            attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
            token_type_ids = torch.tensor([item['token_type_ids'] for item in batch], dtype=torch.long)
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': labels
            }
            
            # Include extant_node_id_raw if present in batch items
            if 'extant_node_id_raw' in batch[0]:
                result['extant_node_id_raw'] = torch.tensor([
                    (-1 if item.get('extant_node_id_raw') is None else int(item['extant_node_id_raw']))
                    for item in batch
                ], dtype=torch.long)
            
            return result
        
        collate_fn = arg_collate_fn
    else:
        # GLUE dataset logic (for backward compatibility)
        try:
            import src.evals.data as data_module
            dataset = data_module.create_glue_dataset(
                task="qnli",
                split=cfg.split,
                tokenizer_name=cfg.tokenizer_name,
                max_seq_length=cfg.max_seq_len,
            )
            collate_fn = transformers.default_data_collator
        except ImportError:
            raise ValueError(f"GLUE dataset support requires src.evals.data module. For ARG data, use name='arg'.")

    dataset = cast(Dataset, dataset)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset, drop_last=cfg.drop_last, shuffle=cfg.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2) if cfg.num_workers > 0 else None,
        persistent_workers=cfg.get("persistent_workers", True) if cfg.num_workers > 0 else None,
        timeout=cfg.get("timeout", 0),
    )

    return dataloader


def train(cfg: DictConfig):
    if dist.get_local_rank() == 0:
        setup_logging(run_name=cfg.get("run_name", "default"))
    print("Training using config: ")
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    cfg = update_batch_size_info(cfg)

    # Build model
    print("Initializing model...")
    model = build_contrastive_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params={n_params:.4e}")

    # Dataloaders
    print("Building train loader...")
    train_loader = build_contrastive_dataloader(
        cfg.train_loader,
        device_batch_size=cfg.global_train_batch_size // dist.get_world_size(),
    )

    print("Building eval loader...")
    global_eval_batch_size = cfg.get("global_eval_batch_size", cfg.global_train_batch_size)
    eval_loader = build_contrastive_dataloader(
        cfg.eval_loader,
        device_batch_size=(global_eval_batch_size // dist.get_world_size()),
    )
    eval_evaluator = Evaluator(
        label="eval",
        dataloader=eval_loader,
        device_eval_microbatch_size=cfg.get("device_eval_microbatch_size", None),
    )

    # Optimizer, scheduler, monitors
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = build_scheduler(cfg.scheduler)
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get("loggers", {}).items()]
    callbacks: List[Callback] = [build_callback(name, cb_cfg) for name, cb_cfg in cfg.get("callbacks", {}).items()]
    # Always add MaxMetricTracker to track best eval accuracy
    callbacks.append(MaxMetricTracker())
    algos = [build_algorithm(name, algo_cfg) for name, algo_cfg in cfg.get("algorithms", {}).items()]

    if cfg.get("run_name") is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "sequence-contrastive")

    # Resolve config variables (e.g., ${run_name} in save_folder)
    om.resolve(cfg)

    # Validate checkpoint save configuration
    save_folder = cfg.get("save_folder")
    if save_folder is not None:
        if "${" in str(save_folder):
            raise ValueError(
                f"save_folder contains unresolved variables: {save_folder}\n"
                f"This will prevent checkpoints from being saved correctly."
            )

    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algos,
        train_dataloader=train_loader,
        eval_dataloader=eval_evaluator,
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get("device"),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        save_folder=cfg.get("save_folder"),
        save_interval=cfg.get("save_interval", "1000ba"),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", -1),
        save_overwrite=cfg.get("save_overwrite", False),
        load_path=cfg.get("load_path"),
        load_weights_only=True,
    )

    print("Logging config...")
    log_config(cfg)
    print("Starting training...")
    trainer.fit()


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    
    # Check for "random-weights" flag
    use_random_weights = "random-weights" in args_list
    if use_random_weights:
        args_list = [arg for arg in args_list if arg != "random-weights"]
    
    with open("yamls/defaults.yaml") as f:
        default_cfg = om.load(f)
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(default_cfg, yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)
    
    # Override pretrained_checkpoint if random-weights flag is set
    if use_random_weights:
        if "model" not in cfg:
            cfg.model = {}
        cfg.model.pretrained_checkpoint = None
    
    train(cfg)


