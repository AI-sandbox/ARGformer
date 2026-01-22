import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from typing import Any
from pathlib import Path
from omegaconf import OmegaConf as om
from typing import Optional

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from composer.utils import dist
from main import build_model as build_mlm_model, build_dataloader, update_batch_size_info
from sequence_contrastive import build_model as build_classification_model, build_my_dataloader
from src.arg_data import ARGDataset
from torch.utils.data import DataLoader

def load_checkpoint_model(yaml_path, checkpoint_path, is_classification=False):
    """
    Load the model and tokenizer from a checkpoint.
    
    Args:
        yaml_path: Path to the YAML configuration file
        checkpoint_path: Path to the checkpoint file (e.g., 'latest-rank0.pt')
        is_classification: Whether the model is for classification
    
    Returns:
        model: The loaded model
    """
    # Load the configuration
    cfg = om.load(yaml_path)
    
    # Update batch size info
    cfg = update_batch_size_info(cfg)
    
    # Build the model from config
    if is_classification:
        # Avoid triggering an internal load of cfg.model.pretrained_checkpoint inside the builder.
        # We will load the provided --checkpoint below with proper key mapping.
        try:
            if getattr(cfg.model, 'pretrained_checkpoint', None):
                print("[embeddings] Ignoring cfg.model.pretrained_checkpoint during embedding extraction; using --checkpoint instead.")
                cfg.model.pretrained_checkpoint = None
        except (AttributeError, TypeError):
            # cfg.model might not have pretrained_checkpoint attribute, which is fine
            pass
        model = build_classification_model(cfg.model)
    else:
        model = build_mlm_model(cfg.model)

    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state = torch.load(checkpoint_path, map_location=device)
    model_state = state.get('state', {}).get('model', {})

    # Map various Composer prefixes to the inner HF model param names
    def _map_key(k: str) -> str:
        # Remove top-level wrapper prefix used by HuggingFaceModel
        if k.startswith('model.'):
            k = k[len('model.'):]
        # Remove prefixes used by ContrastiveComposerModel → ContrastiveModel → backbone
        for pfx in ('base.backbone.', 'backbone.'):
            if k.startswith(pfx):
                k = k[len(pfx):]
                break
        return k

    mapped = {_map_key(k): v for k, v in model_state.items()}
    # Filter out potential missing bias terms in certain norm implementations
    filtered_model_state = {k: v for k, v in mapped.items() if not k.endswith('norm.bias')}

    # Load the mapped state dict into the underlying HF model, allowing missing/unexpected keys
    load_result = model.model.load_state_dict(filtered_model_state, strict=False)
    print(f"Loaded checkpoint params: {len(filtered_model_state)}; missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")

    # Fail fast if we did not load core encoder / pooling head weights.
    missing = set(load_result.missing_keys)
    missing_bert = sorted([k for k in missing if k.startswith("bert.")])
    missing_head = sorted([k for k in missing if k.startswith("head.")])
    if len(missing_bert) > 0:
        raise RuntimeError(
            "Encoder weights were not fully loaded (missing keys under 'bert.'). "
            f"Example missing keys: {missing_bert[:10]}"
        )
    if len(missing_head) > 0:
        raise RuntimeError(
            "Pooling head weights were not fully loaded (missing keys under 'head.'). "
            "If you use --pooling head/head_norm, embeddings will be wrong. "
            f"Example missing keys: {missing_head[:10]}"
        )

    # Set to evaluation mode and move to device
    model.eval()
    model.to(device)
    
    return model, cfg

def get_embeddings(model, batch, layer_index=None, pooling_strategy="cls"):
    """
    Extract embeddings from the model.
    
    Args:
        model: The loaded model
        batch: Input data batch
        layer_index: Which transformer layer to extract embeddings from (None for last layer)
        pooling_strategy: How to pool token embeddings ("cls", "mean", or "all")
    
    Returns:
        embeddings: The extracted embeddings
    """
    # Move batch to the same device as the model
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    assert layer_index is None, "Layer index is not supported for embeddings extraction yet"
    
    with torch.no_grad():
        # If using FlexBertForMaskedLM, we need to use the internal BERT model
    
        if hasattr(model, 'model') and hasattr(model.model, 'bert'):
            # For FlexBertForMaskedLM or FlexBertForSequenceClassification
            bert_model = model.model.bert
            
            # MASK token is 1, verify there is no MASK token in the input_ids if it's not a classification model
            if not hasattr(model.model, 'num_labels'):
                mask_count = (batch["input_ids"] == 1).sum().item()
                if mask_count > 0:
                    raise ValueError(f"Found {mask_count} MASK tokens in input_ids for non-classification model. This is not expected for embedding extraction.")
            
            # Direct call to the bert model instead of going through the MLM or classification head
            
            bert_out = bert_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            hidden_states = bert_out.last_hidden_state if hasattr(bert_out, 'last_hidden_state') else bert_out
        else:
            raise ValueError("Model is not a FlexBertForMaskedLM or FlexBertForSequenceClassification")
            # (Haven't tested this yet)
            # Get the raw model from the MLM wrapper (for other model types)
            bert_model = model.model
            
            # Forward pass through the model
            outputs = bert_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                output_hidden_states=True  # Get all hidden states
            )
            
            # Get the appropriate hidden states
            if layer_index is None:
                # Use the last layer by default
                hidden_states = outputs.last_hidden_state
            else:
                # Get the specified layer's hidden states
                hidden_states = outputs.hidden_states[layer_index]
        
        # Apply pooling strategy
        if pooling_strategy == "cls":
            # Use the [CLS] token embedding (first token)
            embeddings = hidden_states[:, 0, :]
        elif pooling_strategy == "extant":
            embeddings = hidden_states[:, 1, :]
        elif pooling_strategy == "mean":
            # Average over all tokens (considering attention mask if available)
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"]
                # Sum and divide by the number of tokens
                sum_embeddings = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
                count_tokens = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1)
                embeddings = sum_embeddings / count_tokens
            else:
                embeddings = torch.mean(hidden_states, dim=1)

        elif pooling_strategy in ("head", "head_norm"):
            # Use the pooling head exactly as in training: head(bert_out)
            # Some implementations behave differently if you pass a Tensor.
            try:
                pooled = model.model.head(bert_out)  # type: ignore[arg-type]
            except (TypeError, AttributeError) as e:
                # Fallback for heads that only accept a tensor
                print(f"\n[embeddings] Fallback: head() requires tensor input ({type(e).__name__}), using hidden_states instead\n")
                pooled = model.model.head(hidden_states)
            embeddings = F.normalize(pooled, p=2, dim=-1) if pooling_strategy == "head_norm" else pooled
        elif pooling_strategy == "all":
            # Return embeddings for all tokens
            embeddings = hidden_states
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return embeddings


def _build_arg_split_dataloader(local: str, split: str, max_seq_len: int, batch_size: int, num_workers: int = 0, return_metadata: bool = True, use_dataset_labels: bool = False) -> DataLoader:
    """
    Build a simple dataloader for an ARG split using ARGDataset directly.
    Used for embedding splits that may not have labels (e.g., heldout).
    """
    ds = ARGDataset(
        local=local,
        split=split,
        max_seq_len=max_seq_len,
        tokenizer=None,
        pad_sequences=True,
        skip_extant_tokens=True,
        return_metadata=return_metadata,
    )

    # Optional labels from labels.pkl
    labels_list = None
    labels_cursor = {"i": 0}
    if use_dataset_labels:
        labels_path = os.path.join(local, split, "labels.pkl")
        if os.path.exists(labels_path):
            with open(labels_path, 'rb') as f:
                labels_list = pickle.load(f)

    def collate_fn(batch):
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Optional fields
        if "token_type_ids" in batch[0]:
            out["token_type_ids"] = torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long)
        if "extant_node_id_raw" in batch[0]:
            out["extant_node_id_raw"] = torch.tensor([
                (-1 if (b["extant_node_id_raw"] is None) else int(b["extant_node_id_raw"])) for b in batch
            ], dtype=torch.long)
        # Attach labels from file if available
        if labels_list is not None:
            start = labels_cursor["i"]
            end = start + input_ids.shape[0]
            out["labels"] = torch.tensor(labels_list[start:end], dtype=torch.long)
            labels_cursor["i"] = end
        return out

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

def embed_arg_dataset(model, cfg, output_path, batch_size=32, pooling_strategy="extant", layer_index=None, max_samples=None, is_classification=False, include_splits: str = None, use_split_labels: bool = False):
    """
    Embed an entire ARG dataset and save the embeddings.
    
    Args:
        model: The loaded model
        cfg: The configuration
        output_path: Path to save the embeddings
        batch_size: Batch size for processing
        pooling_strategy: Pooling strategy for embeddings
        layer_index: Which layer to extract embeddings from (None for last layer)
        max_samples: Maximum number of samples to process (None or <=0 for all)
    """
    # Create eval dataloader using the same approach as in eval_arg_reconstruction.py
    eval_batch_size = cfg.device_eval_batch_size if hasattr(cfg, 'device_eval_batch_size') else cfg.device_train_batch_size
    
    dataloaders = []
    splits = []
    int_to_pop = None

    # If include_splits is provided, build ARGDataset-based loaders for those splits (supports unlabeled splits)
    if include_splits is not None:
        # Comma-separated string to list
        split_list = [s.strip() for s in include_splits.split(',') if s.strip()]
        data_local = getattr(cfg.train_loader, 'local', None)
        if data_local is None:
            raise ValueError("cfg.train_loader.local must be set to locate ARG data")
        # Use top-level or model-config max seq len
        max_len = None
        try:
            max_len = int(getattr(cfg, 'max_seq_len', None) or getattr(cfg.model.model_config, 'max_seq_len', None))
        except Exception:
            max_len = getattr(cfg, 'max_seq_len', None)
        for sp in split_list:
            split_dir = os.path.join(data_local, sp)
            if not os.path.isdir(split_dir):
                print(f"[WARN] Split directory not found, skipping: {split_dir}")
                continue
            dl = _build_arg_split_dataloader(
                local=data_local,
                split=sp,
                max_seq_len=max_len,
                batch_size=eval_batch_size,
                num_workers=0,
                return_metadata=True,
                use_dataset_labels=bool(use_split_labels),
            )
            dataloaders.append(dl)
            splits.append(sp)
        # Disable classification label processing in this mode
        all_labels = [] if use_split_labels else None
        int_to_pop = None
    elif is_classification:
        # Classification path: use labeled train/val
        train_dl = build_my_dataloader(cfg=cfg.train_loader, device_batch_size=eval_batch_size)
        dataloaders.append(train_dl)
        splits.append("train")
        val_dl = build_my_dataloader(cfg=cfg.eval_loader, device_batch_size=eval_batch_size)
        dataloaders.append(val_dl)
        splits.append("val")
    else:
        # Pretraining-like path: fallback to main.build_dataloader
        embedding_loader_cfg = cfg.train_loader.copy()
        embedding_loader_cfg.dataset.mlm_probability = None
        # Always enable return_metadata for embedding extraction to capture extant_node_ids
        setattr(embedding_loader_cfg.dataset, 'return_metadata', True)
        dataloader_spec = build_dataloader(
            cfg=embedding_loader_cfg,
            tokenizer=model.tokenizer,
            device_batch_size=eval_batch_size
        )
        dataloaders.append(dataloader_spec.dataloader)
        splits.append("train")
    

    all_embeddings = []
    all_input_ids = []
    all_attention_masks = []
    all_labels = [] if is_classification else None
    all_extant_node_ids = []
    
    
    # Process batches
    samples_processed = 0
    # Interpret non-positive max_samples as unlimited
    max_limit = None if (max_samples is None or int(max_samples) <= 0) else int(max_samples)
    
    # Create iterators for all dataloaders to allow interleaved batch sampling
    dl_iters_with_splits = list(zip([iter(dl) for dl in dataloaders], splits))

    with torch.no_grad():
        while dl_iters_with_splits and (max_limit is None or samples_processed < max_limit):
            # Randomly pick a dataloader iterator to sample from
            import random
            random_idx = random.randrange(len(dl_iters_with_splits))
            chosen_iter, split_name = dl_iters_with_splits[random_idx]
            try:
                batch = next(chosen_iter)
                
                # Get embeddings
                embeddings = get_embeddings(
                    model=model, 
                    batch=batch,
                    layer_index=layer_index,
                    pooling_strategy=pooling_strategy
                )
                
                # Store embeddings and IDs
                all_embeddings.append(embeddings.cpu())
                # Save reference ids
                if "input_ids" in batch:
                    all_input_ids.append(batch["input_ids"].cpu())
                # Capture extant node ids if available
                extant_ids_tensor = batch.get('extant_node_id_raw', None)
                if isinstance(extant_ids_tensor, torch.Tensor):
                    all_extant_node_ids.append(extant_ids_tensor.detach().cpu())
                if is_classification and all_labels is not None:
                    if "labels" in batch:
                        all_labels.append(batch["labels"].detach().cpu())
                    else:
                        # No labels provided; skip collecting
                        pass
                
                # Count samples and handle max_samples limit by slicing the last batch if needed
                current_batch_size = batch["input_ids"].size(0)
                if (max_limit is not None) and (samples_processed + current_batch_size > max_limit):
                    needed = max_samples - samples_processed
                    # Slice the last appended tensors to the exact size
                    all_embeddings[-1] = all_embeddings[-1][:needed]
                    if all_input_ids: all_input_ids[-1] = all_input_ids[-1][:needed]
                    if all_extant_node_ids: all_extant_node_ids[-1] = all_extant_node_ids[-1][:needed]

                    # Only slice labels if present and non-empty
                    if is_classification and (all_labels is not None) and (len(all_labels) > 0):
                        all_labels[-1] = all_labels[-1][:needed]
                    samples_processed += needed
                else:
                    samples_processed += current_batch_size
                

            except StopIteration:
                # This dataloader is exhausted, remove it from the list
                dl_iters_with_splits.pop(random_idx)

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Create output data
    output_data = {"embeddings": all_embeddings}
    
    # Add reference data if available
    if all_input_ids:
        output_data["input_ids"] = torch.cat(all_input_ids, dim=0)
    if all_attention_masks:
        output_data["attention_masks"] = torch.cat(all_attention_masks, dim=0)
    if all_labels:
        output_data["labels"] = torch.cat([lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl) for lbl in all_labels], dim=0)
    if all_extant_node_ids:
        output_data["extant_node_ids"] = torch.cat(all_extant_node_ids, dim=0)
    
    # Store metadata
    output_data["metadata"] = {
        "embedding_dim": all_embeddings.shape[1],
        "num_samples": all_embeddings.shape[0],
        "pooling_strategy": pooling_strategy,
        "layer_index": layer_index,
    }
    
    # Add label name mapping for classification if available
    if is_classification and int_to_pop is not None:
        output_data["metadata"]["int_to_pop"] = int_to_pop
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embeddings
    torch.save(output_data, output_path)
    return all_embeddings

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract embeddings from ARG model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output_path", type=str, required=False, help=".pth file path to save the embeddings")
    parser.add_argument("--pooling", type=str, default="extant", choices=["cls", "mean", "all", "extant", "head", "head_norm"], 
                        help="Pooling strategy for token embeddings")
    parser.add_argument("--layer", type=int, default=None, help="Layer to extract (default: last layer)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum samples to process (<=0 means all available across include_splits)")
    parser.add_argument("--include_splits", type=str, default=None, help="Comma-separated list of splits under data_local to embed (e.g., 'train,val,heldout_admix'). If set, uses ARGDataset directly and ignores labels.")
    parser.add_argument("--use_split_labels", action="store_true", help="When using --include_splits, if set, read labels.pkl per split and include them in the saved file.")
    args = parser.parse_args()
    
    # If output_path is not provided, derive it from the checkpoint path
    if args.output_path is None:
        checkpoint_path = Path(args.checkpoint)
        # Remove the file extension and add _embeddings_{pooling}.pt
        args.output_path = str(checkpoint_path.with_suffix('')) + f"_embeddings_{args.pooling}.pt"
    
    # Determine if this is a classification or contrastive task
    is_classification = 'classification' in args.config or 'contrastive' in args.config
    
    # Load model and config
    model, cfg = load_checkpoint_model(args.config, args.checkpoint, is_classification=is_classification)
    
    # Extract embeddings from the dataset
    embed_arg_dataset(
        model=model,
        cfg=cfg,
        output_path=args.output_path,
        batch_size=args.batch_size,
        pooling_strategy=args.pooling,
        layer_index=args.layer,
        max_samples=args.max_samples,
        is_classification=is_classification,
        include_splits=args.include_splits,
        use_split_labels=args.use_split_labels
    )

if __name__ == "__main__":
    main()
