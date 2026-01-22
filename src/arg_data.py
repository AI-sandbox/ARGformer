import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from composer.core import DataSpec
from composer.utils import dist
import transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union, Tuple
import json
import sys
sys.path.append('.')
from src.text_data import DistributedSamplerPCG64DXSM

class ARGTokenizer:
    def __init__(self, vocab: dict, model_max_length: int = 512):
        self.vocab = vocab
        self.pad_token_id = vocab["special"]["[PAD]"]
        self.mask_token_id = vocab["special"].get("[MASK]", None)
        if self.mask_token_id is None:
            raise ValueError("Your vocabulary must include a [MASK] token.")
        self.model_max_length = model_max_length

    def batch_decode(self, batch):
        inv_vocab = {}
        for key, token in self.vocab["special"].items():
            inv_vocab[token] = key
        for key, token in self.vocab.get("node_id_tokens", {}).items():
            inv_vocab[token] = key
        decoded = []
        for tokens in batch:
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            decoded.append(" ".join(inv_vocab.get(t, "[UNK]") for t in tokens))
        return decoded


class ARGPreTrainedTokenizer(PreTrainedTokenizerBase):
    def __init__(
        self, 
        arg_tokenizer: ARGTokenizer,
    ): 
        super().__init__(
            bos_token=None,
            eos_token=None,
            unk_token="[UNK]",
            pad_token="[PAD]",
            mask_token="[MASK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
        )
        
        self.arg_tokenizer = arg_tokenizer
        self._inv_vocab = self._build_inv_vocab()
        object.__setattr__(self, "pad_token_id", arg_tokenizer.pad_token_id)
        object.__setattr__(self, "mask_token_id", arg_tokenizer.mask_token_id)
        self.model_max_length = arg_tokenizer.model_max_length
        self.vocab_size = len(self._inv_vocab)
        self.init_kwargs = {}
        self._added_tokens_decoder = {}
        self._added_tokens_encoder = {}

    @property
    def added_tokens_decoder(self):
        return self._added_tokens_decoder

    @property 
    def added_tokens_encoder(self):
        return self._added_tokens_encoder
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        vocab = self.arg_tokenizer.vocab
        vocab_path = os.path.join(save_directory, f"{filename_prefix}vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        return (vocab_path,)


    def _build_inv_vocab(self):
        inv = {}
        for key, token in self.arg_tokenizer.vocab["special"].items():
            inv[token] = key
        for key, token in self.arg_tokenizer.vocab.get("node_id_tokens", {}).items():
            inv[token] = key
        return inv

    def __call__(self, examples, **kwargs):
        input_ids = [ex["input_ids"] for ex in examples]
        attention_mask = [ex["attention_mask"] for ex in examples]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, batch, **kwargs):
        return self.arg_tokenizer.batch_decode(batch)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, list):
            return [self._inv_vocab.get(i, "[UNK]") for i in ids]
        else:
            return self._inv_vocab.get(ids, "[UNK]")

    def convert_tokens_to_ids(self, tokens):
        vocab = {}
        for key, token in self.arg_tokenizer.vocab["special"].items():
            vocab[key] = token
        for key, token in self.arg_tokenizer.vocab.get("node_id_tokens", {}).items():
            vocab[key] = token
        if isinstance(tokens, list):
            return [vocab.get(t, 0) for t in tokens]
        else:
            return vocab.get(tokens, 0)

    def _tokenize(self, text):
        raise ValueError("Data is always pretokenized and this method should not be called")
        

    def get_vocab(self):
        vocab = {}
        vocab.update(self.arg_tokenizer.vocab.get("special", {}))
        vocab.update(self.arg_tokenizer.vocab.get("node_id_tokens", {}))
        return vocab
    
    def __len__(self):
        return len(self.get_vocab())

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=True):
        special_ids = set(self.arg_tokenizer.vocab["special"].values())
        return [1 if token in special_ids else 0 for token in token_ids]
    
    @property
    def is_fast(self):
        return False

class ARGDataset(Dataset):
    """
    A dataset class for ARG data.
    Supports pretokenized sequences or raw sequences with on-the-fly tokenization
    using the provided vocabulary mappings.
    
    Vocabulary must include:
    - 'special': special tokens like [PAD], [CLS], [SEP], etc.
    - 'node_id_tokens': mapping from "node_X" to token IDs
    
    It is a version of `NoStreamingDataset` that is specialized for ARG data.
    """
    def __init__(
        self, 
        local: str,
        split: Optional[str],
        max_seq_len: Optional[int] = None,
        tokenizer: None = None,
        pad_sequences: bool = True,
        skip_extant_tokens: bool = True,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        assert tokenizer is None, "Tokenizer must be None; ARGDataset handles (pre)tokenization itself"
        
        sequences, vocab, is_pretokenized = self._load_split_data(local, split)
        self.vocab = vocab
        self._data_is_pretokenized = is_pretokenized
        self.sequences = sequences
        self.len = len(self.sequences)
        
        self.tokenizer = None
        self.pad_sequences = pad_sequences
        self.pad_token_id = self.vocab["special"]["[PAD]"]
        self.cls_token_id = self.vocab["special"]["[CLS]"]
        self.sep_token_id = self.vocab["special"]["[SEP]"]
        self.skip_extant_tokens = skip_extant_tokens
        self.return_metadata = return_metadata
        if self.cls_token_id is None:
            raise ValueError("Vocabulary must include a [CLS] token.")
        if self.sep_token_id is None:
            raise ValueError("Vocabulary must include a separator-like token to mimic NoStreamingDataset behavior.")
        
        self._node_id_token_map = self.vocab.get("node_id_tokens", {})
        if not self._node_id_token_map:
            raise ValueError("Vocabulary must include 'node_id_tokens'.")
        self._inv_node_id_token_map = None
        
        if max_seq_len is None:
            max_raw_length = max(len(seq) for seq in self.sequences)
            
            if skip_extant_tokens:
                max_raw_length = max_raw_length - 1
            
            inferred_max_seq_len = max_raw_length + 2
            
            self.max_seq_len = inferred_max_seq_len
        else:
            self.max_seq_len = max_seq_len
    
    @staticmethod
    def _load_split_data(local: str, split: str):
        split_path = os.path.join(local, split)
        
        candidate_files = [f"tokenized_{split}_sequences_and_vocab.pkl"]
        if split is not None:
            candidate_files.append(f"raw_{split}_sequences_and_vocab.pkl")
        candidate_files.append(f"{split}_sequences_and_vocab.pkl")
        
        data_path = None
        for fname in candidate_files:
            path_try = os.path.join(split_path, fname)
            if os.path.exists(path_try):
                data_path = path_try
                break
        if data_path is None:
            data_path = os.path.join(split_path, f"tokenized_{split}_sequences_and_vocab.pkl")
        
        data = pickle.load(open(data_path, "rb"))
        vocab = data["vocab"]
        
        if "tokenized_sequences" in data:
            is_pretokenized = True
            sequences = data["tokenized_sequences"]
        elif "raw_sequences" in data:
            is_pretokenized = False
            sequences = data["raw_sequences"]
        else:
            raise KeyError(
                f"Expected 'tokenized_sequences' or 'raw_sequences' in {data_path}"
            )
        
        return sequences, vocab, is_pretokenized
        
    def _tokenize(self, arg_sample):
        raise ValueError("ARGDataset does not support external tokenization calls")

    def _tokenize_raw_sequence_from_vocab(self, raw_sequence: List[int]) -> List[int]:
        token_ids: List[int] = []
        
        for raw_token in raw_sequence:
            vocab_key = f"node_{raw_token}"
            if vocab_key not in self._node_id_token_map:
                raise KeyError(f"Node token '{vocab_key}' not found in vocabulary.")
            token_ids.append(self._node_id_token_map[vocab_key])
        
        return token_ids

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        if not self._data_is_pretokenized:
            seq = self._tokenize_raw_sequence_from_vocab(seq)
        
        extant_node_id_raw = None
        if self.return_metadata:
            if self._data_is_pretokenized:
                if len(seq) >= 1:
                    if self._inv_node_id_token_map is None:
                        self._inv_node_id_token_map = {v: int(k.split('node_')[1]) for k, v in self._node_id_token_map.items()}
                    token_id = seq[0]
                    extant_node_id_raw = self._inv_node_id_token_map.get(token_id, None)
            else:
                raw_seq = self.sequences[idx]
                if len(raw_seq) >= 1:
                    extant_node_id_raw = raw_seq[0]

        if self.skip_extant_tokens:
            if len(seq) >= 1:
                seq = seq[1:]
            else:
                raise ValueError(f"Sequence at index {idx} is too short to remove extant token (length={len(seq)}).")
        
        available_length = self.max_seq_len - 2

        truncated_seq = seq[:available_length]
        seq = [self.cls_token_id] + truncated_seq + [self.sep_token_id]
        
        assert len(seq) <= self.max_seq_len, f"Sequence is too long: {len(seq)} > {self.max_seq_len}"
        if self.pad_sequences:
            if len(seq) < self.max_seq_len:
                padded = seq + [self.pad_token_id] * (self.max_seq_len - len(seq))
                attention_mask = [1] * len(seq) + [0] * (self.max_seq_len - len(seq))
            else:
                padded = seq
                attention_mask = [1] * len(seq)
        else:
            padded = seq
            attention_mask = [1] * len(seq)
        
        token_type_ids = [0] * self.max_seq_len
        
        item = {
            "input_ids": padded,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
        if self.return_metadata:
            item["extant_node_id_raw"] = extant_node_id_raw
        return item
        
    def __len__(self):
        return self.len
    
    def get_max_seq_len(self) -> int:
        return self.max_seq_len


def build_arg_dataloader(
    cfg: DictConfig,
    tokenizer: ARGPreTrainedTokenizer,
    device_batch_size: int,
    device_microbatch_size: int,
):
    assert cfg.name == "arg", f"Tried to build ARG dataloader with cfg.name={cfg.name}"
    assert cfg.dataset.streaming == False, "ARG dataset does not support streaming for now"
    assert cfg.dataset.local is not None, "Local path must be provided when not using streaming"
    assert cfg.dataset.split is not None, "Split must be provided"
    assert cfg.dataset.remote is None, "ARG dataset does not support remote for now"

    dataset = build_arg_dataset(
        cfg=cfg, 
        tokenizer=None,
        pad_sequences=not cfg.get("sequence_packing", False),
        return_metadata=cfg.dataset.get("return_metadata", False),
    )
    sampler = DistributedSamplerPCG64DXSM(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=cfg.dataset.get("shuffle", False),
        seed=cfg.dataset.get("shuffle_seed", 9176),
        drop_last=cfg.drop_last,
    )
    mlm_probability = cfg.dataset.get("mlm_probability", None)
    
    base_collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability
    )

    def collate_fn(batch):
        if getattr(dataset, 'return_metadata', False):
            extant_ids = [b.get('extant_node_id_raw', None) for b in batch]
            stripped = []
            for b in batch:
                if 'extant_node_id_raw' in b:
                    b = {k: v for k, v in b.items() if k != 'extant_node_id_raw'}
                stripped.append(b)
            collated = base_collate_fn(stripped)
            collated['extant_node_id_raw'] = torch.tensor([
                (-1 if (e is None) else int(e)) for e in extant_ids
            ], dtype=torch.long)
            return collated
        else:
            return base_collate_fn(batch)

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2) if cfg.num_workers > 0 else None,
        persistent_workers=cfg.get("persistent_workers", True) if cfg.num_workers > 0 else False,
        timeout=cfg.get("timeout", 0),
        sampler=sampler,
    )
    
def build_arg_dataset(
    cfg: DictConfig,
    tokenizer: ARGPreTrainedTokenizer,
    pad_sequences: bool = True,
    return_metadata: bool = False,
):
    return ARGDataset(
        tokenizer=tokenizer,
        local=cfg.dataset.get("local", None),
        split=cfg.dataset.get("split", None),
        max_seq_len=cfg.dataset.get("max_seq_len", None),
        pad_sequences=pad_sequences,
        skip_extant_tokens=cfg.dataset.get("skip_extant_tokens", False),
        return_metadata=return_metadata,
    )
