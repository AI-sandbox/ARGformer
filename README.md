# ARGformer: ModernBERT for Ancestral Recombination Graph Data

ARGformer is a transformer encoder based on [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) for Ancestral Recombination Graph (ARG) data. It uses the FlexBERT architecture with YAML-based configuration.

The codebase builds upon [MosaicBERT](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert) and the [unmerged fork with Flash Attention 2](https://github.com/Skylion007/mosaicml-examples/tree/skylion007/add-fa2-to-bert) under Apache 2.0 license.

For ModernBERT details, see the [release blog post](https://huggingface.co/blog/modernbert) and [arXiv preprint](https://arxiv.org/abs/2412.13663).


## Setup

```bash
conda env create -f environment.yaml
conda activate bert24
pip install "flash_attn==2.6.3" --no-build-isolation
```

For H100 GPUs, optionally install Flash Attention 3:
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

## Overview

ARGformer supports:
- **Pretraining**: Masked language modeling on ARG sequences
- **Contrastive Learning**: Fine-tuning for retrieval and similarity tasks
- **Embeddings**: Extracting embeddings for downstream analysis
- **Retrieval**: Finding similar sequences in large corpora

## Data Format

ARG data structure:
```
/path/to/arg/data/
├── train/
│   ├── tokenized_train_sequences_and_vocab.pkl
│   └── labels.pkl  # Optional: for contrastive learning
└── val/
    ├── tokenized_val_sequences_and_vocab.pkl
    └── labels.pkl  # Optional: for contrastive learning
```

The `ARGDataset` class supports pretokenized sequences with vocabulary mappings for node IDs and special tokens ([PAD], [CLS], [SEP]).

## Workflow

### 1. Prepare Pretraining Data

Extract sequences from tree files using `src/data/prepare_data_pretrain.py`:
```bash
python src/data/prepare_data_pretrain.py
```

Edit the script to configure input paths, output directory, and train/val split.

### 2. Pretraining

Configure `yamls/mlm.yaml` with dataset paths and model parameters, then run:
```bash
composer main.py yamls/mlm.yaml
```

### 3. Contrastive Fine-tuning

Configure `yamls/contrastive.yaml` with pretrained checkpoint path and run:
```bash
python sequence_contrastive.py yamls/contrastive.yaml
```

### 4. Extract Embeddings

```bash
python embeddings.py [arguments]
```

See the script for usage examples.

### 5. Retrieval

```bash
python retrieve.py [arguments]
```

See the script for usage examples.

## Configuration

Training uses [composer](https://github.com/mosaicml/composer) with YAML configuration files in `yamls/`:
- `mlm.yaml`: Pretraining configuration
- `contrastive.yaml`: Contrastive learning configuration

Key configuration sections:
- `model`: Model architecture and checkpoint paths
- `train_loader` / `eval_loader`: Dataset paths and data loading settings
- `optimizer` / `scheduler`: Training hyperparameters
- `loggers`: WandB logging configuration
