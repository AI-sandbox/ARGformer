"""
Prepare data from tree sequence.

This script extracts unique marginal paths from the tree sequence and prepares
train/val datasets for ARGformer pretraining.
"""

import os
import tskit
import json
import numpy as np
from tqdm import tqdm
import pickle
import random
import gc

# Reserved special tokens.
SPECIAL_TOKENS = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3}
SPECIAL_OFFSET = len(SPECIAL_TOKENS)  # Start index for node tokens.
RANDOM_SEED = 42  # Random seed for train/val split reproducibility

# --- Configuration ---
# NOTE: Update these paths before running the script
TREE_FILE = "path/to/input.vcf.gz.dated.trees"
TRAIN_FRAC = 0.9

# Save mapping (logs) and prepare processed dir for large sequence files
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)
out_dir = f"path/to/data"  # Update this path to your desired output directory
os.makedirs(out_dir, exist_ok=True)

ts = tskit.load(TREE_FILE)

# Get all sample node IDs
selected_sample_nodes = list(ts.samples())

"""Extract unique marginal paths (from extant node to root) for all samples"""

def get_path_key(tree, leaf_node_id):
    path_nodes = []
    u = int(leaf_node_id)
    while u != tskit.NULL:
        path_nodes.append(u)
        u = tree.parent(u)
    return tuple(path_nodes)

path_key_to_id = {}
next_path_id = 0
unique_paths_rows = {}
# Track which tree indices each path_key (sequence) appears in
path_key_to_trees = {}

# For each sample, sweep trees left-to-right and add unique paths
sample_iterator = tqdm(
    enumerate(selected_sample_nodes), total=len(selected_sample_nodes), desc="Processing Samples", unit="sample"
)
for idx, sample_id in sample_iterator:
    tree_iterator = tqdm(
        enumerate(ts.trees()), total=ts.num_trees, desc=f"Processing Trees (Sample {idx+1})", unit="tree", leave=False
    )
    for tree_idx, tree in tree_iterator:
        key = get_path_key(tree, sample_id)
        if key not in path_key_to_id:
            path_id = next_path_id
            path_key_to_id[key] = path_id
            next_path_id += 1
            leaf_node_id = key[0]
            root_node_id = key[-1]
            unique_paths_rows[path_id] = {
                "path_id": path_id,
                "path_length": len(key),
                "leaf_node_id": int(leaf_node_id),
                "root_node_id": int(root_node_id),
                "path_nodes": json.dumps(key),
            }
            # Initialize tree set for this path_key
            path_key_to_trees[key] = set()
        # Track that this path_key appears in this tree
        path_key_to_trees[key].add(tree_idx)

def deduplicate_sequences(sequences):
    """
    Remove duplicate sequences while preserving order.
    Each sequence is assumed to be a list of tokens.
    """
    seen = set()
    unique_sequences = []
    for seq in sequences:
        seq_tuple = tuple(seq)
        if seq_tuple not in seen:
            seen.add(seq_tuple)
            unique_sequences.append(seq)
    return unique_sequences




# Build sequences from unique paths and sequence-to-tree mapping simultaneously
unique_path_sequences = []
sequence_to_trees = {}  # Map from sequence tuple (raw sequence) to set of tree indices
for row in unique_paths_rows.values():
    try:
        nodes = [int(n) for n in json.loads(row["path_nodes"])]
    except Exception:
        nodes = []
    seq = nodes.copy()
    unique_path_sequences.append(seq)
    
    # Build sequence-to-tree mapping
    # Get the path_key (tuple of node IDs) to look up tree indices
    path_key = tuple(nodes)
    seq_tuple = tuple(seq)
    if path_key in path_key_to_trees:
        if seq_tuple in sequence_to_trees:
            # Merge tree sets if sequence appears multiple times (shouldn't happen after dedup, but be safe)
            sequence_to_trees[seq_tuple].update(path_key_to_trees[path_key])
        else:
            sequence_to_trees[seq_tuple] = path_key_to_trees[path_key].copy()
    else:
        # This shouldn't happen, but handle gracefully
        if seq_tuple not in sequence_to_trees:
            sequence_to_trees[seq_tuple] = set()

unique_path_sequences = deduplicate_sequences(unique_path_sequences)
    
def build_vocab_from_sequences(sequences):
    """
    Build vocabularies from the observed raw tokens in the sequences.
    Sequences contain only node IDs.
    Assign contiguous indices to unique node tokens (after reserved special tokens).
    """
    node_set = set()
    for seq in sequences:
        for token in seq:
            node_set.add(int(token))
    # Sort tokens to ensure reproducibility
    sorted_nodes = sorted(node_set)

    # Assign node tokens
    current_index = SPECIAL_OFFSET
    node_id_tokens = {}
    for n in sorted_nodes:
        key = f"node_{n}"
        node_id_tokens[key] = current_index
        current_index += 1

    vocab = {
        "special": SPECIAL_TOKENS,
        "node_id_tokens": node_id_tokens,
    }
    return vocab

vocab = build_vocab_from_sequences(unique_path_sequences)

def split_unique_sequences(sequences, train_frac=0.9):
    """
    Split the unique sequences (already deduplicated) into train and validation.
    """
    random.seed(RANDOM_SEED)
    random.shuffle(sequences)
    n = len(sequences)
    train_end = int(train_frac * n)
    train_seq_local = sequences[:train_end]
    val_seq_local = sequences[train_end:]
    return train_seq_local, val_seq_local

train_seq, val_seq = split_unique_sequences(unique_path_sequences, train_frac=TRAIN_FRAC)


train_dir = os.path.join(out_dir, "train")
val_dir = os.path.join(out_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Save train and val data
train_data = {
    "raw_sequences": train_seq,
    "vocab": vocab,
}
val_data = {
    "raw_sequences": val_seq,
    "vocab": vocab,
}

train_path = os.path.join(train_dir, "raw_train_sequences_and_vocab.pkl")
val_path = os.path.join(val_dir, "raw_val_sequences_and_vocab.pkl")

with open(train_path, "wb") as f:
    pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(val_path, "wb") as f:
    pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
# Save the vocabulary for reference
vocab_path = os.path.join(out_dir, "vocab.pkl")
with open(vocab_path, "wb") as f:
    pickle.dump(vocab, f)

# Save the sequence-to-tree mapping
sequence_to_trees_path = os.path.join(out_dir, "sequence_to_tree_indices.pkl")
with open(sequence_to_trees_path, "wb") as f:
    pickle.dump(sequence_to_trees, f, protocol=pickle.HIGHEST_PROTOCOL)