import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_embeddings(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def ensure_l2_normalized(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        raise ValueError("Embeddings tensor is None")
    return F.normalize(x, p=2, dim=1)


def labels_from_metadata(saved: Dict) -> Optional[List[str]]:
    """Extract labels from saved embeddings metadata."""
    labels = saved.get("labels")
    if labels is None:
        return None
    if torch.is_tensor(labels):
        labels_list = [int(v) for v in labels.view(-1).tolist()]
    else:
        labels_list = [int(v) for v in labels]
    meta = saved.get("metadata", {}) or {}
    int_to_label = meta.get("int_to_label")
    if isinstance(int_to_label, dict):
        # Normalize keys to int
        key_to_name = {int(k): str(v) for k, v in int_to_label.items()}
        return [key_to_name.get(int(l), str(int(l))) for l in labels_list]
    return [str(int(l)) for l in labels_list]


@torch.inference_mode()
def topk_indices_streaming(
    Q: torch.Tensor,
    E: torch.Tensor,
    device: torch.device,
    k: int = 1,
    q_batch: int = 1024,
    e_chunk: int = 100000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming top-k retrieval that scales to millions of corpus vectors.
    Returns (best_scores[M, k], best_indices[M, k]) where indices are absolute positions into E.
    Results are sorted by score in descending order (best first).
    """
    M = Q.size(0)
    D = Q.size(1)
    N = E.size(0)
    assert E.size(1) == D, "Dim mismatch between Q and E"

    print(f"Starting top-{k} retrieval: {M:,} queries × {N:,} corpus vectors (dim={D})")
    print(f"Using q_batch={q_batch:,}, e_chunk={e_chunk:,}")

    best_scores_all = torch.empty(M, k, dtype=torch.float32)
    best_idx_all = torch.empty(M, k, dtype=torch.long)

    # Progress bar for query batches
    q_pbar = tqdm(total=M, desc="Query batches", unit="queries")
    
    for qs in range(0, M, q_batch):
        qe = min(qs + q_batch, M)
        batch_size = qe - qs
        q = Q[qs:qe].to(device, non_blocking=True)
        
        # Running top-k per query in the current batch
        # Initialize with -inf scores so any real score will replace them
        best_scores = torch.full((batch_size, k), float("-inf"), device=device)
        best_idx = torch.full((batch_size, k), -1, dtype=torch.long, device=device)

        # Progress bar for corpus chunks within this query batch
        e_pbar = tqdm(total=N, desc=f"Corpus chunks (queries {qs:,}-{qe:,})", unit="vectors", leave=False)
        
        for es in range(0, E.size(0), e_chunk):
            ee = min(es + e_chunk, E.size(0))
            e_chunk_tensor = E[es:ee].to(device, non_blocking=True)
            # scores shape: [batch_size, chunk_size]
            scores = q @ e_chunk_tensor.T
            
            # Get top-k from this chunk
            chunk_k = min(k, scores.size(1))
            chunk_vals, chunk_idxs = scores.topk(chunk_k, dim=1, largest=True, sorted=True)
            # Adjust indices to be absolute positions in E
            chunk_idxs = chunk_idxs + es
            
            # Merge with running top-k: concatenate and re-select top-k
            combined_scores = torch.cat([best_scores, chunk_vals], dim=1)  # [batch_size, k + chunk_k]
            combined_idx = torch.cat([best_idx, chunk_idxs], dim=1)
            
            # Select top-k from combined
            topk_vals, topk_positions = combined_scores.topk(k, dim=1, largest=True, sorted=True)
            best_scores = topk_vals
            best_idx = torch.gather(combined_idx, 1, topk_positions)
            
            e_pbar.update(ee - es)
        
        e_pbar.close()
        best_scores_all[qs:qe] = best_scores.cpu()
        best_idx_all[qs:qe] = best_idx.cpu()
        
        q_pbar.update(qe - qs)
        
        # Print progress every few batches
        if (qs // q_batch) % 10 == 0:
            print(f"Completed {qe:,}/{M:,} queries ({100*qe/M:.3f}%)")
    
    q_pbar.close()
    print(f"Top-{k} retrieval completed!")

    return best_scores_all, best_idx_all


def main():
    p = argparse.ArgumentParser(description="Streaming top-k retrieval")
    p.add_argument("--corpus", type=str, required=True, help="Path to corpus embeddings .pt")
    p.add_argument("--queries", type=str, required=True, help="Path to query embeddings .pt")
    p.add_argument("--out", type=str, required=True, help="Output path for results .pt file")
    p.add_argument("--q_batch", type=int, default=1024, help="Query batch size per step")
    p.add_argument("--e_chunk", type=int, default=100000, help="Corpus chunk size per step")
    p.add_argument("--topk", type=int, default=1, help="Number of nearest neighbors to retrieve per query (default: 1)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load embeddings
    print("Loading corpus embeddings...")
    corpus = load_embeddings(args.corpus)
    print("Loading query embeddings...")
    queries = load_embeddings(args.queries)

    # Extract labels if available (for saving with results)
    query_labels = labels_from_metadata(queries)
    corpus_labels = labels_from_metadata(corpus)

    E = ensure_l2_normalized(corpus["embeddings"])  # [N, D]
    
    Q = ensure_l2_normalized(queries["embeddings"])  # [M, D]
    print(f"Corpus: {E.shape}, Queries: {Q.shape}")

    # Streaming top-k retrieval
    k = int(args.topk)
    print(f"\nStarting streaming top-{k} retrieval...")
    best_scores, best_idx = topk_indices_streaming(Q, E, device=device, k=k, q_batch=int(args.q_batch), e_chunk=int(args.e_chunk))
    # best_scores and best_idx are [M, k] tensors

    # Save results
    print("\nSaving results...")
    output_data = {
        "topk_indices": best_idx,  # [M, k] tensor
        "topk_scores": best_scores,  # [M, k] tensor
        "top1_indices": best_idx[:, 0] if k > 1 else best_idx,  # [M] tensor for backward compat
        "top1_scores": best_scores[:, 0] if k > 1 else best_scores,  # [M] tensor for backward compat
        "k": k,
    }
    
    # Add labels if available
    if query_labels is not None:
        output_data["query_labels"] = query_labels
    if corpus_labels is not None:
        output_data["corpus_labels"] = corpus_labels
    
    out_path = args.out if args.out.endswith(".pt") else f"{args.out}.pt"
    torch.save(output_data, out_path)

if __name__ == "__main__":
    main()
