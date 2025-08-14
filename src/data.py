# src/data.py
from __future__ import annotations

import os
import json
import math
import time
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
from torch import Tensor

# Optional dependency: PyTorch Geometric
try:
    from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor
    from torch_geometric.utils import to_undirected, add_self_loops as pyg_add_self_loops
    _HAVE_PYG = True
except Exception:
    _HAVE_PYG = False


# ----------------------------
# Public API
# ----------------------------

def load_dataset(
    name: str,
    root: str = "data",
    split_id: Optional[int] = None,
    n_random_splits: int = 10,
    add_loops: bool = True,
    row_norm: bool = True,
    cache: bool = True,
    seed: int = 1,
) -> Dict[str, object]:
    """
    Load a graph dataset and return a dict with:
      X          : FloatTensor [n, F]
      y          : LongTensor  [n]
      train_mask : BoolTensor  [n]
      val_mask   : BoolTensor  [n]
      test_mask  : BoolTensor  [n]
      L_sym      : Sparse COO  [n, n]  (I - D^{-1/2} A D^{-1/2})
      L_hat      : Sparse COO  [n, n]  (L_sym - I)
      meta       : dict(name, n, num_classes)

    Args:
      name: one of {"cora","citeseer","pubmed","chameleon","squirrel",
                    "cornell","texas","wisconsin","actor"}
      split_id: if None, use public split (if available), else pick random split index
      n_random_splits: how many stratified random splits to precompute (when needed)
      add_loops: add self-loops once to adjacency before Laplacian (idempotent)
      row_norm: row-normalize features X (sum=1 per node) if True
      cache: cache processed tensors to data/processed
      seed: global seed for random splits

    Notes:
      - Requires PyTorch Geometric for dataset downloads.
      - Caches are per (name, add_loops, row_norm).
    """
    if not _HAVE_PYG:
        raise ImportError(
            "PyTorch Geometric is required for dataset loading.\n"
            "Install with: pip install torch-geometric torch-sparse torch-scatter "
            "and follow the official installation guide for your platform."
        )

    name = name.lower().strip()
    root = str(root)
    os.makedirs(root, exist_ok=True)

    # Compose cache key that depends on name + preprocessing flags
    cache_key = f"{name}|loops={int(add_loops)}|rownorm={int(row_norm)}"
    cache_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:8]
    proc_path = Path(root) / "processed" / f"{name}_{cache_hash}.pt"
    splits_path = Path(root) / "splits" / f"{name}_splits.pt"
    os.makedirs(proc_path.parent, exist_ok=True)
    os.makedirs(splits_path.parent, exist_ok=True)

    # Try cache
    if cache and Path(proc_path).exists():
        obj = torch.load(proc_path, map_location="cpu", weights_only=True)
        X = obj["X"]
        y = obj["y"]
        L_sym = obj["L_sym"]
        L_hat = obj["L_hat"]
        meta = obj["meta"]
        # Splits: use requested logic below (public if present, else from splits file)
        train_mask, val_mask, test_mask = _resolve_splits(
            name=name,
            y=y,
            splits_path=splits_path,
            split_id=split_id,
            n_random_splits=n_random_splits,
            seed=seed,
            public_masks=obj.get("public_masks", None),
        )
        return {
            "X": X, "y": y,
            "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
            "L_sym": L_sym.coalesce(), "L_hat": L_hat.coalesce(),
            "meta": meta,
        }

    # Otherwise: load raw via PyG
    X, y, edge_index, public_masks, meta = _load_with_pyg(name=name, root=root)

    # Row-normalize features if requested
    if row_norm:
        X = _row_normalize_features(X)

    # Ensure undirected graph
    edge_index = to_undirected(edge_index, num_nodes=meta["n"])

    # Add self-loops (idempotent)
    if add_loops:
        edge_index, _ = pyg_add_self_loops(edge_index, num_nodes=meta["n"])

    # Build L_sym and L_hat
    L_sym, L_hat = build_laplacian(edge_index=edge_index, num_nodes=meta["n"])

    # Save processed cache
    if cache:
        torch.save(
            {
                "X": X, "y": y,
                "L_sym": L_sym.coalesce(), "L_hat": L_hat.coalesce(),
                "meta": meta,
                "public_masks": public_masks,  # may be None
                "created_at": time.time(),
                "preproc": {"add_loops": add_loops, "row_norm": row_norm},
            },
            proc_path,
        )

    # Resolve splits (public if available; else create/load random)
    train_mask, val_mask, test_mask = _resolve_splits(
        name=name,
        y=y,
        splits_path=splits_path,
        split_id=split_id,
        n_random_splits=n_random_splits,
        seed=seed,
        public_masks=public_masks,
    )

    return {
        "X": X, "y": y,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
        "L_sym": L_sym.coalesce(), "L_hat": L_hat.coalesce(),
        "meta": meta,
    }


# ----------------------------
# Loaders (PyG-based)
# ----------------------------

def _load_with_pyg(name: str, root: str) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]], Dict]:
    """
    Return:
      X: [n,F], y: [n], edge_index: [2,E],
      public_masks: dict(train/val/test BoolTensor) or None,
      meta: dict(name, n, num_classes)
    """
    ds_name = name.lower()
    if ds_name in {"cora", "citeseer", "pubmed"}:
        dataset = Planetoid(root=os.path.join(root, "raw", "Planetoid"), name=ds_name.capitalize())
        data = dataset[0]
        public_masks = {
            "train_mask": data.train_mask.bool(),
            "val_mask": data.val_mask.bool(),
            "test_mask": data.test_mask.bool(),
        }
    elif ds_name in {"chameleon", "squirrel"}:
        dataset = WikipediaNetwork(root=os.path.join(root, "raw", "WikipediaNetwork"), name=ds_name.capitalize())
        data = dataset[0]
        # Some WikipediaNetwork variants come without masks -> treat as None
        public_masks = None
        if hasattr(data, "train_mask") and data.train_mask is not None:
            public_masks = {
                "train_mask": data.train_mask.bool(),
                "val_mask": data.val_mask.bool(),
                "test_mask": data.test_mask.bool(),
            }
    elif ds_name in {"cornell", "texas", "wisconsin"}:
        dataset = WebKB(root=os.path.join(root, "raw", "WebKB"), name=ds_name.capitalize())
        data = dataset[0]
        public_masks = None
        if hasattr(data, "train_mask") and data.train_mask is not None:
            public_masks = {
                "train_mask": data.train_mask.bool(),
                "val_mask": data.val_mask.bool(),
                "test_mask": data.test_mask.bool(),
            }
    elif ds_name in {"actor"}:
        dataset = Actor(root=os.path.join(root, "raw", "Actor"))
        data = dataset[0]
        public_masks = None
        if hasattr(data, "train_mask") and data.train_mask is not None:
            public_masks = {
                "train_mask": data.train_mask.bool(),
                "val_mask": data.val_mask.bool(),
                "test_mask": data.test_mask.bool(),
            }
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    X = data.x.float()
    y = data.y.long()
    edge_index = data.edge_index.long()
    n = X.size(0)
    num_classes = int(y.max().item() + 1) if y.numel() > 0 else 0

    meta = {"name": name, "n": n, "num_classes": num_classes}
    return X, y, edge_index, public_masks, meta


# ----------------------------
# Laplacian builders
# ----------------------------

def build_laplacian(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
    """
    Build normalized Laplacian L_sym and L_hat = L_sym - I as sparse COO tensors.

    L_sym = I - D^{-1/2} A D^{-1/2}
    L_hat = L_sym - I

    Args:
      edge_index: LongTensor [2, E] (assumed undirected; self-loops allowed)
      num_nodes : int
    """
    A = _edge_index_to_sparse_adj(edge_index, num_nodes)  # COO
    A = A.coalesce()
    n = num_nodes

    # Degree (sum over rows)
    deg = torch.sparse.sum(A, dim=1).to_dense()  # [n]
    # Handle isolated nodes: set deg=1 to avoid division by zero (they behave like self-only)
    deg_safe = deg.clone()
    deg_safe[deg_safe == 0] = 1.0
    d_inv_sqrt = torch.pow(deg_safe, -0.5)

    # Normalize values in A: val' = val * d^{-1/2}_i * d^{-1/2}_j
    row, col = A.indices()
    val = A.values()
    val_norm = val * d_inv_sqrt[row] * d_inv_sqrt[col]
    A_norm = torch.sparse_coo_tensor(indices=torch.stack([row, col], dim=0),
                                     values=val_norm,
                                     size=(n, n)).coalesce()

    # L_sym = I - A_norm
    I = torch.sparse_coo_tensor(
        indices=torch.arange(n, dtype=torch.long).repeat(2, 1),
        values=torch.ones(n, dtype=val_norm.dtype),
        size=(n, n)
    ).coalesce()

    # Subtract sparse matrices: (I - A_norm)
    L_sym = _sparse_sub(I, A_norm).coalesce()
    # L_hat = L_sym - I
    L_hat = _sparse_sub(L_sym, I).coalesce()

    return L_sym, L_hat


# ----------------------------
# Splits
# ----------------------------

def _resolve_splits(
    name: str,
    y: Tensor,
    splits_path: Path,
    split_id: Optional[int],
    n_random_splits: int,
    seed: int,
    public_masks: Optional[Dict[str, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Choose public masks if available and split_id is None; else use/create random splits."""
    if public_masks is not None and split_id is None:
        # Handle the case where masks are 2D tensors [n, splits] - take the first split (column)
        train_mask = public_masks["train_mask"]
        val_mask = public_masks["val_mask"]
        test_mask = public_masks["test_mask"]
        
        # Convert 2D masks to 1D if needed
        if train_mask.dim() > 1:
            train_mask = train_mask[:, 0]
        if val_mask.dim() > 1:
            val_mask = val_mask[:, 0]
        if test_mask.dim() > 1:
            test_mask = test_mask[:, 0]
            
        return train_mask, val_mask, test_mask

    # Load or create random splits file
    if splits_path.exists():
        saved = torch.load(splits_path, map_location="cpu", weights_only=True)
        splits = saved.get("splits", None)
        if not splits:
            splits = _make_and_save_random_splits(y, splits_path, n_random_splits, seed)
    else:
        splits = _make_and_save_random_splits(y, splits_path, n_random_splits, seed)

    # Pick split by index (default 0)
    idx = 0 if split_id is None else int(split_id)
    if not (0 <= idx < len(splits)):
        raise ValueError(f"split_id {idx} out of range (have {len(splits)} splits)")

    s = splits[idx]
    return s["train_mask"], s["val_mask"], s["test_mask"]


def _make_and_save_random_splits(
    y: Tensor,
    splits_path: Path,
    n_random_splits: int,
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> List[Dict[str, Tensor]]:
    torch.manual_seed(seed)
    n = y.size(0)
    num_classes = int(y.max().item() + 1)
    indices_by_class = [ (y == c).nonzero(as_tuple=False).flatten().tolist() for c in range(num_classes) ]

    splits: List[Dict[str, Tensor]] = []
    for sidx in range(n_random_splits):
        # Stratified per class
        train_idx, val_idx, test_idx = [], [], []
        for cls_indices in indices_by_class:
            cls_indices = _shuffled(cls_indices, seed + sidx)
            n_c = len(cls_indices)
            n_tr = int(round(train_ratio * n_c))
            n_va = int(round(val_ratio * n_c))
            n_te = n_c - n_tr - n_va
            # Edge cases
            n_tr = max(n_tr, 1 if n_c >= 1 else 0)
            n_va = max(n_va, 1 if n_c >= 2 else 0)
            n_te = max(n_te, 0)
            # Adjust if overflow
            while n_tr + n_va + n_te > n_c:
                n_te = max(0, n_te - 1)
            # Partition
            cls_tr = cls_indices[:n_tr]
            cls_va = cls_indices[n_tr:n_tr+n_va]
            cls_te = cls_indices[n_tr+n_va:n_tr+n_va+n_te]
            train_idx += cls_tr
            val_idx += cls_va
            test_idx += cls_te

        train_mask = _mask_from_indices(train_idx, n)
        val_mask   = _mask_from_indices(val_idx, n)
        test_mask  = _mask_from_indices(test_idx, n)
        splits.append({"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask})

    torch.save({"splits": splits, "created_at": time.time(), "seed": seed}, splits_path)
    return splits


# ----------------------------
# Utilities (internal)
# ----------------------------

def _edge_index_to_sparse_adj(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Build a (coalesced) sparse COO adjacency with unit weights from edge_index."""
    edge_index = edge_index.long()
    row, col = edge_index[0], edge_index[1]
    values = torch.ones(row.numel(), dtype=torch.float32)
    A = torch.sparse_coo_tensor(indices=torch.stack([row, col], dim=0),
                                values=values,
                                size=(num_nodes, num_nodes))
    return A.coalesce()


def _sparse_sub(A: Tensor, B: Tensor) -> Tensor:
    """Compute sparse A - B (COO)."""
    A = A.coalesce()
    B = B.coalesce()
    indices = torch.cat([A.indices(), B.indices()], dim=1)
    values  = torch.cat([A.values(), -B.values()], dim=0)
    out = torch.sparse_coo_tensor(indices=indices, values=values, size=A.size())
    return out.coalesce()


def _mask_from_indices(indices: List[int], n: int) -> Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    if len(indices) > 0:
        mask[torch.tensor(indices, dtype=torch.long)] = True
    return mask


def _shuffled(xs: List[int], seed: int) -> List[int]:
    g = torch.Generator()
    g.manual_seed(seed)
    if len(xs) == 0:
        return []
    idx = torch.randperm(len(xs), generator=g).tolist()
    return [xs[i] for i in idx]


def _row_normalize_features(X: Tensor, eps: float = 1e-9) -> Tensor:
    """Row-normalize features so each node's feature vector sums to 1 (if nonzero)."""
    if X.dim() != 2:
        return X
    row_sum = X.abs().sum(dim=1, keepdim=True)  # L1 norm
    scale = torch.where(row_sum > eps, 1.0 / row_sum, torch.zeros_like(row_sum))
    return X * scale
