# src/metrics.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

# For rebuilding Laplacians after edge dropout
from . import data as data_mod


# ---------------------------
# Core classification metrics
# ---------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Top-1 accuracy on a masked set.
    logits: [n, C], y: [n], mask: [n] bool
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    correct = (pred[mask] == y[mask]).sum().item()
    total = int(mask.sum().item())
    return float(correct) / max(1, total)


def macro_f1(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Macro-averaged F1 (unweighted mean across classes) on a masked set.
    Pure torch (no sklearn).
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    y_m = y[mask]
    p_m = pred[mask]

    C = int(y.max().item() + 1) if y.numel() > 0 else 0
    if C == 0:
        return 0.0

    eps = 1e-12
    f1s = []
    for c in range(C):
        tp = ((p_m == c) & (y_m == c)).sum().float()
        fp = ((p_m == c) & (y_m != c)).sum().float()
        fn = ((p_m != c) & (y_m == c)).sum().float()
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        # Only count classes that appear in mask at least once
        if (y_m == c).any():
            f1s.append(f1)
    if not f1s:
        return 0.0
    return float(torch.stack(f1s).mean().item())


def ece(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected Calibration Error on a masked set.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return 0.0

    with torch.no_grad():
        probs = F.softmax(logits[mask], dim=1)
        conf, pred = probs.max(dim=1)
        y_true = y[mask]

        bins = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
        ece_val = 0.0
        N = conf.numel()
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            sel = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
            M = sel.sum().item()
            if M == 0:
                continue
            acc_b = (pred[sel] == y_true[sel]).float().mean().item()
            conf_b = conf[sel].float().mean().item()
            ece_val += (M / N) * abs(acc_b - conf_b)
        return float(ece_val)


# ---------------------------
# Unified evaluation
# ---------------------------

def evaluate(
    model,
    X: torch.Tensor,
    L_hat: torch.Tensor,
    y: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    device: torch.device,
    need_aux: bool = False,
) -> Dict[str, Any]:
    """
    Compute accuracy/macro-F1 (and ECE if you want to add it) on train/val/test.
    If need_aux=True, also returns model-provided aux diagnostics under "aux".
    """
    model.eval()
    with torch.no_grad():
        out = model(X, L_hat, need_aux=need_aux)
        if need_aux:
            logits, aux = out
        else:
            logits, aux = out, None

    res: Dict[str, Any] = {}

    # Train/Val/Test metrics (presence is optional but we expect these keys)
    train_m = masks.get("train", None)
    val_m   = masks.get("val", None)
    test_m  = masks.get("test", None)

    if train_m is not None:
        res["acc_train"] = accuracy(logits, y, train_m)
        res["macro_f1_train"] = macro_f1(logits, y, train_m)
        # res["ece_train"] = ece(logits, y, train_m)  # optional

    if val_m is not None:
        res["acc_val"] = accuracy(logits, y, val_m)
        res["macro_f1_val"] = macro_f1(logits, y, val_m)
        # res["ece_val"] = ece(logits, y, val_m)  # optional

    if test_m is not None:
        res["acc_test"] = accuracy(logits, y, test_m)
        res["macro_f1_test"] = macro_f1(logits, y, test_m)
        # res["ece_test"] = ece(logits, y, test_m)  # optional

    if need_aux and aux is not None:
        res["aux"] = aux
    return res


# ---------------------------
# Robustness evaluations
# ---------------------------

def feature_noise_eval(
    model,
    X: torch.Tensor,
    L_hat: torch.Tensor,
    y: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    sigmas: List[float],
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate robustness to additive Gaussian feature noise at test time.
    Returns a list of dicts with {"sigma": s, "acc_test": ...}.
    """
    device = device or X.device
    test_mask = masks.get("test", None)
    if test_mask is None:
        return []

    model.eval()
    out = []
    for s in sigmas:
        Xp = X + torch.randn_like(X) * float(s)
        with torch.no_grad():
            logits = model(Xp, L_hat, need_aux=False)
        out.append({"sigma": float(s), "acc_test": accuracy(logits, y, test_mask)})
    return out


def edge_dropout_eval(
    model,
    X: torch.Tensor,
    L_sym: torch.Tensor,
    y: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    ps: List[float],
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate robustness to random undirected edge dropout at test time.
    Steps:
      1) Recover an undirected edge list from L_sym (off-diagonal entries only).
      2) Sample a subset of undirected edges to keep (prob = 1-p).
      3) Reconstruct a symmetric edge_index (both directions) + add self-loops.
      4) Build new L_sym', L_hat' via data.build_laplacian and evaluate.
    """
    device = device or X.device
    test_mask = masks.get("test", None)
    if test_mask is None:
        return []

    n = L_sym.size(0)
    undirected_pairs = _undirected_edge_pairs_from_Lsym(L_sym)  # shape [2, M], i<j guaranteed
    if undirected_pairs.numel() == 0:
        return [{"p": float(p), "acc_test": 0.0} for p in ps]

    results = []
    model.eval()
    for p in ps:
        keep_prob = max(0.0, min(1.0, 1.0 - float(p)))
        M = undirected_pairs.size(1)
        # Bernoulli keep per undirected edge
        keep_mask = (torch.rand(M, device=undirected_pairs.device) < keep_prob)
        kept = undirected_pairs[:, keep_mask]  # [2, M_keep]

        # Rebuild symmetric edge_index (both directions) + self-loops
        if kept.numel() == 0:
            # Graph collapses to self-loops only
            ei = _identity_edge_index(n, device=kept.device)
        else:
            ei_dir = torch.cat([kept, kept.flip(0)], dim=1)  # [2, 2*M_keep]
            ei = torch.cat([ei_dir, _identity_edge_index(n, device=kept.device)], dim=1)

        # Laplacian
        L_sym_p, L_hat_p = data_mod.build_laplacian(ei.long().cpu(), num_nodes=n)
        L_hat_p = L_hat_p.to(device)

        with torch.no_grad():
            logits = model(X, L_hat_p, need_aux=False)
            acc = accuracy(logits, y, test_mask)
        results.append({"p": float(p), "acc_test": acc})

    return results


# ---------------------------
# Sparse helpers (no PyG)
# ---------------------------

def _undirected_edge_pairs_from_Lsym(L_sym: torch.Tensor) -> torch.Tensor:
    """
    Get a unique undirected edge list (i<j) from L_sym by inspecting off-diagonal entries.
    Returns LongTensor [2, M] where each column is (i, j) with i<j.
    """
    assert L_sym.is_sparse, "L_sym must be a sparse COO tensor."
    L_sym = L_sym.coalesce()
    idx = L_sym.indices()
    row, col = idx[0], idx[1]
    n = L_sym.size(0)

    # Exclude diagonal
    off = row != col
    if off.sum() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=idx.device)

    row = row[off]
    col = col[off]
    # Reduce to undirected keys by sorting endpoints
    i_min = torch.minimum(row, col)
    i_max = torch.maximum(row, col)
    key = i_min * n + i_max  # unique key per undirected pair

    uniq, first_idx = torch.unique(key, return_index=True)
    i = i_min[first_idx].long()
    j = i_max[first_idx].long()
    return torch.stack([i, j], dim=0)  # [2, M]


def _identity_edge_index(n: int, device=None) -> torch.Tensor:
    """Return [2, n] edge_index of self-loops (i, i)."""
    ar = torch.arange(n, device=device, dtype=torch.long)
    return torch.stack([ar, ar], dim=0)
