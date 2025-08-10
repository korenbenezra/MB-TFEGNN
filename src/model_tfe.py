# src/model_tfe.py
from __future__ import annotations

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse helpers from MB-TFE impl to avoid duplication
from .model_mbtfe import _cheb_basis, _bessel_coeffs_heat, X0


class TFEConv(nn.Module):
    """
    Lean TFE-style block:
      - Chebyshev basis Psi_k = T_k(L_hat) X
      - LP(τ_lp) = sum_k a_k(τ_lp) Psi_k
      - HP proxy = X - LP(τ_hp)
      - Project each branch (and optionally the raw skip) then fuse:
          * fusion='concat': concat([H_lp, H_hp, X]) -> Linear -> out_dim
          * fusion='sum'   : proj_lp + proj_hp + proj_skip -> out_dim
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: int,
        tau_lp: float = 0.5,
        tau_hp: float = 1.5,
        dropout: float = 0.0,
        fusion: str = "concat",           # 'concat' or 'sum'
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert in_dim > 0 and out_dim > 0 and K >= 0
        assert fusion in {"concat", "sum"}

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = int(K)
        self.tau_lp = float(tau_lp)
        self.tau_hp = float(tau_hp)
        self.fusion = fusion
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = activation if activation is not None else nn.ReLU(inplace=True)

        # Branch projections
        self.proj_lp = nn.Linear(in_dim, out_dim, bias=True)
        self.proj_hp = nn.Linear(in_dim, out_dim, bias=True)

        if self.fusion == "concat":
            # concat([H_lp, H_hp, X]) -> out_dim
            self.fuse = nn.Linear(out_dim + out_dim + in_dim, out_dim, bias=True)
            self.proj_skip = None
        else:
            # sum of projected branches, include a linear skip to match dims
            self.proj_skip = nn.Linear(in_dim, out_dim, bias=True)
            self.fuse = None  # not used

    def forward(
        self,
        X: torch.Tensor,       # [n, F_in]
        L_hat: torch.Tensor,   # sparse COO [n, n]
        need_aux: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        device, dtype = X.device, X.dtype

        # 1) Chebyshev basis
        Psi = _cheb_basis(L_hat, X, self.K)  # list len K+1

        # 2) LP and HP signals via heat kernel
        # LP(τ) = Σ a_k(τ) Psi_k ; HP ≈ X - LP(τ_hp)
        a_lp = _bessel_coeffs_heat(torch.tensor([self.tau_lp], device=device, dtype=dtype), self.K, device=device, dtype=dtype)
        a_hp = _bessel_coeffs_heat(torch.tensor([self.tau_hp], device=device, dtype=dtype), self.K, device=device, dtype=dtype)

        LP = torch.zeros_like(X)
        HPc = torch.zeros_like(X)  # complement LP(τ_hp); we will do X - HPc
        for k in range(self.K + 1):
            LP  = LP  + a_lp[k] * Psi[k]
            HPc = HPc + a_hp[k] * Psi[k]
        HP = X - HPc

        # 3) Per-branch linear + activation
        H_lp = self.act(self.proj_lp(self.dropout(LP)))
        H_hp = self.act(self.proj_hp(self.dropout(HP)))

        # 4) Fusion
        if self.fusion == "concat":
            U = torch.cat([H_lp, H_hp, X], dim=1)  # [n, 2*out + in]
            Z = self.fuse(self.dropout(U))
        else:
            Z = self.dropout(H_lp + H_hp + self.proj_skip(X))

        if not need_aux:
            return Z

        with torch.no_grad():
            aux = {
                "lp_energy": float((H_lp.pow(2).sum(dim=1).mean()).item()),
                "hp_energy": float((H_hp.pow(2).sum(dim=1).mean()).item()),
            }
        return Z, aux


class TFEModel(nn.Module):
    """
    Small TFE network with 1..N blocks + classifier.
    Matches the MBTFEModel API (accepts need_aux, returns (logits, aux) when requested).
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        K: int,
        tau_lp: float = 0.5,
        tau_hp: float = 1.5,
        n_layers: int = 1,
        dropout: float = 0.5,
        fusion: str = "concat",
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert n_layers >= 1
        act = activation if activation is not None else nn.ReLU(inplace=True)

        layers: List[TFEConv] = []
        if n_layers == 1:
            layers.append(TFEConv(in_dim, hidden, K=K, tau_lp=tau_lp, tau_hp=tau_hp,
                                  dropout=dropout, fusion=fusion, activation=act))
        else:
            layers.append(TFEConv(in_dim, hidden, K=K, tau_lp=tau_lp, tau_hp=tau_hp,
                                  dropout=dropout, fusion=fusion, activation=act))
            for _ in range(n_layers - 1):
                layers.append(TFEConv(hidden, hidden, K=K, tau_lp=tau_lp, tau_hp=tau_hp,
                                      dropout=dropout, fusion=fusion, activation=act))

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        X: torch.Tensor,
        L_hat: torch.Tensor,
        need_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, any]]:
        aux_all = {"layers": []} if need_aux else None

        H = X
        for lyr in self.layers:
            if need_aux:
                H, aux = lyr(H, L_hat, need_aux=True)
                aux_all["layers"].append(aux)
            else:
                H = lyr(H, L_hat, need_aux=False)

        logits = self.classifier(self.dropout(H))
        if not need_aux:
            return logits
        return logits, aux_all
