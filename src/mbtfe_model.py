# src/model_mbtfe.py
from __future__ import annotations

import torch
import torch.nn as nn

from mbtfe_conv import MBTFEConv
from mbtfe_helper import X0

class MBTFEModel(nn.Module):
    """
    A simple 1–N layer MB-TFE network:
        - (L-1) hidden MBTFEConv blocks (hidden->hidden)
        - 1 final MBTFEConv to hidden (if L==1, we still have one block)
        - classifier Linear(hidden -> out_dim)

    Forward expects (X, L_hat). Returns logits or (logits, aux) if need_aux=True.
    Diversity loss across layers can be obtained via .diversity_loss_total (last forward).
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, K: int, taus: list[float], n_layers: int = 1,
                    dropout: float = 0.5, lambda_div: float = 0.0, activation: nn.Module|None = None):
        
        super().__init__()
        assert n_layers >= 1
        act = activation if activation is not None else nn.ReLU(inplace=True)

        layers: list[MBTFEConv] = []
        if n_layers == 1:
            layers.append(
                MBTFEConv(in_dim, hidden, K=K, taus=taus, dropout=dropout,
                            lambda_div=lambda_div, activation=act, use_diversity=True)
            )
        else:
            layers.append(
                MBTFEConv(in_dim, hidden, K=K, taus=taus, dropout=dropout,
                            lambda_div=lambda_div, activation=act, use_diversity=True)
            )
            for _ in range(n_layers - 1):
                layers.append(
                    MBTFEConv(hidden, hidden, K=K, taus=taus, dropout=dropout,
                                lambda_div=lambda_div, activation=act, use_diversity=True)
                )

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)
        self._last_div_total = torch.tensor(0.0)

    def forward(
        self,
        X: torch.Tensor,
        L_hat: torch.Tensor,
        need_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        aux_all = {"layers": []}
        div_total = X0(X)

        H = X
        for i, layer in enumerate(self.layers):
            if need_aux:
                H, aux = layer(H, L_hat, need_aux=True)
                aux_all["layers"].append(aux)
            else:
                H = layer(H, L_hat, need_aux=False)
            div_total = div_total + layer.diversity_loss()

        logits = self.classifier(self.dropout(H))
        self._last_div_total = div_total.detach()

        if not need_aux:
            return logits

        aux_all["diversity_loss_total"] = float(self._last_div_total.item())
        return logits, aux_all

    def diversity_loss_total(self) -> torch.Tensor:
        return self._last_div_total
