
import torch
import torch.nn as nn

from mbtfe_helper import _cheb_basis, _bessel_coeffs_heat, X0

class MBTFEConv(nn.Module):
    """
    Multi-Band TFE convolutional block:
        - Chebyshev basis Psi_k = T_k(L_hat) X
        - Heat-kernel smoothings Y(tau_i) = sum_k a_k(tau_i) Psi_k
        - Bands: Band_i = Y(tau_{i-1}) - Y(tau_i)  (i=1..m), UltraLow = Y(tau_m)
        - Per-band 1x1 linear + activation
        - Fusion: concat([H_1..H_m, H_0, X]) -> linear
        - Optional diversity loss over band features
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: int,
        taus: list[float],
        dropout: float = 0.0,
        lambda_div: float = 0.0,
        activation: nn.Module | None = None,
        use_diversity: bool = True,
    ):
        super().__init__()
        assert in_dim > 0 and out_dim > 0 and K >= 0
        assert len(taus) >= 1, "Provide at least one tau (e.g., [0.5, 1.5, 4.0])."

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = int(K)
        self.m = len(taus)                     # number of 'intermediate' smoothings
        self.taus = [float(t) for t in taus]   # strictly increasing recommended
        self.lambda_div = float(lambda_div)
        self.use_diversity = bool(use_diversity)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = activation if activation is not None else nn.ReLU(inplace=True)

        # Per-band linear projections: H_i = sigma(Band_i W_i + b_i)
        # We have m "diff bands" plus 1 "UltraLow" band => (m+1) heads
        self.band_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=True) for _ in range(self.m + 1)]
        )

        # Fusion layer: concat([H_1..H_m, H_0, X]) -> out_dim
        fusion_input_dim = self.m * out_dim + out_dim + in_dim  # (m bands + UltraLow) + skip X
        self.fuse = nn.Linear(fusion_input_dim, out_dim, bias=True)

        # Buffers for diagnostics
        self._last_div_loss: torch.Tensor = torch.tensor(0.0)
        self._last_band_energy: list[float] | None = None
        self._last_telescoping_residual: float | None = None

    @torch.no_grad()
    def _telescoping_residual(self, X: torch.Tensor, bands: list[torch.Tensor], ultralow: torch.Tensor) -> float:
        rec = ultralow + torch.stack(bands, dim=0).sum(dim=0)
        num = torch.norm(rec - X, p='fro')
        den = torch.norm(X, p='fro') + 1e-12
        return float((num / den).item())

    def _compute_diversity(self, H_list: list[torch.Tensor]) -> torch.Tensor:
        if not self.use_diversity or self.lambda_div <= 0 or len(H_list) <= 1:
            return X0(H_list[0]).sum() * 0.0 if H_list else torch.tensor(0.0, device=self.fuse.weight.device)
        # Row-normalize features
        # Stack excluding the raw skip; H_list are band activations (m+1 tensors)
        sims = []
        for i in range(len(H_list)):
            Hi = H_list[i]
            Hi = Hi / (Hi.norm(p=2, dim=1, keepdim=True) + 1e-12)
            for j in range(i + 1, len(H_list)):
                Hj = H_list[j]
                Hj = Hj / (Hj.norm(p=2, dim=1, keepdim=True) + 1e-12)
                # Frobenius norm of cross-correlation
                s = torch.matmul(Hi.t(), Hj)
                sims.append((s * s).sum())
        if not sims:
            return torch.tensor(0.0, device=self.fuse.weight.device)
        return self.lambda_div * torch.stack(sims).sum()

    def forward(
        self,
        X: torch.Tensor,           # [n, F_in]
        L_hat: torch.Tensor,       # sparse COO [n, n]
        need_aux: bool = False,    # if True, return (Z, aux)
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        device = X.device
        dtype = X.dtype

        # 1) Chebyshev basis
        Psi = _cheb_basis(L_hat, X, self.K)  # list len K+1, each [n, F_in]

        # 2) Heat-kernel coefficients for tau_0=0 plus provided taus
        # tau_seq = [0.0, tau1, tau2, ..., taum]
        tau_seq = [0.0] + self.taus
        A = []
        for t in tau_seq:
            a = _bessel_coeffs_heat(torch.tensor([t], device=device, dtype=dtype), self.K, device=device, dtype=dtype)
            A.append(a)  # shape [K+1]
        # Stack into [m+1, K+1]
        A = torch.stack(A, dim=0)

        # 3) Smoothed signals Y(tau_i) = sum_k a_k(tau_i) Psi_k
        # Accumulate efficiently without stacking Psi (memory-friendly)
        Y = [torch.zeros_like(X) for _ in range(self.m + 1)]  # i=0..m
        for k in range(self.K + 1):
            coeffs = A[:, k].view(-1, 1, 1)  # [m+1,1,1]
            # broadcast over nodes and features
            contrib = coeffs * Psi[k].unsqueeze(0)  # [m+1, n, F]
            for i in range(self.m + 1):
                Y[i] = Y[i] + contrib[i]

        # 4) Bands: Band_i = Y(tau_{i-1}) - Y(tau_i), i=1..m; UltraLow = Y(tau_m)
        bands = []
        for i in range(1, self.m + 1):
            bands.append(Y[i - 1] - Y[i])  # [n, F_in]
        ultra = Y[self.m]  # [n, F_in]

        # 5) Per-band linear + activation
        H_list: list[torch.Tensor] = []
        for i, B in enumerate(bands):            # H_1..H_m
            H = self.band_linears[i](self.dropout(B))
            H = self.act(H)
            H_list.append(H)
        H0 = self.band_linears[-1](self.dropout(ultra))  # UltraLow head
        H0 = self.act(H0)
        H_list_with_ultra = H_list + [H0]

        # 6) Fusion: concat([H_1..H_m, H_0, X]) -> out_dim
        U = torch.cat(H_list_with_ultra + [X], dim=1)  # [n, m*out + out + in]
        Z = self.fuse(self.dropout(U))

        # 7) Optional diversity loss
        div_loss = self._compute_diversity(H_list_with_ultra)
        self._last_div_loss = div_loss.detach()

        if not need_aux:
            return Z

        # Diagnostics (no grad)
        with torch.no_grad():
            band_energy = []
            for H in H_list_with_ultra:
                # mean squared L2 per node
                e = (H.pow(2).sum(dim=1).mean()).item()
                band_energy.append(e)
            res = self._telescoping_residual(X, bands, ultra)
            self._last_band_energy = band_energy
            self._last_telescoping_residual = res

        aux = {
            "band_energy": band_energy,                # list length m+1
            "div_loss": float(self._last_div_loss.item()),
            "telescoping_residual": self._last_telescoping_residual,
        }
        return Z, aux

    def diversity_loss(self) -> torch.Tensor:
        # Expose last computed diversity loss (0 if not used)
        return self._last_div_loss


