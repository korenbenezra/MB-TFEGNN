
import torch

def _cheb_basis(L_hat: torch.Tensor, X: torch.Tensor, K: int) -> list[torch.Tensor]:
    """
    Build Chebyshev basis Psi_k = T_k(L_hat) X for k=0..K.
    L_hat: sparse COO [n,n]
    X:     dense [n, F]
    Returns list of K+1 dense tensors, each [n, F].
    """
    assert L_hat.is_sparse, "L_hat must be a sparse COO tensor (n x n)."
    n, _ = L_hat.size()
    assert X.size(0) == n, "X and L_hat size mismatch."

    Psi = [X]  # Psi_0
    if K == 0:
        return Psi

    Psi_1 = torch.sparse.mm(L_hat, X)  # Psi_1 = L_hat X
    Psi.append(Psi_1)

    for k in range(2, K + 1):
        # Psi_k = 2 L_hat Psi_{k-1} - Psi_{k-2}
        LP = torch.sparse.mm(L_hat, Psi[-1])
        Psi_k = 2.0 * LP - Psi[-2]
        Psi.append(Psi_k)

    return Psi


def _bessel_coeffs_heat(tau: torch.Tensor, K: int, device=None, dtype=None) -> torch.Tensor:
    """
    Closed-form Chebyshev coefficients for f(z) = e^{-tau} * e^{-tau z}
    on z in [-1,1]:
        a_0 = e^{-tau} I_0(tau)
        a_k = 2 e^{-tau} (-1)^k I_k(tau), k>=1
    where I_k is the modified Bessel function of the first kind (order k).

    Returns: a tensor a[0..K] (shape [K+1]).
    """
    device = device if device is not None else tau.device
    dtype = dtype if dtype is not None else tau.dtype
    tau = tau.to(device=device, dtype=dtype)

    # Handle tau == 0 exactly: I_0(0)=1, I_k>0(0)=0
    if float(tau.item()) == 0.0:
        a = torch.zeros(K + 1, device=device, dtype=dtype)
        a[0] = 1.0
        return a

    # torch.special.iv(v, x): modified Bessel I_v(x)
    # Build orders 0..K as a tensor
    orders = torch.arange(0, K + 1, device=device, dtype=dtype)
    # For integer orders, iv should work; fallback if not available
    if hasattr(torch.special, "iv"):
        Ik = torch.special.iv(orders, tau)  # shape [K+1]
    else:
        # Fallback using scipy's Bessel functions
        try:
            from scipy import special as sp_special
            import numpy as np
            
            # Convert to numpy, compute with scipy, then back to torch
            tau_np = tau.cpu().numpy()
            orders_np = orders.cpu().numpy()
            
            # Compute for each order (manually since scipy doesn't broadcast like torch)
            Ik_list = []
            for k in range(K + 1):
                Ik_list.append(float(sp_special.iv(k, tau_np.item())))
            
            # Convert back to torch tensor
            Ik = torch.tensor(Ik_list, device=device, dtype=dtype)
        except ImportError:
            # If scipy isn't available, provide a helpful error
            raise RuntimeError(
                "Neither torch.special.iv nor scipy.special is available. "
                "Please either upgrade PyTorch (>=1.11) or install scipy: pip install scipy"
            )
    sign = torch.where((orders % 2) == 0, torch.ones_like(orders), -torch.ones_like(orders))  # (-1)^k
    e = torch.exp(-tau)

    a = torch.empty(K + 1, device=device, dtype=dtype)
    a[0] = e * Ik[0]
    if K >= 1:
        a[1:] = 2.0 * e * sign[1:] * Ik[1:]
    return a

def X0(t: torch.Tensor) -> torch.Tensor:
    """Utility: returns a 0-tensor on the same device/dtype as t."""
    return t.new_zeros(())