"""
src/models/pde_baseline.py

Reference PDE model from:
    Shan, Sun & Guo, "Multiplicative Noise Removal Based on the Smooth Diffusion
    Equation", J. Math. Imaging Vis. (2019).
    https://link.springer.com/article/10.1007/s10851-018-00870-z

Implements the Explicit Finite Difference Method (EFDM, Algorithm 1) for:

    du/dt = div( g(u_sigma, |grad u_sigma|) * grad u )

with diffusion coefficient

    g(u_sigma, |grad u_sigma|) = (u_sigma / M)^alpha  *  1 / (1 + |grad u_sigma|^beta)

This is NON-trainable; it's the classical PDE baseline your assignment requires
you to compare TNRD against. Used at evaluation time only.

Recommended params from Table 2 of the paper for L=10:
    sigma=1, alpha~1.5-2.0, beta~1.5-1.8, dt=0.1, n_iter=200-400.
"""

import torch
import torch.nn.functional as F


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g.outer(g).unsqueeze(0).unsqueeze(0)  # (1,1,K,K)
    return k2d


def _gaussian_smooth(u: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gaussian_kernel(sigma, u.device)
    pad = k.shape[-1] // 2
    return F.conv2d(u, k, padding=pad)


def _forward_diff(u: torch.Tensor, axis: str) -> torch.Tensor:
    """Forward difference with reflective boundary."""
    if axis == "x":
        return F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    else:
        return F.pad(u[:, :, 1:, :] - u[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")


def _backward_diff(u: torch.Tensor, axis: str) -> torch.Tensor:
    if axis == "x":
        return F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (1, 0, 0, 0), mode="replicate")
    else:
        return F.pad(u[:, :, 1:, :] - u[:, :, :-1, :], (0, 0, 1, 0), mode="replicate")


def shan_pde_denoise(
    f: torch.Tensor,
    sigma: float = 1.0,
    alpha: float = 1.5,
    beta: float = 1.5,
    dt: float = 0.1,
    n_iter: int = 250,
) -> torch.Tensor:
    """
    Apply the Shan 2019 smooth diffusion model to a noisy image f.

    Args:
        f:       (B, 1, H, W) noisy image in [0, 1].
        sigma:   Gaussian smoothing scale for u_sigma.
        alpha:   gray-level indicator exponent.
        beta:    edge-detection exponent.
        dt:      time step (must be small for EFDM stability; <= 0.25).
        n_iter:  number of iterations.

    Returns:
        Denoised image, same shape as f, clamped to [0, 1].
    """
    u = f.clone()
    for _ in range(n_iter):
        u_sigma = _gaussian_smooth(u, sigma)
        # Gradient of smoothed image (centred via fwd diff is fine for EFDM).
        grad_x = _forward_diff(u_sigma, "x")
        grad_y = _forward_diff(u_sigma, "y")
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        M = u_sigma.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-3)
        a = (u_sigma / M).clamp(min=1e-6) ** alpha          # gray-level indicator
        b = 1.0 / (1.0 + grad_mag ** beta)                  # edge detector
        g = a * b                                            # diffusion coefficient

        # div( g * grad u ) using fwd/bwd differences:
        ux = _forward_diff(u, "x")
        uy = _forward_diff(u, "y")
        flux_x = g * ux
        flux_y = g * uy
        div = _backward_diff(flux_x, "x") + _backward_diff(flux_y, "y")

        u = u + dt * div
        u = u.clamp(0.0, 1.0)
    return u
