"""
src/models/baseline_tnrd.py

Spec-compliant TNRD per the assignment image:

    u_t = u_{t-1} - ( sum_i k_i^T * (u_sigma / M * phi_i^t(k_i^t * u_{t-1})) + lambda*(u-f)/(u^2 + eps) )

- phi_i (RBF influence) is FIXED during training (assignment: "Fix phi_i and learn other parameters stage-wise").
- Stage-wise training is supported via .freeze_stages(...) / .active_stages(...).
- Reaction term denominator is (u^2 + eps), matching the assignment specification
  (this is the MAP gradient for multiplicative Gamma noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedRBFInfluence(nn.Module):
    """
    RBF-based influence function with FROZEN parameters.

    Initialised so that the overall response is approximately tanh(x), which is the
    standard choice in the original TNRD paper as a reasonable fixed influence.
    """

    def __init__(self, num_basis: int = 31, gamma: float = 0.1):
        super().__init__()
        mu = torch.linspace(-1.0, 1.0, num_basis)
        # Fit weights so that sum_j w_j * exp(-(x-mu_j)^2 / (2 gamma^2)) ~ tanh(x).
        x = torch.linspace(-1.0, 1.0, 256)
        target = torch.tanh(3.0 * x)  # mild slope
        # Build basis matrix (256, num_basis) and least-squares solve.
        diff = x.unsqueeze(-1) - mu                # (256, num_basis)
        basis = torch.exp(-(diff ** 2) / (2 * gamma ** 2))
        weights = torch.linalg.lstsq(basis, target).solution

        self.register_buffer("mu", mu)
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-1) - self.mu
        rbf = torch.exp(-(diff ** 2) / (2 * self.gamma ** 2))
        return torch.sum(self.weights * rbf, dim=-1)


class BaselineTNRDStage(nn.Module):
    """One reaction-diffusion stage. Implements the assignment formula exactly."""

    def __init__(self, num_filters: int = 24, kernel_size: int = 5, eps: float = 1e-3):
        super().__init__()
        # Trainable filters (zero-mean DCT-like initialisation works well; here we use small Gaussian).
        self.filters = nn.Parameter(
            torch.randn(num_filters, 1, kernel_size, kernel_size) * 0.05
        )
        self.influences = nn.ModuleList([FixedRBFInfluence() for _ in range(num_filters)])
        self.lambda_param = nn.Parameter(torch.tensor(0.10))
        self.pad = kernel_size // 2
        self.eps = eps

    def forward(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        diffusion = torch.zeros_like(u)
        # Local gray-level statistic (u_sigma) — local mean via 3x3 average pool.
        u_sigma = F.avg_pool2d(u, kernel_size=3, stride=1, padding=1)
        M = torch.mean(u_sigma) + 1e-3

        for i in range(self.filters.shape[0]):
            k = self.filters[i:i + 1]
            conv = F.conv2d(u, k, padding=self.pad)
            phi = self.influences[i](conv)
            scaled_phi = (u_sigma / M) * phi
            kT = torch.flip(k, [2, 3])
            diffusion = diffusion + F.conv2d(scaled_phi, kT, padding=self.pad)

        # Reaction term per the assignment: lambda * (u - f) / (u^2 + eps).
        reaction = self.lambda_param * (u - f) / (u * u + self.eps)

        u_next = u - diffusion - reaction
        return torch.clamp(u_next, 0.0, 1.0)


class BaselineTNRD(nn.Module):
    def __init__(self, T: int = 5, num_filters: int = 24, kernel_size: int = 5):
        super().__init__()
        self.stages = nn.ModuleList(
            [BaselineTNRDStage(num_filters, kernel_size) for _ in range(T)]
        )
        self.T = T

    # ---------- stage-wise training helpers ----------
    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_stage(self, idx: int) -> None:
        for p in self.stages[idx].parameters():
            p.requires_grad = True

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---------- forward ----------
    def forward(self, f: torch.Tensor, n_stages: int = None):
        """
        Run only the first `n_stages` stages (used during stage-wise training).
        If None, run all T stages.
        """
        if n_stages is None:
            n_stages = self.T
        u = f.clone()
        intermediates = []
        for t in range(n_stages):
            u = self.stages[t](u, f)
            intermediates.append(u)
        return u, intermediates
