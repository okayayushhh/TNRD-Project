"""
src/models/modified_tnrd.py

Modified TNRD with three improvements over the baseline:

1) LOG-DOMAIN PROCESSING.  Multiplicative noise becomes additive in log space:
        log(y) = log(x) + log(eta)
   We process log(y + delta), denoise additively, then exponentiate. This is the
   single most effective change for Gamma-noise denoising and is the standard
   trick in SAR despeckling.

2) LEARNABLE INFLUENCE FUNCTIONS (phi_i).  After stage-wise warm-up of the
   baseline, jointly train the RBF weights too. This is the original
   Chen & Pock TNRD setup and consistently outperforms fixed influence.

3) HIGHER CAPACITY.  48 filters / 7x7 kernels / 8 stages.

Reaction term in log domain becomes additive (l1-style toward log(f)):
        reaction = lambda * (u_log - f_log)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableRBFInfluence(nn.Module):
    def __init__(self, num_basis: int = 31, gamma: float = 0.1, init_scale: float = 0.1):
        super().__init__()
        mu = torch.linspace(-1.0, 1.0, num_basis)
        # Init weights from a tanh-shaped curve (good prior), then let SGD fine-tune.
        x = torch.linspace(-1.0, 1.0, 256)
        target = torch.tanh(3.0 * x) * init_scale
        diff = x.unsqueeze(-1) - mu
        basis = torch.exp(-(diff ** 2) / (2 * gamma ** 2))
        weights_init = torch.linalg.lstsq(basis, target).solution

        self.register_buffer("mu", mu)
        self.weights = nn.Parameter(weights_init)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-1) - self.mu
        rbf = torch.exp(-(diff ** 2) / (2 * self.gamma ** 2))
        return torch.sum(self.weights * rbf, dim=-1)


class ModifiedTNRDStage(nn.Module):
    def __init__(self, num_filters: int = 48, kernel_size: int = 7):
        super().__init__()
        self.filters = nn.Parameter(
            torch.randn(num_filters, 1, kernel_size, kernel_size) * 0.05
        )
        self.influences = nn.ModuleList(
            [LearnableRBFInfluence() for _ in range(num_filters)]
        )
        self.lambda_param = nn.Parameter(torch.tensor(0.10))
        self.pad = kernel_size // 2

    def forward(self, u_log: torch.Tensor, f_log: torch.Tensor) -> torch.Tensor:
        diffusion = torch.zeros_like(u_log)
        for i in range(self.filters.shape[0]):
            k = self.filters[i:i + 1]
            conv = F.conv2d(u_log, k, padding=self.pad)
            phi = self.influences[i](conv)
            kT = torch.flip(k, [2, 3])
            diffusion = diffusion + F.conv2d(phi, kT, padding=self.pad)

        # Additive reaction in log domain (correct for multiplicative noise).
        reaction = self.lambda_param * (u_log - f_log)
        return u_log - diffusion - reaction


class ModifiedTNRD(nn.Module):
    """
    Forward signature mirrors BaselineTNRD so evaluate.py can use either model
    interchangeably. Internally it runs in log-domain.
    """

    DELTA = 1e-3  # offset for log to avoid -inf at zero pixels

    def __init__(self, T: int = 8, num_filters: int = 48, kernel_size: int = 7):
        super().__init__()
        self.stages = nn.ModuleList(
            [ModifiedTNRDStage(num_filters, kernel_size) for _ in range(T)]
        )
        self.T = T

    def trainable_params(self):
        return list(self.parameters())

    def forward(self, f: torch.Tensor, n_stages: int = None):
        if n_stages is None:
            n_stages = self.T
        f_log = torch.log(torch.clamp(f, min=self.DELTA))
        u_log = f_log.clone()
        intermediates = []
        for t in range(n_stages):
            u_log = self.stages[t](u_log, f_log)
            intermediates.append(u_log)
        u = torch.exp(u_log)
        return torch.clamp(u, 0.0, 1.0), intermediates
