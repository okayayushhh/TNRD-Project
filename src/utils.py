"""
src/utils.py

Shared helpers: device selection, metrics (PSNR / SSIM), early stopping, visualisation.
"""

import os
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    """CUDA if present, else MPS on Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR in dB, assuming inputs in [0, 1]."""
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + 1e-10))


def _gaussian_window(window_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Standard SSIM on grayscale images in [0, 1]. Inputs are (B, 1, H, W)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    device = pred.device
    window = _gaussian_window(window_size, 1.5, device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad)
    mu2 = F.conv2d(target, window, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return s.mean()


# -----------------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------------
class EarlyStopping:
    """
    Stop when the monitored value has not improved by `min_delta` for `patience` epochs.

    Used for training-loss convergence (you asked: stop when loss has converged,
    not after a fixed number of epochs). A `max_epochs` safety cap is enforced
    by the training loop, not here.
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
def save_triplet(noisy: torch.Tensor, denoised: torch.Tensor, clean: torch.Tensor,
                 out_path: str) -> None:
    """Save side-by-side noisy | denoised | clean as a single PNG."""
    def _to_np(t):
        a = t.detach().squeeze().cpu().numpy()
        a = np.clip(a, 0.0, 1.0) * 255.0
        return a.astype(np.uint8)

    n, d, c = _to_np(noisy), _to_np(denoised), _to_np(clean)
    grid = np.concatenate([n, d, c], axis=1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, grid)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
