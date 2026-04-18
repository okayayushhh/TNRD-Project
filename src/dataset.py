"""
src/dataset.py

BSD dataset for multiplicative-Gamma-noise denoising.

Key points:
- Grayscale, normalised to [0, 1].
- Multiplicative Gamma noise with configurable look number L (assignment uses L=1 and L=10).
- Train mode: random crop + flip augmentation.
- Test mode: full image, no augmentation, no resize (PSNR is computed on native size).
"""

import os
import random
import cv2
import torch
from torch.utils.data import Dataset


def add_gamma_noise(img: torch.Tensor, L: int) -> torch.Tensor:
    """Multiplicative Gamma noise with mean 1, shape L, scale 1/L. Lower L = stronger noise."""
    gamma = torch.distributions.Gamma(concentration=float(L), rate=float(L))
    n = gamma.sample(img.shape)
    noisy = img * n
    # Clamp to [0, 1] only for visualisation safety; the *target* stays clean.
    return torch.clamp(noisy, 0.0, 1.0)


class BSDDataset(Dataset):
    """
    Args:
        folder:    directory of .png/.jpg/.jpeg images.
        L:         Gamma look number. Smaller L = stronger noise.
        patch:     crop size for training; ignored when train=False.
        train:     if True, apply random crop + horizontal flip; else return full image.
    """

    def __init__(self, folder: str, L: int = 1, patch: int = 64, train: bool = True):
        self.paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {folder}")
        self.L = L
        self.patch = patch
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read {self.paths[idx]}")

        if self.train:
            # Random crop, then random flip — much better than resizing for denoising.
            h, w = img.shape
            p = self.patch
            if h < p or w < p:
                img = cv2.resize(img, (max(p, w), max(p, h)))
                h, w = img.shape
            top = random.randint(0, h - p)
            left = random.randint(0, w - p)
            img = img[top:top + p, left:left + p]
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
        else:
            # Test: pad to multiple of 8 so pooling/conv shapes stay clean.
            h, w = img.shape
            ph = (8 - h % 8) % 8
            pw = (8 - w % 8) % 8
            if ph or pw:
                img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)

        clean = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        noisy = add_gamma_noise(clean, L=self.L)
        return noisy, clean
