# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TNRD (Trainable Non-linear Reaction Diffusion) for multiplicative Gamma noise removal on grayscale images. Compares a spec-compliant baseline TNRD, a log-domain modified TNRD with learnable influence functions, and a classical PDE reference (Shan 2019). Evaluated on BSD68 under Gamma noise with L=1 (strong) and L=10 (mild).

## Commands

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
# Baseline (stage-wise, fixed phi)
python -m src.training.train_baseline --L 1
python -m src.training.train_baseline --L 10

# Modified (joint, log-domain, learnable phi)
python -m src.training.train_modified --L 1
python -m src.training.train_modified --L 10
```

### Evaluation
```bash
python scripts/evaluate.py --L 1
python scripts/evaluate.py --L 10
```

### Ablation
```bash
python scripts/ablation.py
```

### Apple Silicon
Prefix with `PYTORCH_ENABLE_MPS_FALLBACK=1` if you hit unimplemented MPS kernels.

## Architecture

**Three models compared:**

1. **BaselineTNRD** (`src/models/baseline_tnrd.py`) — Spec-compliant: fixed RBF influence functions (`FixedRBFInfluence`), stage-wise greedy training (train stage 1, then 1+2, etc.), reaction term `lambda*(u-f)/(u^2+eps)` (MAP gradient for Gamma noise). Default: 5 stages, 24 filters, 5x5 kernels.

2. **ModifiedTNRD** (`src/models/modified_tnrd.py`) — Log-domain processing (multiplicative noise becomes additive), learnable RBF influence functions (`LearnableRBFInfluence`), jointly trained end-to-end with ReduceLROnPlateau. Reaction is additive in log space: `lambda*(u_log - f_log)`. Default: 8 stages, 48 filters, 7x7 kernels.

3. **PDE baseline** (`src/models/pde_baseline.py`) — Non-trainable. Shan 2019 explicit finite difference smooth diffusion. Called directly at eval time via `shan_pde_denoise()`.

**Training approach:** Both TNRD variants use MSE loss, Adam optimizer, gradient clipping (max_norm=1.0), and `EarlyStopping` (patience-based convergence). Baseline trains stage-wise; modified trains jointly. Training stops on convergence, not a fixed epoch count.

**Data pipeline** (`src/dataset.py`): Loads grayscale images, applies multiplicative Gamma noise on-the-fly via `add_gamma_noise()`. Training uses random 64x64 crops + flips. Test pads to multiples of 8.

**Forward signature:** Both TNRD models share `forward(f, n_stages=None) -> (output, intermediates)` so `evaluate.py` can use them interchangeably.

**Key directories:**
- `checkpoints/baseline/`, `checkpoints/modified/` — saved `.pt` files (gitignored)
- `results/` — per-model metrics CSVs and triplet PNGs (gitignored)
- `data/train/`, `data/test/` — BSD images (gitignored)
- `report/` — LaTeX paper and figures

## Conventions

- All modules run as `python -m src.training.<module>` from project root (not `python src/...`).
- Scripts in `scripts/` insert the project root into `sys.path` so they can also be run as `python scripts/<script>.py`.
- Device auto-selected: CUDA > MPS > CPU (`src/utils.get_device()`).
- Images are grayscale, float32, range [0,1], shape (B, 1, H, W).
