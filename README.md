# TNRD: Trainable Non-linear Reaction Diffusion for Multiplicative Gamma Noise

Implementation of TNRD with stage-wise training (baseline) and a log-domain
variant with learnable RBF influence functions (modified), evaluated on BSD68
under multiplicative Gamma noise (L=1, L=10). Compared against the smooth
diffusion PDE from Shan, Sun & Guo, *J. Math. Imaging Vis.* (2019).

## Project layout

```
src/
  dataset.py              gamma noise + BSD loader
  utils.py                metrics, device, early stopping
  models/
    baseline_tnrd.py      spec-compliant TNRD (fixed phi, lambda*(u-f)/(u^2+eps))
    modified_tnrd.py      log-domain TNRD, learnable phi, larger capacity
    pde_baseline.py       Shan 2019 EFDM (non-trainable PDE reference)
  training/
    train_baseline.py     stage-wise training
    train_modified.py     joint training with convergence stopping
scripts/
  evaluate.py             runs all 3 models on BSD68, writes metrics + PNGs
  ablation.py             small ablation grid for the report
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
# Baseline (stage-wise, spec-compliant).
python -m src.training.train_baseline --L 1
python -m src.training.train_baseline --L 10

# Modified (joint, log-domain, learnable influence functions).
python -m src.training.train_modified --L 1
python -m src.training.train_modified --L 10
```

Training stops automatically when the loss has not improved for `--patience`
epochs (default 8). A `--max_epochs` safety cap prevents runaway training.

## Evaluate

```bash
python scripts/evaluate.py --L 1
python scripts/evaluate.py --L 10
```

This writes per-image metrics + side-by-side `noisy | denoised | clean`
visualisations under `results/`, and a summary CSV per noise level.

## Ablation

```bash
python scripts/ablation.py
```

## Notes on Apple Silicon

The code auto-selects `mps` on M-series Macs. If you hit a kernel that's not
implemented on MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.training.train_baseline --L 1
```
