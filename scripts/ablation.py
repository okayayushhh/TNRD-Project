"""
scripts/ablation.py

Small ablation study for the report. Trains the modified TNRD with several
configurations and writes a summary CSV.

Configurations swept:
    - number of stages T in {3, 5, 8}
    - number of filters in {24, 48}
    - L in {1, 10}

Each run uses early stopping (patience=8) and a tight max_epochs cap so the
whole sweep is feasible on a laptop. Results go to results/ablation_summary.csv.

Run:
    python scripts/ablation.py
"""

import csv
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import BSDDataset
from src.models.modified_tnrd import ModifiedTNRD
from src.utils import EarlyStopping, get_device, count_params, psnr, ssim


CONFIGS = [
    {"T": 3, "num_filters": 24, "L": 1},
    {"T": 5, "num_filters": 24, "L": 1},
    {"T": 5, "num_filters": 48, "L": 1},
    {"T": 8, "num_filters": 48, "L": 1},
    {"T": 5, "num_filters": 48, "L": 10},
    {"T": 8, "num_filters": 48, "L": 10},
]

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
OUT_CSV = "results/ablation_summary.csv"
PATCH = 64
BATCH = 8
LR = 5e-4
PATIENCE = 8
MAX_EPOCHS = 60   # tight cap for sweep feasibility


def train_and_eval(cfg, device):
    print(f"\n=== {cfg} ===")
    train_set = BSDDataset(TRAIN_DIR, L=cfg["L"], patch=PATCH, train=True)
    test_set = BSDDataset(TEST_DIR, L=cfg["L"], train=False)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = ModifiedTNRD(T=cfg["T"], num_filters=cfg["num_filters"]).to(device)
    n_params = count_params(model)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    stopper = EarlyStopping(patience=PATIENCE)

    t0 = time.time()
    epochs_used = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optim.zero_grad()
            out, _ = model(noisy)
            loss = torch.mean((out - clean) ** 2)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * noisy.size(0); n += noisy.size(0)
        epoch_loss /= n
        epochs_used = epoch + 1
        if stopper.step(epoch_loss):
            break
    train_time = time.time() - t0

    # Eval
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out, _ = model(noisy)
            psnrs.append(psnr(out, clean).item())
            ssims.append(ssim(out, clean).item())
    return {
        **cfg,
        "params": n_params,
        "epochs": epochs_used,
        "train_time_s": round(train_time, 1),
        "test_psnr": round(sum(psnrs) / len(psnrs), 3),
        "test_ssim": round(sum(ssims) / len(ssims), 4),
    }


def main():
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    for cfg in CONFIGS:
        try:
            rows.append(train_and_eval(cfg, device))
        except Exception as e:
            print(f"  FAILED {cfg}: {e}")
            rows.append({**cfg, "error": str(e)})

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
