"""
scripts/evaluate.py

Evaluate baseline TNRD, modified TNRD, and the PDE reference (Shan 2019)
on the test set. Writes:
    results/<model>/metrics.csv           per-image PSNR + SSIM
    results/<model>/<image>.png           noisy | denoised | clean triplet
    results/summary_L{L}.csv              mean PSNR/SSIM per model

Run:
    python scripts/evaluate.py --L 1
    python scripts/evaluate.py --L 10
"""

import argparse
import csv
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

# Allow running from project root with `python scripts/evaluate.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import BSDDataset
from src.models.baseline_tnrd import BaselineTNRD
from src.models.modified_tnrd import ModifiedTNRD
from src.models.pde_baseline import shan_pde_denoise
from src.utils import get_device, psnr, ssim, save_triplet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str, default="data/test")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--L", type=int, default=1, choices=[1, 5, 10])
    p.add_argument("--ckpt_baseline", type=str, default=None,
                   help="Default: checkpoints/baseline/baseline_L{L}_final.pt")
    p.add_argument("--ckpt_modified", type=str, default=None,
                   help="Default: checkpoints/modified/modified_L{L}_final.pt")
    p.add_argument("--pde_alpha", type=float, default=1.5)
    p.add_argument("--pde_beta", type=float, default=1.5)
    p.add_argument("--pde_iter", type=int, default=250)
    return p.parse_args()


def evaluate_model(name, model_fn, loader, device, out_dir):
    """model_fn(noisy_tensor) -> denoised_tensor.  Returns (mean_psnr, mean_ssim, rows)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    psnrs, ssims, runtimes = [], [], []

    for i, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)

        t0 = time.time()
        with torch.no_grad():
            denoised = model_fn(noisy)
        runtime = time.time() - t0

        p = psnr(denoised, clean).item()
        s = ssim(denoised, clean).item()
        psnrs.append(p); ssims.append(s); runtimes.append(runtime)

        rows.append({"image": i, "psnr": p, "ssim": s, "runtime_s": runtime})
        save_triplet(noisy, denoised, clean, os.path.join(out_dir, f"{i:03d}.png"))

    mean_p = sum(psnrs) / len(psnrs)
    mean_s = sum(ssims) / len(ssims)
    mean_t = sum(runtimes) / len(runtimes)
    print(f"  {name:10s}  PSNR {mean_p:6.3f}  SSIM {mean_s:.4f}  "
          f"avg time {mean_t * 1000:.0f} ms")

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim", "runtime_s"])
        w.writeheader()
        w.writerows(rows)

    return mean_p, mean_s, mean_t


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"L = {args.L}")

    test_set = BSDDataset(args.test_dir, L=args.L, train=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print(f"Test images: {len(test_set)}\n")

    summary = {}

    # ---------- baseline TNRD ----------
    ckpt_b = args.ckpt_baseline or f"checkpoints/baseline/baseline_L{args.L}_final.pt"
    if os.path.exists(ckpt_b):
        ckpt = torch.load(ckpt_b, map_location=device, weights_only=False)
        a = ckpt.get("args", {})
        baseline = BaselineTNRD(T=a.get("T", 5),
                                num_filters=a.get("num_filters", 24),
                                kernel_size=a.get("kernel_size", 5)).to(device)
        baseline.load_state_dict(ckpt["model_state"])
        baseline.eval()
        out = os.path.join(args.results_dir, f"baseline_L{args.L}")
        summary["baseline"] = evaluate_model(
            "baseline", lambda x: baseline(x)[0], loader, device, out
        )
    else:
        print(f"  baseline    SKIP (no checkpoint at {ckpt_b})")

    # ---------- modified TNRD ----------
    ckpt_m = args.ckpt_modified or f"checkpoints/modified/modified_L{args.L}_final.pt"
    if os.path.exists(ckpt_m):
        ckpt = torch.load(ckpt_m, map_location=device, weights_only=False)
        a = ckpt.get("args", {})
        modified = ModifiedTNRD(T=a.get("T", 8),
                                num_filters=a.get("num_filters", 48),
                                kernel_size=a.get("kernel_size", 7)).to(device)
        modified.load_state_dict(ckpt["model_state"])
        modified.eval()
        out = os.path.join(args.results_dir, f"modified_L{args.L}")
        summary["modified"] = evaluate_model(
            "modified", lambda x: modified(x)[0], loader, device, out
        )
    else:
        print(f"  modified    SKIP (no checkpoint at {ckpt_m})")

    # ---------- PDE reference ----------
    out = os.path.join(args.results_dir, f"pde_L{args.L}")
    summary["pde"] = evaluate_model(
        "pde",
        lambda x: shan_pde_denoise(x, alpha=args.pde_alpha, beta=args.pde_beta,
                                   n_iter=args.pde_iter),
        loader, device, out,
    )

    # ---------- summary CSV ----------
    summary_path = os.path.join(args.results_dir, f"summary_L{args.L}.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "psnr", "ssim", "avg_runtime_s"])
        for name, (p, s, t) in summary.items():
            w.writerow([name, f"{p:.4f}", f"{s:.4f}", f"{t:.4f}"])
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
