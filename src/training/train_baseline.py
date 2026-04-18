"""
src/training/train_baseline.py

Stage-wise training of the spec-compliant baseline TNRD.

Per the assignment: "Fix phi_i and learn other parameters stage-wise."
We train one stage at a time greedily: train stage 1 to convergence with the
network using only stage 1, then unfreeze stage 2 and train stages 1+2, etc.

Convergence: training stops when the running average loss has not improved
by `min_delta` for `patience` epochs. A `max_epochs_per_stage` safety cap
prevents pathological non-convergence.

Run:
    python -m src.training.train_baseline \
        --L 1 --train_dir data/train --ckpt_dir checkpoints/baseline
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from src.dataset import BSDDataset
from src.models.baseline_tnrd import BaselineTNRD
from src.utils import EarlyStopping, get_device, count_params, psnr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, default="data/train")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/baseline")
    p.add_argument("--L", type=int, default=1, choices=[1, 5, 10],
                   help="Gamma noise look number; assignment uses 1 and 10.")
    p.add_argument("--T", type=int, default=5, help="Number of TNRD stages.")
    p.add_argument("--num_filters", type=int, default=24)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8,
                   help="Early-stop patience in epochs (per stage).")
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--max_epochs_per_stage", type=int, default=80,
                   help="Safety cap per stage; convergence usually hits earlier.")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def train_one_stage(model, loader, device, stage_idx, args):
    """Train stages 0..stage_idx with stages > stage_idx frozen out (not used)."""
    # Freeze everything, then unfreeze the stages we actually want to train.
    model.freeze_all()
    for s in range(stage_idx + 1):
        model.unfreeze_stage(s)

    optimizer = torch.optim.Adam(model.trainable_params(), lr=args.lr)
    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    n_active = stage_idx + 1
    print(f"\n--- Training stage {stage_idx + 1}/{model.T} "
          f"(active stages: 1..{n_active}) ---")
    print(f"    trainable params: {count_params(model):,}")

    for epoch in range(args.max_epochs_per_stage):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        n = 0

        model.train()
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output, _ = model(noisy, n_stages=n_active)
            loss = torch.mean((output - clean) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * noisy.size(0)
            epoch_psnr += psnr(output, clean).item() * noisy.size(0)
            n += noisy.size(0)

        epoch_loss /= n
        epoch_psnr /= n
        dt = time.time() - t0
        print(f"  stage {stage_idx + 1} | epoch {epoch + 1:3d} "
              f"| loss {epoch_loss:.6f} | psnr {epoch_psnr:5.2f} dB | {dt:.1f}s")

        if stopper.step(epoch_loss):
            print(f"  -> converged after {epoch + 1} epochs (no improvement "
                  f"for {args.patience} epochs).")
            break


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    print(f"L = {args.L} (Gamma look number)")

    dataset = BSDDataset(args.train_dir, L=args.L, patch=args.patch, train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    print(f"Training images: {len(dataset)}")

    model = BaselineTNRD(T=args.T, num_filters=args.num_filters,
                         kernel_size=args.kernel_size).to(device)

    for stage_idx in range(args.T):
        train_one_stage(model, loader, device, stage_idx, args)
        ckpt_path = os.path.join(args.ckpt_dir, f"baseline_L{args.L}_stage{stage_idx + 1}.pt")
        torch.save({"model_state": model.state_dict(),
                    "args": vars(args),
                    "stage": stage_idx + 1}, ckpt_path)
        print(f"  saved {ckpt_path}")

    final_path = os.path.join(args.ckpt_dir, f"baseline_L{args.L}_final.pt")
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, final_path)
    print(f"\nDone. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
