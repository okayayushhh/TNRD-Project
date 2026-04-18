"""
src/training/train_modified.py

Joint training of the modified (log-domain, learnable-phi) TNRD with
patience-based convergence stopping.

Run:
    python -m src.training.train_modified \
        --L 1 --train_dir data/train --ckpt_dir checkpoints/modified
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from src.dataset import BSDDataset
from src.models.modified_tnrd import ModifiedTNRD
from src.utils import EarlyStopping, get_device, count_params, psnr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, default="data/train")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/modified")
    p.add_argument("--L", type=int, default=1, choices=[1, 5, 10])
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--num_filters", type=int, default=48)
    p.add_argument("--kernel_size", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min_delta", type=float, default=1e-6)
    p.add_argument("--max_epochs", type=int, default=200,
                   help="Safety cap; convergence usually hits earlier.")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    print(f"L = {args.L} | T = {args.T} | filters = {args.num_filters} | "
          f"kernel = {args.kernel_size}x{args.kernel_size}")

    dataset = BSDDataset(args.train_dir, L=args.L, patch=args.patch, train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    print(f"Training images: {len(dataset)}")

    model = ModifiedTNRD(T=args.T, num_filters=args.num_filters,
                         kernel_size=args.kernel_size).to(device)
    print(f"Trainable params: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    print("\nTraining (will stop on convergence; max_epochs is safety cap)...")
    best_loss = float("inf")
    for epoch in range(args.max_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        n = 0

        model.train()
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output, _ = model(noisy)
            loss = torch.mean((output - clean) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * noisy.size(0)
            epoch_psnr += psnr(output, clean).item() * noisy.size(0)
            n += noisy.size(0)

        epoch_loss /= n
        epoch_psnr /= n
        scheduler.step(epoch_loss)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch + 1:3d} | loss {epoch_loss:.6f} | psnr {epoch_psnr:5.2f} dB "
              f"| lr {lr_now:.1e} | {dt:.1f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model_state": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch + 1,
                        "loss": epoch_loss},
                       os.path.join(args.ckpt_dir, f"modified_L{args.L}_best.pt"))

        if stopper.step(epoch_loss):
            print(f"\n-> converged after {epoch + 1} epochs "
                  f"(no improvement for {args.patience} epochs).")
            break

    final_path = os.path.join(args.ckpt_dir, f"modified_L{args.L}_final.pt")
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, final_path)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
