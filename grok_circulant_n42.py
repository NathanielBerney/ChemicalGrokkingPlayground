#!/usr/bin/env python3
"""Paper-inspired grokking probe on n=42 circulant HOMO/LUMO regression.

Uses one CUDA device. This adapts the one-layer, four-head, d=128, MLP=512,
AdamW setup of Nanda et al. (2023) to raw Hückel matrices and a two-value
regression target. The task, loss, and 42-token input differ from the paper's
three-token modular-addition classification task.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MatrixTransformer(nn.Module):
    def __init__(self, n: int = 42, d_model: int = 128, heads: int = 4, mlp: int = 512):
        super().__init__()
        self.row_embed = nn.Linear(n, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, mlp), nn.ReLU(), nn.Linear(mlp, d_model))
        self.out = nn.Linear(d_model, 2)

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        batch = matrix.shape[0]
        x = self.row_embed(matrix)
        x = torch.cat((self.cls.expand(batch, -1, -1), x), dim=1) + self.pos
        attention, _ = self.attn(x, x, x, need_weights=False)
        x = x + attention
        x = x + self.ff(x)
        return self.out(x[:, 0])


def load_data(path: Path, seed: int, train_fraction: float):
    with np.load(path, allow_pickle=False) as data:
        x = data["matrices"].astype(np.float32)
        spectra = data["eigenvalues"].astype(np.float32)
        n = x.shape[1]
        y = spectra[:, n // 2 - 1:n // 2 + 1]
    if not (x.ndim == 3 and x.shape[1:] == (42, 42) and y.shape == (len(x), 2)):
        raise ValueError("Expected the original 10,000-row n=42 dataset")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = int(round(train_fraction * len(x)))
    train, test = order[:cut], order[cut:]
    mean, std = y[train].mean(0), y[train].std(0)
    std = np.maximum(std, 1e-6)
    return x, y, train, test, mean, std


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.sum() <= 0:
        return 0.0
    n = len(values)
    return float((2 * np.arange(1, n + 1) - n - 1) @ values / (n * values.sum()))


def fourier_weight_gini(model: MatrixTransformer) -> float:
    # An exploratory analogue of the paper's Fourier sparsity measure, not
    # its restricted/excluded loss. The row projection is only one pathway.
    weight = model.row_embed.weight.detach().float().cpu().numpy()
    power = (np.abs(np.fft.rfft(weight, axis=1)) ** 2).sum(axis=0)
    return gini(power[1:])


@torch.no_grad()
def evaluate(model, x, y, indices, mean, std, batch_size=1024):
    model.eval()
    total_sq = torch.zeros(2, device=x.device)
    total_abs = torch.zeros(2, device=x.device)
    for batch in indices.split(batch_size):
        pred = model(x[batch]) * std + mean
        delta = pred - y[batch]
        total_sq += (delta * delta).sum(0)
        total_abs += delta.abs().sum(0)
    rmse = (total_sq / len(indices)).sqrt().cpu().numpy()
    mae = (total_abs / len(indices)).cpu().numpy()
    return rmse, mae


def run(args) -> None:
    if args.steps < 1 or args.eval_every < 1 or args.checkpoint_every < 1:
        raise ValueError("steps, eval_every, and checkpoint_every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA PyTorch environment is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{args.gpu}")
    x_np, y_np, train_np, test_np, mean_np, std_np = load_data(
        args.dataset, args.seed, args.train_fraction)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    train = torch.from_numpy(train_np).to(device)
    test = torch.from_numpy(test_np).to(device)
    x_train = x[train]
    y_train = y[train]
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    model = MatrixTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if not np.array_equal(checkpoint["train_indices"], train_np):
            raise ValueError("Resume checkpoint uses a different train/test split")
        start_step = int(checkpoint["step"]) + 1
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["dataset"] = str(args.dataset.resolve())
    config["outdir"] = str(args.outdir.resolve())
    config["resume"] = str(args.resume.resolve()) if args.resume is not None else None
    config["device_name"] = torch.cuda.get_device_name(device)
    config["train_size"] = len(train)
    config["test_size"] = len(test)
    config["target_mean"] = mean_np.tolist()
    config["target_std"] = std_np.tolist()
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def save_checkpoint(step: int, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "train_indices": train_np, "test_indices": test_np,
                    "target_mean": mean_np, "target_std": std_np}, temporary)
        temporary.replace(path)

    fields = ["step", "elapsed_seconds", "train_loss_normalized", "train_homo_rmse",
              "train_lumo_rmse", "test_homo_rmse", "test_lumo_rmse",
              "train_homo_mae", "train_lumo_mae", "test_homo_mae", "test_lumo_mae",
              "row_weight_fourier_gini"]
    start = time.perf_counter()
    timed_steps = []
    metrics_path = args.outdir / "metrics.csv"
    append = start_step > 1 and metrics_path.exists()
    with metrics_path.open("a" if append else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for step in range(start_step, args.steps + 1):
            model.train()
            if args.sync_timing:
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train)
            loss = ((pred - (y_train - mean) / std) ** 2).mean()
            loss.backward()
            optimizer.step()
            if args.sync_timing:
                torch.cuda.synchronize(device)
            if step > args.warmup_steps:
                timed_steps.append(time.perf_counter() - tick)
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                train_rmse, train_mae = evaluate(model, x, y, train, mean, std)
                test_rmse, test_mae = evaluate(model, x, y, test, mean, std)
                torch.cuda.synchronize(device)
                record = {
                    "step": step, "elapsed_seconds": time.perf_counter() - start,
                    "train_loss_normalized": float(loss.item()),
                    "train_homo_rmse": float(train_rmse[0]),
                    "train_lumo_rmse": float(train_rmse[1]),
                    "test_homo_rmse": float(test_rmse[0]),
                    "test_lumo_rmse": float(test_rmse[1]),
                    "train_homo_mae": float(train_mae[0]),
                    "train_lumo_mae": float(train_mae[1]),
                    "test_homo_mae": float(test_mae[0]),
                    "test_lumo_mae": float(test_mae[1]),
                    "row_weight_fourier_gini": fourier_weight_gini(model),
                }
                writer.writerow(record)
                handle.flush()
                print(json.dumps(record), flush=True)
            if step % args.checkpoint_every == 0:
                save_checkpoint(step, args.outdir / "latest.pt")
    save_checkpoint(args.steps, args.outdir / "last.pt")
    if timed_steps:
        summary = {"median_train_step_seconds": float(np.median(timed_steps)),
                   "mean_train_step_seconds": float(np.mean(timed_steps)),
                   "total_seconds": time.perf_counter() - start,
                   "device": config["device_name"]}
        (args.outdir / "timing.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/circulant_n42.npz"))
    parser.add_argument("--outdir", type=Path, default=Path("circulant_grokking_runs/seed0"))
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--sync-timing", action="store_true",
                        help="Synchronize each step for accurate pilot timing")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
