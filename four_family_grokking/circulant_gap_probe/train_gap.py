#!/usr/bin/env python3
"""Controlled circulant HOMO-LUMO gap probes on the original 10k sizes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sys import path as sys_path
sys_path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_family import BitSizeTransformer, split_indices  # noqa: E402


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parents[1] / "constrained_huckel_datasets/circulant/dataset.npz"
VARIANTS = ("raw_bits", "log_bits", "log_flatbits", "log_scalar", "log_posbits")


def load_data():
    with np.load(DATASET, allow_pickle=False) as data:
        n = data["n"].astype(np.int64)
        gap = data["homo_lumo_gap"].astype(np.float64)
        actual_gap = data["lumo_energy"] - data["homo_energy"]
    if not np.allclose(gap, actual_gap, rtol=1e-13, atol=1e-13):
        raise ValueError("Stored gap differs from LUMO minus HOMO")
    exact = 4 * np.sin(np.pi / n)
    if not np.allclose(gap, exact, rtol=1e-12, atol=1e-12):
        raise ValueError("Stored gap differs from the periodic Hückel formula")
    return n, gap


class ScalarGapMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class FlatBitsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(16, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class GapBitTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = BitSizeTransformer()
        self.inner.out = nn.Linear(128, 1)

    def forward(self, x):
        return self.inner(x).squeeze(-1)


class PositionAwareEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(32, 128)

    def forward(self, bits):
        positions = 2 * torch.arange(16, device=bits.device)
        return self.table(bits + positions)


class PositionAwareGapTransformer(GapBitTransformer):
    def __init__(self):
        super().__init__()
        self.inner.embed = PositionAwareEmbedding()


def score(pred, truth, idx):
    p, y = pred[idx], truth[idx]
    relative = np.abs(p - y) / y
    return {
        "accuracy_rel_1pct": float(np.mean(relative < .01)),
        "accuracy_rel_5pct": float(np.mean(relative < .05)),
        "accuracy_rel_10pct": float(np.mean(relative < .10)),
        "median_relative_error": float(np.median(relative)),
        "mean_relative_error": float(np.mean(relative)),
        "p90_relative_error": float(np.quantile(relative, .9)),
        "log_rmse": float(np.sqrt(np.mean((np.log(np.maximum(p, 1e-30)) - np.log(y)) ** 2))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
    }


@torch.inference_mode()
def predict_all(model, x, variant, center, scale, batch_size=2048):
    model.eval()
    pieces = []
    for first in range(0, len(x), batch_size):
        pieces.append(model(x[first:first + batch_size]).cpu().numpy().reshape(-1))
    z = np.concatenate(pieces) * scale + center
    return np.exp(np.clip(z, -30, 5)) if variant != "raw_bits" else z


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{args.gpu}")
    n, gap = load_data()
    train, test = split_indices(len(n), args.seed, .30)
    if args.variant == "log_scalar":
        log_n = np.log(n.astype(np.float64))
        input_mean = float(log_n[train].mean())
        input_std = float(log_n[train].std())
        x = torch.as_tensor(((log_n - input_mean) / input_std)[:, None],
                            dtype=torch.float32, device=device)
        model = ScalarGapMLP().to(device)
    elif args.variant == "log_flatbits":
        input_mean = input_std = None
        bits = ((n[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.float32)
        x = torch.as_tensor(bits, device=device)
        model = FlatBitsMLP().to(device)
    else:
        input_mean = input_std = None
        bits = ((n[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.int64)
        x = torch.as_tensor(bits, device=device)
        model = (PositionAwareGapTransformer() if args.variant == "log_posbits"
                 else GapBitTransformer()).to(device)
    log_target = args.variant != "raw_bits"
    transformed = np.log(gap) if log_target else gap
    center = float(transformed[train].mean())
    scale = float(transformed[train].std())
    y_train = torch.as_tensor((transformed[train] - center) / scale,
                              dtype=torch.float32, device=device)
    x_train = x[torch.as_tensor(train, device=device)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(.9, .98))
    def learning_rate_factor(step):
        warmup = min(step / 10, 1)
        if not args.cosine_decay:
            return warmup
        progress = min(max((step - 10) / max(args.steps - 10, 1), 0), 1)
        return warmup * (.01 + .99 * .5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = {
        "variant": args.variant, "dataset": str(DATASET), "samples": len(n),
        "train_size": len(train), "test_size": len(test), "seed": args.seed,
        "steps": args.steps, "eval_every": args.eval_every,
        "archive_every": args.archive_every,
        "input": ("normalized log(n)" if args.variant == "log_scalar" else
                  "16 binary n features" if args.variant == "log_flatbits" else
                  "16 position-specific binary n tokens" if args.variant == "log_posbits" else
                  "16 binary n tokens"),
        "target": "log(gap)" if log_target else "gap",
        "architecture": "2-layer 128-wide ReLU MLP" if args.variant in ("log_scalar", "log_flatbits")
                        else "one-layer four-head d128 MLP512 ReLU no LayerNorm transformer"
                        + (" with position-specific bit embedding" if args.variant == "log_posbits" else ""),
        "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
                      "betas": [.9, .98], "warmup_steps": 10,
                      "cosine_decay_to_fraction": .01 if args.cosine_decay else None},
        "gap_formula": "4 sin(pi/n)",
        "constant_baseline_test": score(np.full_like(gap, float(np.exp(np.mean(np.log(gap[train]))))
                                                 if log_target else float(gap[train].mean())), gap, test),
        "zero_baseline_test_relative_1pct": 0.0,
        "exact_formula_max_abs_error": float(np.max(np.abs(gap - 4 * np.sin(np.pi / n)))),
    }
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    snapshots = args.outdir / "snapshots"
    def save_snapshot(step):
        torch.save({"model": model.state_dict(), "step": step,
                    "train_indices": train, "test_indices": test,
                    "target_center": center, "target_scale": scale,
                    "input_mean": input_mean, "input_std": input_std},
                   snapshots / f"step_{step:06d}.pt")
    if args.archive_every:
        snapshots.mkdir(exist_ok=True)
        save_snapshot(0)
    fieldnames = ["step", "elapsed_seconds"] + [f"{subset}_{key}" for subset in ("train", "test")
                   for key in score(gap, gap, train)]
    fields = fieldnames + ["train_loss", "test_loss"]
    began = time.perf_counter()
    with (args.outdir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train)
            loss = torch.mean((prediction - y_train) ** 2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                torch.cuda.synchronize(device)
                predicted = predict_all(model, x, args.variant, center, scale)
                normalized_error = ((np.log(np.maximum(predicted, 1e-30)) if log_target else predicted)
                                    - transformed) / scale
                row = {"step": step, "elapsed_seconds": time.perf_counter() - began,
                       "train_loss": float(np.mean(normalized_error[train] ** 2)),
                       "test_loss": float(np.mean(normalized_error[test] ** 2))}
                for subset, idx in (("train", train), ("test", test)):
                    row.update({f"{subset}_{key}": value
                                for key, value in score(predicted, gap, idx).items()})
                writer.writerow(row)
                handle.flush()
                print(json.dumps({"step": step, "train_rel_1pct": row["train_accuracy_rel_1pct"],
                                  "test_rel_1pct": row["test_accuracy_rel_1pct"],
                                  "test_log_rmse": row["test_log_rmse"]}), flush=True)
            if step % args.checkpoint_every == 0:
                torch.save({"model": model.state_dict(), "step": step, "train_indices": train,
                            "test_indices": test, "target_center": center, "target_scale": scale,
                            "input_mean": input_mean, "input_std": input_std},
                           args.outdir / "latest.pt")
            if args.archive_every and (step % args.archive_every == 0 or step in (200, 400, 600, 800)):
                save_snapshot(step)
    torch.save({"model": model.state_dict(), "step": args.steps, "train_indices": train,
                "test_indices": test, "target_center": center, "target_scale": scale,
                "input_mean": input_mean, "input_std": input_std}, args.outdir / "last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--archive-every", type=int, default=0,
                        help="Save model-only snapshots every N steps (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--cosine-decay", action="store_true")
    run(parser.parse_args())
