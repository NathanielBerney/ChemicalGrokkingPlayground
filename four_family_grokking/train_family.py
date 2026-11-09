#!/usr/bin/env python3
"""One-GPU grokking probes for three frontier-energy datasets and pericyclic rules.

All models use a one-layer, four-head, d=128, MLP=512, no-LayerNorm
transformer and 30% full-batch AdamW training, as in the Nanda benchmark.
The energy tasks encode the exact sparse Hamiltonian by its varying size n;
their graph type and couplings are fixed within each separately trained family.
"""

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
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ("circulant", "mobius", "linear", "pericyclic")


class BitSizeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 128
        self.embed = nn.Embedding(2, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 17, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, d))
        self.out = nn.Linear(d, 2)

    def forward(self, bits: torch.Tensor, mask=None) -> torch.Tensor:
        x = torch.cat((self.cls.expand(len(bits), -1, -1), self.embed(bits)), dim=1) + self.pos
        update, _ = self.attn(x, x, x, need_weights=False)
        x = x + update
        x = x + self.ff(x)
        return self.out(x[:, 0])


class MatrixTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 128
        self.row_embed = nn.Linear(8, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 9, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, d))
        self.out = nn.Linear(d, 1)

    def forward(self, H: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.cls.expand(len(H), -1, -1), self.row_embed(H)), dim=1) + self.pos
        key_padding_mask = torch.cat((torch.zeros((len(H), 1), dtype=torch.bool, device=H.device),
                                      ~site_mask), dim=1)
        update, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + update
        x = x + self.ff(x)
        return self.out(x[:, 0]).squeeze(-1)


def load_data(family: str, device: torch.device):
    if family == "pericyclic":
        path = ROOT / "pericyclic_thermal_dataset" / "dataset.npz"
        with np.load(path, allow_pickle=False) as data:
            H = data["H"].astype(np.float32)
            mask = data["site_mask"].astype(bool)
            n = data["n_sites"].astype(np.int64)
            y = data["y"].astype(np.float32)
        x = torch.from_numpy(H).to(device)
        site_mask = torch.from_numpy(mask).to(device)
        target = torch.from_numpy(y).to(device)
        return path, x, site_mask, target, n, y
    path = ROOT / "constrained_huckel_datasets" / family / "dataset.npz"
    with np.load(path, allow_pickle=False) as data:
        n = data["n"].astype(np.int64)
        y = np.stack([data["homo_energy"], data["lumo_energy"]], axis=1).astype(np.float32)
    if n.max() >= 2**16:
        raise ValueError("16-bit size encoding is too short")
    bits = ((n[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.int64)
    x = torch.from_numpy(bits).to(device)
    target = torch.from_numpy(y).to(device)
    return path, x, None, target, n, y


def split_indices(count, seed, train_fraction):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie in (0,1)")
    order = np.random.default_rng(seed).permutation(count)
    cut = int(round(train_fraction * count))
    return order[:cut], order[cut:]


def exact_rule_baseline(family, n, H=None):
    if family == "pericyclic":
        values = []
        for h, size in zip(H, n):
            edge = h[np.arange(size), (np.arange(size) + 1) % size]
            phase = int(np.prod(np.sign(-edge)))
            values.append(int((size % 4 == 2) == (phase == 1)))
        return np.asarray(values)
    angle = np.pi / (2 * (n + 1)) if family == "linear" else np.pi / n
    h = 2 * np.sin(angle)
    return np.stack((-h, h), axis=1)


def energy_accuracy(pred, target, tolerance):
    error = np.abs(pred - target)
    absolute = float(np.mean(np.all(error < tolerance, axis=1)))
    relative = float(np.mean(np.all(error / np.maximum(np.abs(target), 1e-8) < 0.05, axis=1)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return absolute, relative, rmse


@torch.inference_mode()
def predictions(model, x, site_mask, batch_size=2048):
    model.eval()
    parts = []
    for first in range(0, len(x), batch_size):
        last = first + batch_size
        parts.append(model(x[first:last], None if site_mask is None else site_mask[first:last]).cpu())
    return torch.cat(parts).numpy()


def write_checkpoint(path, model, optimizer, scheduler, step, train_idx, test_idx, mean, std):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step,
                "train_indices": train_idx, "test_indices": test_idx,
                "target_mean": mean, "target_std": std}, tmp)
    tmp.replace(path)


def write_snapshot(path, model, step, train_idx, test_idx, mean, std):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": {name: value.detach().cpu().clone()
                           for name, value in model.state_dict().items()},
                "step": step, "train_indices": train_idx, "test_indices": test_idx,
                "target_mean": mean, "target_std": std}, tmp)
    tmp.replace(path)


def evaluate(family, model, x, site_mask, target, y_np, n, train_idx, test_idx,
             mean, std, tolerance, target_transform, step, elapsed):
    pred = predictions(model, x, site_mask)
    if family == "pericyclic":
        labels = y_np.astype(np.int64)
        def subset(idx):
            z, y = pred[idx], labels[idx]
            loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
            return float(np.mean(loss)), float(np.mean((z >= 0) == y))
        train_loss, train_acc = subset(train_idx)
        test_loss, test_acc = subset(test_idx)
        record = {"step": step, "elapsed_seconds": elapsed,
                  "train_loss": train_loss, "test_loss": test_loss,
                  "train_accuracy": train_acc, "test_accuracy": test_acc,
                  "train_relative_accuracy": "", "test_relative_accuracy": "",
                  "train_rmse": "", "test_rmse": "", "test_small_n_rmse": "",
                  "test_small_n_relative_accuracy": ""}
    else:
        if target_transform == "log":
            raw = np.exp(np.clip(pred * std + mean, -30, 5)) * np.array([-1, 1])
            z = pred - (np.log(np.abs(y_np)) - mean) / std
        else:
            raw = pred * std + mean
            z = (raw - y_np) / std
        train_loss = float(np.mean(z[train_idx] ** 2))
        test_loss = float(np.mean(z[test_idx] ** 2))
        train_acc, train_rel, train_rmse = energy_accuracy(raw[train_idx], y_np[train_idx], tolerance)
        test_acc, test_rel, test_rmse = energy_accuracy(raw[test_idx], y_np[test_idx], tolerance)
        small = test_idx[n[test_idx] <= 1000]
        _, small_rel, small_rmse = energy_accuracy(raw[small], y_np[small], tolerance)
        record = {"step": step, "elapsed_seconds": elapsed,
                  "train_loss": train_loss, "test_loss": test_loss,
                  "train_accuracy": train_acc, "test_accuracy": test_acc,
                  "train_relative_accuracy": train_rel, "test_relative_accuracy": test_rel,
                  "train_rmse": train_rmse, "test_rmse": test_rmse,
                  "test_small_n_rmse": small_rmse,
                  "test_small_n_relative_accuracy": small_rel}
    return record


def run(args):
    if (args.family not in FAMILIES or args.steps < 1 or args.eval_every < 1
            or args.snapshot_every < 0):
        raise ValueError("Invalid family or step count")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{args.gpu}")
    path, x, site_mask, target, n, y_np = load_data(args.family, device)
    train_idx, test_idx = split_indices(len(x), args.seed, args.train_fraction)
    train_t = torch.as_tensor(train_idx, device=device)
    if args.family == "pericyclic":
        model = MatrixTransformer().to(device)
        mean = std = None
        zero_or_majority = float(max(np.mean(y_np[test_idx]), 1 - np.mean(y_np[test_idx])))
        exact = exact_rule_baseline(args.family, n, x.cpu().numpy())
        baseline = {"majority_test_accuracy": zero_or_majority,
                    "exact_loop_rule_test_accuracy": float(np.mean(exact[test_idx] == y_np[test_idx]))}
    else:
        model = BitSizeTransformer().to(device)
        transformed_y = np.log(np.abs(y_np)) if args.target_transform == "log" else y_np
        mean = transformed_y[train_idx].mean(axis=0)
        std = np.maximum(transformed_y[train_idx].std(axis=0), 1e-6)
        zero = np.zeros_like(y_np[test_idx])
        train_mean = (np.broadcast_to(np.exp(mean) * np.array([-1, 1]), y_np[test_idx].shape)
                      if args.target_transform == "log" else np.broadcast_to(mean, y_np[test_idx].shape))
        exact = exact_rule_baseline(args.family, n[test_idx])
        baseline = {"zero_test_abs_accuracy": energy_accuracy(zero, y_np[test_idx], args.tolerance)[0],
                    "zero_test_relative_accuracy": energy_accuracy(zero, y_np[test_idx], args.tolerance)[1],
                    "train_mean_test_abs_accuracy": energy_accuracy(train_mean, y_np[test_idx], args.tolerance)[0],
                    "train_mean_test_relative_accuracy": energy_accuracy(train_mean, y_np[test_idx], args.tolerance)[1],
                    "exact_rule_max_error": float(np.max(np.abs(exact - y_np[test_idx])))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(.9, .98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if not np.array_equal(checkpoint["train_indices"], train_idx):
            raise ValueError("Resume split differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        if args.snapshot_every:
            for path in (args.outdir / "snapshots").glob("step_*.pt"):
                if int(path.stem.split("_")[-1]) > start_step:
                    path.unlink()
    train_x = x[train_t]
    train_mask = None if site_mask is None else site_mask[train_t]
    train_target = target[train_t]
    if args.family != "pericyclic":
        center = torch.as_tensor(mean, device=device)
        scale = torch.as_tensor(std, device=device)
        transformed_target = train_target.abs().log() if args.target_transform == "log" else train_target
        normalized_target = (transformed_target - center) / scale
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = {"family": args.family, "dataset": str(path), "n_samples": len(x),
              "train_size": len(train_idx), "test_size": len(test_idx),
              "seed": args.seed, "steps": args.steps, "train_fraction": args.train_fraction,
              "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every,
              "snapshot_every": args.snapshot_every,
              "tolerance_abs_beta": args.tolerance, "lr": args.lr,
              "energy_target_transform": args.target_transform if args.family != "pericyclic" else None,
              "weight_decay": args.weight_decay,
              "architecture": "one-layer four-head d128 MLP512 ReLU no LayerNorm",
              "input": "16 binary n tokens" if args.family != "pericyclic" else "8x8 TS matrix row tokens plus site mask",
              "target": "HOMO and LUMO energies" if args.family != "pericyclic" else "thermal allowed label",
              "device": torch.cuda.get_device_name(device), "baselines": baseline}
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    fields = ["step", "elapsed_seconds", "train_loss", "test_loss", "train_accuracy",
              "test_accuracy", "train_relative_accuracy", "test_relative_accuracy",
              "train_rmse", "test_rmse", "test_small_n_rmse",
              "test_small_n_relative_accuracy"]
    metrics = args.outdir / "metrics.csv"
    start = time.perf_counter()
    append = start_step > 0 and metrics.exists()
    elapsed_offset = 0.0
    if append:
        with metrics.open() as prior_handle:
            prior_rows = list(csv.DictReader(prior_handle))
        if prior_rows:
            elapsed_offset = float(prior_rows[-1]["elapsed_seconds"])
    with metrics.open("a" if append else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        if args.snapshot_every and start_step == 0:
            write_snapshot(args.outdir / "snapshots" / "step_000000.pt", model, 0,
                           train_idx, test_idx, mean, std)
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x, train_mask)
            if args.family == "pericyclic":
                loss = F.binary_cross_entropy_with_logits(logits, train_target)
            else:
                loss = ((logits - normalized_target) ** 2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.snapshot_every and step % args.snapshot_every == 0:
                write_snapshot(args.outdir / "snapshots" / f"step_{step:06d}.pt", model,
                               step, train_idx, test_idx, mean, std)
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                torch.cuda.synchronize(device)
                record = evaluate(args.family, model, x, site_mask, target, y_np, n,
                                  train_idx, test_idx, mean, std, args.tolerance,
                                  args.target_transform,
                                  step, elapsed_offset + time.perf_counter() - start)
                writer.writerow(record)
                handle.flush()
                print(json.dumps(record), flush=True)
            if step % args.checkpoint_every == 0:
                write_checkpoint(args.outdir / "latest.pt", model, optimizer, scheduler,
                                 step, train_idx, test_idx, mean, std)
    write_checkpoint(args.outdir / "last.pt", model, optimizer, scheduler,
                     args.steps, train_idx, test_idx, mean, std)
    if args.snapshot_every and args.steps % args.snapshot_every:
        write_snapshot(args.outdir / "snapshots" / f"step_{args.steps:06d}.pt", model,
                       args.steps, train_idx, test_idx, mean, std)
    (args.outdir / "timing.json").write_text(json.dumps({"total_seconds": elapsed_offset + time.perf_counter() - start}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="Save model-only states at this step interval, plus step 0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=.30)
    parser.add_argument("--tolerance", type=float, default=.05)
    parser.add_argument("--target-transform", choices=("linear", "log"), default="linear")
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
