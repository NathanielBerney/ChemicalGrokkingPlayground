#!/usr/bin/env python3
"""Reproduce Omnigrok Figure 5 on PyG QM9 polarizability.

The network, first-N QM9 subset, 50/50 split, optimizers, learning rate,
batch size, initialization scaling, and fixed-norm landscape rule follow the
authors' public QM9 notebooks. This script records true per-molecule MSE
alongside the notebooks' batch-mean-divided-by-graph-count quantity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_add_pool


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "qm9"
TARGET = 1  # Isotropic polarizability, alpha, in Bohr radii cubed.
SOURCE_COMMIT = "3a0a57346230a7255e5390fcd565812d00f1a281"


class ExampleNet(nn.Module):
    """Exact two-NNConv architecture in the authors' QM9 notebooks."""

    def __init__(self):
        super().__init__()
        conv1_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 11 * 32))
        conv2_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32 * 16))
        self.conv1 = NNConv(11, 32, conv1_net)
        self.conv2 = NNConv(32, 16, conv2_net)
        self.fc_1 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, batch):
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        x = F.relu(self.conv2(x, batch.edge_index, batch.edge_attr))
        x = global_add_pool(x, batch.batch)
        return self.out(F.relu(self.fc_1(x)))


def weight_norm(model):
    return torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters()))


@torch.no_grad()
def rescale(model, factor):
    for parameter in model.parameters():
        parameter.mul_(factor)


def get_data(total_size: int, seed: int, batch_size: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = QM9(root=str(DATA_ROOT))
    if total_size > len(dataset) or total_size < 2 or total_size % 2:
        raise ValueError(f"total_size must be even and between 2 and {len(dataset)}")
    subset = dataset[:total_size]
    train, test = random_split(subset, [total_size // 2, total_size // 2])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, train.indices, test.indices


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        squared_error += F.mse_loss(model(batch), batch.y[:, TARGET:TARGET + 1],
                                    reduction="sum").item()
        count += batch.num_graphs
    return squared_error / count


def save_checkpoint(path, model, optimizer, epoch, step, block, device):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "next_epoch": epoch + 1, "step": step, "block": block,
             "torch_rng": torch.get_rng_state(),
             "numpy_rng": np.random.get_state()}
    if device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state(device)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    np.random.set_state(state["numpy_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    return state["next_epoch"], state["step"], state["block"]


def truncate_csv(path, max_step):
    if not path.exists():
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["step"]) <= max_step]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def save_snapshot(path, model, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"step": step,
             "model": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()}}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def run_grokking(args):
    device = torch.device(args.device)
    train_loader, test_loader, train_ids, test_ids = get_data(
        args.total_size, args.seed, args.batch_size)
    eval_train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size)
    eval_test_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size)
    model = ExampleNet().to(device)
    rescale(model, args.alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.0)
    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    run_dir = ROOT / "runs" / f"grokking_total{args.total_size}_alpha{args.alpha:g}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"total_size": args.total_size, "train_size": args.total_size // 2,
              "test_size": args.total_size // 2, "alpha": args.alpha,
              "seed": args.seed, "batch_size": args.batch_size, "device": str(device),
              "epochs": args.epochs or int(5_000_000 / args.total_size),
              "archive_every_steps": args.archive_every_steps,
              "source_commit": SOURCE_COMMIT, "target_index": TARGET,
              "optimizer": "AdamW(lr=0.001, weight_decay=0)",
              "train_indices": list(map(int, train_ids)),
              "test_indices": list(map(int, test_ids))}
    config_path = run_dir / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ("total_size", "alpha", "seed", "batch_size", "epochs",
                    "archive_every_steps"):
            old_value = old.get(key, 0) if key == "archive_every_steps" else old[key]
            if old_value != config[key]:
                raise ValueError(f"Existing run has a different {key}: {old_value}")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    checkpoint = run_dir / "latest.pt"
    epoch_start = step = 0
    block = {"count": 0, "train": 0.0, "test": 0.0}
    if checkpoint.exists():
        epoch_start, step, block = load_checkpoint(checkpoint, model, optimizer, device)
        truncate_csv(run_dir / "batch20_metrics.csv", step)
        truncate_csv(run_dir / "full_metrics.csv", step)
        for path in (run_dir / "snapshots").glob("step_*.pt"):
            if int(path.stem.split("_")[-1]) > step:
                path.unlink()
        print(f"Resuming {run_dir.name} at epoch {epoch_start}, step {step}", flush=True)
    elif (run_dir / "full_metrics.csv").exists():
        raise ValueError(f"Existing metrics without checkpoint in {run_dir}")

    batch_path = run_dir / "batch20_metrics.csv"
    full_path = run_dir / "full_metrics.csv"
    batch_fields = ["step", "author_scaled_train_mse", "author_scaled_test_mse"]
    full_fields = ["step", "epoch", "train_mse", "test_mse", "weight_norm"]
    with batch_path.open("a" if epoch_start else "w", newline="") as bf, \
         full_path.open("a" if epoch_start else "w", newline="") as ff:
        bw = csv.DictWriter(bf, fieldnames=batch_fields)
        fw = csv.DictWriter(ff, fieldnames=full_fields)
        if not epoch_start:
            bw.writeheader(); fw.writeheader()
            if args.archive_every_steps:
                save_snapshot(run_dir / "snapshots" / "step_000000.pt", model, 0)
            fw.writerow({"step": 0, "epoch": 0,
                         "train_mse": evaluate(model, eval_train_loader, device),
                         "test_mse": evaluate(model, eval_test_loader, device),
                         "weight_norm": weight_norm(model).item()})
            ff.flush()
        start_time = time.monotonic()
        epochs = config["epochs"]
        for epoch in range(epoch_start, epochs):
            for train_batch in train_loader:
                train_batch = train_batch.to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                train_mse = F.mse_loss(model(train_batch),
                                       train_batch.y[:, TARGET:TARGET + 1])
                train_mse.backward()
                optimizer.step()
                # The notebook draws a new shuffled test batch after every update.
                test_batch = next(iter(test_loader)).to(device)
                with torch.inference_mode():
                    model.eval()
                    test_mse = F.mse_loss(model(test_batch),
                                          test_batch.y[:, TARGET:TARGET + 1])
                step += 1
                if args.archive_every_steps and step % args.archive_every_steps == 0:
                    save_snapshot(run_dir / "snapshots" / f"step_{step:06d}.pt", model, step)
                block["count"] += 1
                block["train"] += train_mse.item() / train_batch.num_graphs
                block["test"] += test_mse.item() / test_batch.num_graphs
                if block["count"] == 20:
                    bw.writerow({"step": step,
                                 "author_scaled_train_mse": block["train"] / 20,
                                 "author_scaled_test_mse": block["test"] / 20})
                    block = {"count": 0, "train": 0.0, "test": 0.0}
            if (epoch + 1) % args.eval_every_epochs == 0 or epoch + 1 == epochs:
                row = {"step": step, "epoch": epoch + 1,
                       "train_mse": evaluate(model, eval_train_loader, device),
                       "test_mse": evaluate(model, eval_test_loader, device),
                       "weight_norm": weight_norm(model).item()}
                fw.writerow(row); ff.flush(); bf.flush()
                if (epoch + 1) % max(args.eval_every_epochs * 5, 1) == 0:
                    print(run_dir.name, row, "elapsed_s", round(time.monotonic()-start_time, 1),
                          flush=True)
            if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch + 1 == epochs:
                bf.flush(); ff.flush()
                save_checkpoint(checkpoint, model, optimizer, epoch, step, block, device)
        if args.archive_every_steps and step % args.archive_every_steps:
            save_snapshot(run_dir / "snapshots" / f"step_{step:06d}.pt", model, step)
    (run_dir / "last.pt").write_bytes(checkpoint.read_bytes())
    print(f"Finished {run_dir.name}: {step} steps", flush=True)


def run_landscape(args):
    device = torch.device(args.device)
    sizes = [int(value / 2) * 2 for value in np.logspace(1, 4, 22)]
    alphas = list(np.logspace(-1, 1, 21))
    if args.cuts:
        jobs = [(3000, float(alpha)) for alpha in alphas]
        offset = len(sizes) * len(alphas)
    else:
        jobs = [(size, float(alpha)) for size in sizes for alpha in alphas]
        offset = 0
    selected = [(i + offset, size, alpha) for i, (size, alpha) in enumerate(jobs)
                if i % args.workers == args.worker_index]
    if args.reverse:
        selected.reverse()
    if args.max_jobs:
        selected = selected[:args.max_jobs]
    out_dir = ROOT / "runs" / "landscape"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = QM9(root=str(DATA_ROOT))
    start = time.monotonic()
    for n, (job_index, size, alpha) in enumerate(selected, 1):
        path = out_dir / f"size_{size:05d}_alpha_{alpha:.6f}.json"
        if path.exists() and not args.force:
            continue
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        subset = data[:size]
        train, test = random_split(subset, [size // 2, size // 2])
        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        test_loader = DataLoader(test, batch_size=32, shuffle=True)
        model = ExampleNet().to(device)
        rescale(model, alpha)
        initial_norm = weight_norm(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        epochs = int(100_000 / size)
        best = {"author_scaled_train_mse": math.inf,
                "author_scaled_test_mse": math.inf,
                "train_mse": math.inf, "test_mse": math.inf}
        steps = 0
        for _ in range(epochs):
            train_batch_means = 0.0
            train_squared_sum = 0.0
            train_count = 0
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(batch), batch.y[:, TARGET:TARGET + 1])
                loss.backward()
                train_batch_means += loss.item()
                train_squared_sum += loss.item() * batch.num_graphs
                train_count += batch.num_graphs
                optimizer.step()
                rescale(model, initial_norm / weight_norm(model))
                steps += 1
            test_batch_means = 0.0
            test_squared_sum = 0.0
            test_count = 0
            model.eval()
            with torch.inference_mode():
                for batch in test_loader:
                    batch = batch.to(device)
                    value = F.mse_loss(model(batch), batch.y[:, TARGET:TARGET + 1]).item()
                    test_batch_means += value
                    test_squared_sum += value * batch.num_graphs
                    test_count += batch.num_graphs
            values = {"author_scaled_train_mse": train_batch_means / train_count,
                      "author_scaled_test_mse": test_batch_means / test_count,
                      "train_mse": train_squared_sum / train_count,
                      "test_mse": test_squared_sum / test_count}
            for key, value in values.items():
                best[key] = min(best[key], value)
        result = {"job_index": job_index, "total_size": size,
                  "train_size": size // 2, "test_size": size // 2,
                  "alpha": alpha, "seed": args.seed, "epochs": epochs,
                  "steps": steps, "initial_weight_norm": initial_norm.item(),
                  "final_weight_norm": weight_norm(model).item(),
                  "best": best, "source_commit": SOURCE_COMMIT}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(path)
        print(f"worker {args.worker_index}: {n}/{len(selected)} N={size} alpha={alpha:.3g} "
              f"test={best['author_scaled_test_mse']:.4g} "
              f"elapsed_s={time.monotonic()-start:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    grok = sub.add_parser("grokking", help="Train the Figure 5(d) comparison")
    grok.add_argument("--total-size", type=int, default=400,
                      help="First N QM9 molecules, split 50/50; 400 gives 200 train")
    grok.add_argument("--alpha", type=float, required=True)
    grok.add_argument("--seed", type=int, default=0)
    grok.add_argument("--batch-size", type=int, default=32)
    grok.add_argument("--epochs", type=int, default=0)
    grok.add_argument("--eval-every-epochs", type=int, default=100)
    grok.add_argument("--checkpoint-every-epochs", type=int, default=1000)
    grok.add_argument("--archive-every-steps", type=int, default=0)
    grok.add_argument("--run-suffix", default="")
    grok.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    landscape = sub.add_parser("landscape", help="Train Figure 5(a–c) fixed-norm grid")
    landscape.add_argument("--worker-index", type=int, default=0)
    landscape.add_argument("--workers", type=int, default=1)
    landscape.add_argument("--max-jobs", type=int, default=0)
    landscape.add_argument("--reverse", action="store_true",
                           help="Process this worker's assigned points from largest N to smallest")
    landscape.add_argument("--cuts", action="store_true",
                           help="Run the extra N=3000 alpha slice for Figure 5(c)")
    landscape.add_argument("--force", action="store_true")
    landscape.add_argument("--seed", type=int, default=0)
    landscape.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "grokking":
        if args.alpha <= 0 or args.eval_every_epochs < 1 or args.checkpoint_every_epochs < 1:
            parser.error("alpha and evaluation/checkpoint intervals must be positive")
        if args.archive_every_steps < 0 or (args.run_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_suffix)):
            parser.error("archive interval must be nonnegative and run suffix must use letters, digits, _ or -")
        run_grokking(args)
    else:
        if args.workers < 1 or not 0 <= args.worker_index < args.workers:
            parser.error("worker-index must be in 0..workers-1")
        run_landscape(args)


if __name__ == "__main__":
    main()
