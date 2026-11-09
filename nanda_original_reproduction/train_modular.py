#!/usr/bin/env python3
"""Self-contained reproduction of Nanda et al.'s mod-113 grokking run.

Architecture and training parameters follow the authors' mainline setup:
one causal ReLU transformer layer, d=128, four heads, MLP=512, no LayerNorm;
30% of all mod-113 addition pairs; full-batch AdamW with lr=1e-3, wd=1,
betas=(0.9, 0.98), ten-step warmup, and 40,000 steps. Test evaluation is
less frequent than in the published reference code; it does not change
training updates. No wandb service is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


P = 113
D = 128
HEADS = 4
MLP = 512


class OriginalStyleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(D, P + 1) / math.sqrt(D))
        self.W_pos = nn.Parameter(torch.randn(3, D) / math.sqrt(D))
        self.W_Q = nn.Parameter(torch.randn(HEADS, D // HEADS, D) / math.sqrt(D))
        self.W_K = nn.Parameter(torch.randn(HEADS, D // HEADS, D) / math.sqrt(D))
        self.W_V = nn.Parameter(torch.randn(HEADS, D // HEADS, D) / math.sqrt(D))
        self.W_O = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.W_in = nn.Parameter(torch.randn(MLP, D) / math.sqrt(D))
        self.b_in = nn.Parameter(torch.zeros(MLP))
        self.W_out = nn.Parameter(torch.randn(D, MLP) / math.sqrt(D))
        self.b_out = nn.Parameter(torch.zeros(D))
        self.W_U = nn.Parameter(torch.randn(D, P + 1) / math.sqrt(P + 1))
        self.register_buffer("causal_mask", torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.W_E[:, tokens].permute(1, 2, 0) + self.W_pos
        q = torch.einsum("hmd,btd->bhtm", self.W_Q, x)
        k = torch.einsum("hmd,btd->bhtm", self.W_K, x)
        v = torch.einsum("hmd,btd->bhtm", self.W_V, x)
        scores = torch.einsum("bhtm,bhsm->bhts", q, k) / math.sqrt(D // HEADS)
        scores = scores.masked_fill(self.causal_mask, -1e9)
        attention = scores.softmax(dim=-1)
        z = torch.einsum("bhts,bhsm->bhtm", attention, v)
        z = z.permute(0, 2, 1, 3).reshape(tokens.shape[0], 3, D)
        x = x + torch.einsum("dm,btm->btd", self.W_O, z)
        hidden = F.relu(torch.einsum("md,btd->btm", self.W_in, x) + self.b_in)
        x = x + torch.einsum("dm,btm->btd", self.W_out, hidden) + self.b_out
        # The reference model trains over p+1 logits, including the unused
        # '=' token class; only labels 0..p-1 occur.
        return x[:, -1] @ self.W_U


def dataset(seed: int, device: torch.device):
    pairs = [(a, b) for a in range(P) for b in range(P)]
    random.Random(seed).shuffle(pairs)
    cut = int(0.30 * len(pairs))
    def tensors(part):
        array = np.asarray(part, dtype=np.int64)
        tokens = np.column_stack((array, np.full(len(array), P, dtype=np.int64)))
        labels = (array[:, 0] + array[:, 1]) % P
        return torch.from_numpy(tokens).to(device), torch.from_numpy(labels).to(device)
    return tensors(pairs[:cut]), tensors(pairs[cut:])


def cross_entropy_high_precision(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = logits.double()
    return (torch.logsumexp(values, dim=-1) - values.gather(1, labels[:, None]).squeeze(1)).mean()


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = len(values)
    return float(((2 * np.arange(1, n + 1) - n - 1) @ values) / (n * values.sum()))


@torch.no_grad()
def fourier_ginis(model: OriginalStyleTransformer) -> tuple[float, float]:
    embedding = model.W_E[:, :P].detach().cpu().numpy()
    neuron_logit = (model.W_U[:, :P].T @ model.W_out).detach().cpu().numpy()
    embedding_power = (np.abs(np.fft.rfft(embedding, axis=1)) ** 2).sum(axis=0)[1:]
    logit_power = (np.abs(np.fft.rfft(neuron_logit, axis=0)) ** 2).sum(axis=1)[1:]
    return gini(embedding_power), gini(logit_power)


@torch.no_grad()
def evaluate(model, tokens, labels, batch_size=4096):
    model.eval()
    loss_sum = 0.0
    correct = 0
    for token_batch, label_batch in zip(tokens.split(batch_size), labels.split(batch_size)):
        logits = model(token_batch)
        loss_sum += float(cross_entropy_high_precision(logits, label_batch)) * len(label_batch)
        correct += int((logits.argmax(dim=-1) == label_batch).sum())
    return loss_sum / len(labels), correct / len(labels)


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def save_figure_snapshot(path: Path, model, step: int):
    """Save a compact model-only snapshot for Fourier progress plots."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{args.gpu}")
    (train_x, train_y), (test_x, test_y) = dataset(args.seed, device)
    model = OriginalStyleTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0,
                                   betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = {"paper": "https://arxiv.org/abs/2301.05217",
              "reference_code": "https://github.com/mechanistic-interpretability-grokking/progress-measures-paper",
              "p": P, "d_model": D, "heads": HEADS, "mlp": MLP,
              "train_size": len(train_x), "test_size": len(test_x),
              "seed": args.seed, "steps": args.steps, "eval_every": args.eval_every,
              "checkpoint_every": args.checkpoint_every, "archive_every": args.archive_every,
              "device": torch.cuda.get_device_name(device),
              "training": "full-batch AdamW, lr 0.001, wd 1.0, betas (0.9,0.98), warmup 10, no LayerNorm"}
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    snapshots = args.outdir / "figure_snapshots"
    if args.archive_every:
        snapshots.mkdir(exist_ok=True)
        if start_step == 0:
            save_figure_snapshot(snapshots / "step_000000.pt", model, 0)
    fields = ["step", "elapsed_seconds", "train_loss", "test_loss", "train_accuracy",
              "test_accuracy", "embedding_fourier_gini", "neuron_logit_fourier_gini"]
    metrics_path = args.outdir / "metrics.csv"
    append = args.resume is not None and metrics_path.exists()
    start = time.perf_counter()
    with metrics_path.open("a" if append else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for step in range(start_step, args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy_high_precision(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            completed = step + 1
            if completed == 1 or completed % args.eval_every == 0 or completed == args.steps:
                train_loss, train_accuracy = evaluate(model, train_x, train_y)
                test_loss, test_accuracy = evaluate(model, test_x, test_y)
                gini_embedding, gini_logit = fourier_ginis(model)
                torch.cuda.synchronize(device)
                record = {"step": completed, "elapsed_seconds": time.perf_counter() - start,
                          "train_loss": train_loss, "test_loss": test_loss,
                          "train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                          "embedding_fourier_gini": gini_embedding,
                          "neuron_logit_fourier_gini": gini_logit}
                writer.writerow(record)
                handle.flush()
                print(json.dumps(record), flush=True)
            if completed % args.checkpoint_every == 0:
                save_checkpoint(args.outdir / "latest.pt", model, optimizer, scheduler, completed)
            if args.archive_every and completed % args.archive_every == 0:
                save_figure_snapshot(snapshots / f"step_{completed:06d}.pt", model, completed)
    save_checkpoint(args.outdir / "last.pt", model, optimizer, scheduler, args.steps)
    (args.outdir / "timing.json").write_text(json.dumps({"total_seconds": time.perf_counter() - start}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("run_seed0"))
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--archive-every", type=int, default=0,
                        help="Save compact model-only snapshots every N steps (0 disables).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
