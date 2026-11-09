#!/usr/bin/env python3
"""Re-score completed frontier models with size-independent relative errors."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_family import BitSizeTransformer, load_data, predictions


HERE = Path(__file__).resolve().parent
FAMILIES = ("circulant", "mobius", "linear")


def metrics(pred, target, idx):
    relative = np.max(np.abs(pred[idx] - target[idx]) / np.abs(target[idx]), axis=1)
    return {"both_within_1pct": float(np.mean(relative < .01)),
            "both_within_2pct": float(np.mean(relative < .02)),
            "both_within_5pct": float(np.mean(relative < .05)),
            "median_worst_energy_relative_error": float(np.median(relative)),
            "p90_worst_energy_relative_error": float(np.quantile(relative, .9))}


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    report = {}
    for family in FAMILIES:
        _, x, mask, _, n, target = load_data(family, device)
        report[family] = {}
        for name, root in (("raw", "runs"), ("log", "runs_log")):
            checkpoint = torch.load(HERE / root / family / "last.pt", map_location=device,
                                    weights_only=False)
            model = BitSizeTransformer().to(device)
            model.load_state_dict(checkpoint["model"])
            output = predictions(model, x, mask)
            values = output * checkpoint["target_std"] + checkpoint["target_mean"]
            pred = (np.exp(np.clip(values, -30, 5)) * np.array([-1, 1])
                    if name == "log" else values)
            train = checkpoint["train_indices"]
            test = checkpoint["test_indices"]
            report[family][name] = {"train": metrics(pred, target, train),
                                    "test": metrics(pred, target, test),
                                    "test_n_le_1000": metrics(pred, target,
                                                             test[n[test] <= 1000])}
    (HERE / "strict_frontier_final.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    xloc = np.arange(len(FAMILIES))
    for ax, threshold in zip(axes, ("both_within_1pct", "both_within_5pct")):
        for offset, name, color in ((-.2, "raw", "#2563eb"), (.2, "log", "#ea580c")):
            scores = [report[family][name]["test"][threshold] for family in FAMILIES]
            ax.bar(xloc + offset, scores, width=.38, label=name, color=color)
        ax.set_xticks(xloc, [f.capitalize() for f in FAMILIES])
        ax.set(ylabel="Held-out accuracy", ylim=(0, 1.05),
               title="Both energies within " + ("1%" if "1pct" in threshold else "5%") + " relative error")
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    fig.suptitle("Final frontier-energy checkpoints · strict relative accuracy")
    fig.savefig(HERE / "plots/strict_frontier_accuracy.png", dpi=175)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
