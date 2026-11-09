#!/usr/bin/env python3
"""Compare two initializations on the identical molecular split."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import HERE

KINDS = ("matrix_rows", "pi_graph", "molecular_graph", "smiles")


def rows(kind, seed):
    path = HERE / f"runs_20k_seed{seed}" / kind / "history.csv"
    with path.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#9467bd", "#ff7f0e", "#2ca02c", "#d62728"]
    results = {}
    for kind, color in zip(KINDS, colors):
        histories = [rows(kind, seed) for seed in (0, 1)]
        x = [r["step"] for r in histories[0]]
        for ax, metric, title in zip(axes, ("test_accuracy_rel_1pct", "test_mae"),
                                     ("Held-out 1% accuracy", "Held-out gap MAE (|β|)")):
            values = np.array([[r[metric] for r in history] for history in histories])
            ax.plot(x, values.mean(0), color=color, lw=2, label=kind)
            ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=.13)
            ax.set_xlabel("Gradient steps")
            ax.set_ylabel(title)
        results[kind] = {metric: {
            "seed0": histories[0][-1][metric], "seed1": histories[1][-1][metric],
            "mean": float(np.mean([h[-1][metric] for h in histories])),
            "range": [float(min(h[-1][metric] for h in histories)),
                      float(max(h[-1][metric] for h in histories))],
        } for metric in ("train_accuracy_rel_1pct", "test_accuracy_rel_1pct",
                         "test_accuracy_rel_5pct", "train_mae", "test_mae")}
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = HERE / "plots_20k_seed0"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "seed_comparison.png", dpi=180)
    (HERE / "seed_comparison.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
