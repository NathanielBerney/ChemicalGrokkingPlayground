#!/usr/bin/env python3
"""Show the effect of supplying Hückel annotations to graph and SMILES models."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import HERE

KINDS = ("molecular_graph_raw", "molecular_graph", "smiles_raw", "smiles")
COLORS = ("#5cb85c", "#176a35", "#ee9999", "#b51f2b")
LABELS = ("Atom graph only", "Atom graph + π annotations", "SMILES only", "SMILES + π metadata")


def rows(kind, seed):
    with (HERE / f"runs_20k_seed{seed}" / kind / "history.csv").open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metrics = (("train_accuracy_rel_1pct", "Train 1% accuracy"),
               ("test_accuracy_rel_1pct", "Test 1% accuracy"),
               ("train_mae", "Train MAE (|β|)"),
               ("test_mae", "Test MAE (|β|)"))
    results = {}
    for kind, color, label in zip(KINDS, COLORS, LABELS):
        h = [rows(kind, s) for s in (0, 1)]
        x = [r["step"] for r in h[0]]
        results[kind] = {}
        for ax, (metric, title) in zip(axes.ravel(), metrics):
            values = np.asarray([[r[metric] for r in run] for run in h])
            ax.plot(x, values.mean(0), color=color, lw=2, label=label)
            ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=.14)
            ax.set_title(title)
            ax.set_xlabel("Gradient steps")
            results[kind][metric] = {"seed0": values[0, -1].item(),
                                     "seed1": values[1, -1].item(),
                                     "mean": values[:, -1].mean().item()}
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    out = HERE / "plots_20k_seed0"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "structure_only_controls.png", dpi=180)
    (HERE / "structure_only_controls.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
