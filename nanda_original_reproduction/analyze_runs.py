#!/usr/bin/env python3
"""Plot original-task grokking curves and summarize all completed seeds."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def analyze(root: Path) -> dict:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained")
    report = {}
    for run in sorted(root.glob("run_seed*")):
        metrics_path = run / "metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        step = np.array([int(r["step"]) for r in rows])
        elapsed = np.array([float(r["elapsed_seconds"]) for r in rows])
        train = np.array([float(r["train_accuracy"]) for r in rows])
        test = np.array([float(r["test_accuracy"]) for r in rows])
        train_loss = np.array([float(r["train_loss"]) for r in rows])
        test_loss = np.array([float(r["test_loss"]) for r in rows])
        gini_e = np.array([float(r["embedding_fourier_gini"]) for r in rows])
        gini_l = np.array([float(r["neuron_logit_fourier_gini"]) for r in rows])
        first_train = np.flatnonzero(train >= 0.99)
        first_test = np.flatnonzero(test >= 0.99)
        seed = run.name.replace("run_seed", "")
        report[seed] = {
            "last_step": int(step[-1]),
            "first_train_accuracy_99pct_step": int(step[first_train[0]]) if len(first_train) else None,
            "first_train_accuracy_99pct_seconds": float(elapsed[first_train[0]]) if len(first_train) else None,
            "first_test_accuracy_99pct_step": int(step[first_test[0]]) if len(first_test) else None,
            "first_test_accuracy_99pct_seconds": float(elapsed[first_test[0]]) if len(first_test) else None,
            "last_train_accuracy": float(train[-1]),
            "last_test_accuracy": float(test[-1]),
            "last_embedding_fourier_gini": float(gini_e[-1]),
            "last_neuron_logit_fourier_gini": float(gini_l[-1]),
            "last_elapsed_seconds": float(elapsed[-1]),
        }
        line, = axes[0].plot(step, test, label=f"seed {seed}")
        color = line.get_color()
        axes[0].plot(step, train, color=color, ls="--", alpha=0.65)
        axes[1].plot(step, np.maximum(test_loss, 1e-10), color=color)
        axes[1].plot(step, np.maximum(train_loss, 1e-10), color=color, ls="--", alpha=0.65)
        axes[2].plot(step, gini_e, color=color)
        axes[2].plot(step, gini_l, color=color, ls="--", alpha=0.7)
    axes[0].axhline(0.99, color="#64748b", ls=":", label="99%")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Nanda et al. mod-113 addition: delayed generalization")
    seed_handles = axes[0].get_legend_handles_labels()[0]
    style_handles = [Line2D([0], [0], color="black", ls="-", label="test / embedding"),
                     Line2D([0], [0], color="black", ls="--", label="train / neuron-logit")]
    axes[0].legend(handles=seed_handles + style_handles, ncol=4, fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)
    axes[2].set_xlabel("Full-batch optimizer step")
    axes[2].set_ylabel("Fourier Gini coefficient")
    axes[2].grid(alpha=0.2)
    fig.savefig(root / "reproduction_curves.png", dpi=170)
    plt.close(fig)
    (root / "reproduction_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(Path(__file__).resolve().parent), indent=2))
