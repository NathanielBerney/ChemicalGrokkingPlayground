#!/usr/bin/env python3
"""Compare raw-energy and log-magnitude energy training on identical splits."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def series(run_root, family, field):
    with (HERE / run_root / family / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return np.array([float(r[field]) for r in rows])


def main():
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, layout="constrained")
    report = {}
    for ax, family in zip(axes, ("circulant", "mobius", "linear")):
        report[family] = {}
        for run_root, label, color in (("runs", "raw energy", "#2563eb"),
                                       ("runs_log", "log magnitude", "#ea580c")):
            step = series(run_root, family, "step")
            train = series(run_root, family, "train_relative_accuracy")
            test = series(run_root, family, "test_relative_accuracy")
            ax.plot(step, test, color=color, label=f"{label} · test")
            ax.plot(step, train, color=color, ls="--", alpha=.6, label=f"{label} · train")
            report[family][label] = {"last_train_relative_accuracy": float(train[-1]),
                                     "last_test_relative_accuracy": float(test[-1]),
                                     "best_test_relative_accuracy": float(np.max(test))}
        ax.set(ylabel="Both energies within 5%", title=family.capitalize(), ylim=(0, 1.03))
        ax.grid(alpha=.2)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Full-batch optimizer step")
    fig.suptitle("Energy objective comparison · same models and train/test splits")
    (HERE / "plots").mkdir(exist_ok=True)
    fig.savefig(HERE / "plots" / "energy_objective_comparison.png", dpi=175)
    plt.close(fig)
    (HERE / "objective_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
