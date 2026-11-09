#!/usr/bin/env python3
"""Compare constant and decayed learning rates for the two bit-input models."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_gap import final_prediction
from train_gap import ROOT, load_data, score


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n, truth = load_data()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    report = {}
    for row, variant in enumerate(("log_bits", "log_flatbits")):
        report[variant] = {}
        for name, run_root, linestyle in (("constant", ROOT / "runs", "-"),
                                          ("cosine", ROOT / "runs_decay", "--")):
            data = rows(run_root / variant / "metrics.csv")
            step = np.array([float(item["step"]) for item in data])
            for col, pct in enumerate((1, 5)):
                ax = axes[row, col]
                for subset, color in (("train", "#2563eb"), ("test", "#ea580c")):
                    accuracy = [float(item[f"{subset}_accuracy_rel_{pct}pct"]) for item in data]
                    ax.plot(step, accuracy, color=color, ls=linestyle,
                            label=f"{name} {subset}")
                ax.set(xlabel="Full-batch step", ylabel="Accuracy", ylim=(0, 1.02),
                       title=f"{variant.replace('_', ' ')} · gap within {pct}%")
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            pred, train, test = final_prediction(variant, device, n, run_root)
            report[variant][name] = {
                "train_final": score(pred, truth, train),
                "test_final": score(pred, truth, test),
                "test_best_1pct": max(float(item["test_accuracy_rel_1pct"]) for item in data),
            }
    fig.suptitle("Circulant gap · constant versus cosine learning rate")
    fig.savefig(ROOT / "decay_comparison.png", dpi=175)
    plt.close(fig)
    architecture_fig, architecture_axes = plt.subplots(3, 2, figsize=(12, 11),
                                                       layout="constrained")
    for row, variant in enumerate(("log_bits", "log_posbits", "log_flatbits")):
        data = rows(ROOT / "runs_decay" / variant / "metrics.csv")
        step = np.array([float(item["step"]) for item in data])
        for col, pct in enumerate((1, 5)):
            ax = architecture_axes[row, col]
            for subset, color in (("train", "#2563eb"), ("test", "#ea580c")):
                accuracy = [float(item[f"{subset}_accuracy_rel_{pct}pct"]) for item in data]
                ax.plot(step, accuracy, color=color, label=subset)
            ax.set(xlabel="Full-batch step", ylabel="Accuracy", ylim=(0, 1.02),
                   title=f"{variant.replace('_', ' ')} · gap within {pct}%")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        if variant == "log_posbits":
            pred, train, test = final_prediction(variant, device, n, ROOT / "runs_decay")
            report[variant] = {"cosine": {
                "train_final": score(pred, truth, train),
                "test_final": score(pred, truth, test),
                "test_best_1pct": max(float(item["test_accuracy_rel_1pct"]) for item in data),
            }}
    architecture_fig.suptitle("Circulant gap · tokenization and architecture controls")
    architecture_fig.savefig(ROOT / "architecture_comparison.png", dpi=175)
    plt.close(architecture_fig)
    (ROOT / "decay_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({v: {name: round(x["test_final"]["accuracy_rel_1pct"], 4)
                          for name, x in variants.items()}
                      for v, variants in report.items()}, indent=2))


if __name__ == "__main__":
    main()
