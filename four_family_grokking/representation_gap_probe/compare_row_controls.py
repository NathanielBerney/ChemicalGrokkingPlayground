#!/usr/bin/env python3
"""Compare row-depth, learning-rate, and permutation controls over 100k steps."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import HERE

CASES = (
    ("one_layer", "long_100k_seed0", "matrix_rows", "One layer, original decay", "#7b4ca0"),
    ("one_layer_slow_decay", "long_100k_slowdecay_seed0", "matrix_rows", "One layer, slow decay", "#b78acb"),
    ("two_layers", "long_100k_seed0", "matrix_rows_2layer", "Two layers, original decay", "#1f77b4"),
    ("perm_aug", "long_100k_seed0", "matrix_rows_permaug", "One layer + random site order", "#d07825"),
    ("perm_aug_slow_decay", "long_100k_slowdecay_seed0", "matrix_rows_permaug",
     "Random site order + slow decay", "#e7ad70"),
)


def read(path):
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    out = HERE / "long_100k_seed0" / "plots"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    summary = {}
    for key, runset, kind, label, color in CASES:
        p = HERE / runset / kind / "history.csv"
        if not p.exists():
            continue
        rows = read(p)
        if int(rows[-1]["step"]) != 100000:
            continue
        x = [r["step"] for r in rows]
        for col, split in enumerate(("train", "test")):
            for row, (metric, title) in enumerate((("accuracy_rel_1pct", "1% accuracy"),
                                                   ("mae", "Gap MAE (|β|)"))):
                axes[row, col].plot(x, [r[f"{split}_{metric}"] for r in rows],
                                    lw=1.8, color=color, label=label)
                axes[row, col].set_title(f"{split.capitalize()} {title}")
        summary[key] = {m: rows[-1][m] for m in ("train_accuracy_rel_1pct", "test_accuracy_rel_1pct",
                                                 "train_mae", "test_mae")}
    for ax in axes[-1]:
        ax.set_xlabel("Gradient steps")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Row input: extra steps, depth, learning-rate schedule, and site-order augmentation")
    fig.tight_layout()
    fig.savefig(out / "row_controls.png", dpi=180)
    (HERE / "long_100k_seed0" / "row_controls.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
