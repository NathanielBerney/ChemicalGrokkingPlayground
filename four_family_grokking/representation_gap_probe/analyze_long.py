#!/usr/bin/env python3
"""Compare 100k-step learning and ask whether train accuracy actually plateaus."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE / "long_100k_seed0"
KINDS = ("matrix_rows", "matrix_rows_2layer", "pi_graph", "molecular_graph",
         "smiles", "matrix_rows_permaug")
COLORS = {"matrix_rows": "#9467bd", "pi_graph": "#ff7f0e",
          "molecular_graph": "#2ca02c", "smiles": "#d62728",
          "matrix_rows_permaug": "#2f4f9f", "matrix_rows_2layer": "#a163dc"}
NAMES = {"matrix_rows": "Matrix rows", "pi_graph": "π graph", "molecular_graph": "Atom graph + π",
         "smiles": "SMILES + π", "matrix_rows_permaug": "Matrix rows + site permutations",
         "matrix_rows_2layer": "Matrix rows, two layers"}


def history(path):
    with path.open() as f:
        return {int(r["step"]): {k: float(v) for k, v in r.items()}
                for r in csv.DictReader(f)}


def main():
    available = [k for k in KINDS if (ROOT / k / "history.csv").exists()]
    h = {k: history(ROOT / k / "history.csv") for k in available}
    complete = [k for k in available if 100000 in h[k]]
    if not complete:
        raise RuntimeError("No 100k run has completed yet")
    plot = ROOT / "plots"
    plot.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex="col")
    metrics = (("accuracy_rel_1pct", "1% relative-gap accuracy"),
               ("accuracy_rel_5pct", "5% relative-gap accuracy"),
               ("mae", "Gap MAE (|β|)"))
    summary = {}
    for kind in complete:
        rows = h[kind]
        x = sorted(rows)
        for col, split in enumerate(("train", "test")):
            for row, (metric, label) in enumerate(metrics):
                axes[row, col].plot(x, [rows[s][f"{split}_{metric}"] for s in x],
                                    lw=1.7, color=COLORS[kind], label=NAMES[kind])
                axes[row, col].set_ylabel(label)
                axes[row, col].set_title(split.capitalize())
                axes[row, col].axvline(20000, color="gray", ls="--", lw=.8)
        summary[kind] = {
            "milestones": {str(step): {m: rows[step][m] for m in
                             ("train_accuracy_rel_1pct", "test_accuracy_rel_1pct",
                              "train_accuracy_rel_5pct", "test_accuracy_rel_5pct",
                              "train_mae", "test_mae", "lr")}
                           for step in (20000, 50000, 80000, 90000, 100000) if step in rows},
            "last_20k_change": {
                m: rows[100000][m] - rows[80000][m]
                for m in ("train_accuracy_rel_1pct", "test_accuracy_rel_1pct",
                          "train_mae", "test_mae")
            } if 80000 in rows else {},
        }
    for ax in axes[-1]:
        ax.set_xlabel("Gradient steps")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Extended training: fixed 20k LR decay, then constant 0.1× LR")
    fig.tight_layout()
    fig.savefig(plot / "extended_training_dynamics.png", dpi=180)
    plt.close(fig)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    for kind, value in summary.items():
        p = value["milestones"]["100000"]
        print(kind, f"train1%={p['train_accuracy_rel_1pct']:.3f}",
              f"test1%={p['test_accuracy_rel_1pct']:.3f}",
              f"trainMAE={p['train_mae']:.4f}", f"testMAE={p['test_mae']:.4f}",
              "last20k", value["last_20k_change"])


if __name__ == "__main__":
    main()
