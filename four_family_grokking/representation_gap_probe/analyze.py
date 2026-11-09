#!/usr/bin/env python3
"""Plot matched training dynamics and probe learned gap representations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_20k_seed0"
KINDS = ("matrix_rows", "pi_graph", "molecular_graph", "smiles")
SPECTRAL_CONTROLS = ("matrix_spectral", "matrix_frontier")
COLORS = {"matrix_rows": "#9467bd", "matrix_spectral": "#1f77b4",
          "matrix_frontier": "#17becf", "pi_graph": "#ff7f0e",
          "molecular_graph": "#2ca02c", "smiles": "#d62728"}
NAMES = {"matrix_rows": "Matrix row attention", "matrix_spectral": "Eigenvalues supplied: flat MLP",
         "matrix_frontier": "HOMO/LUMO energies supplied: readout", "pi_graph": "π-graph GNN",
         "molecular_graph": "Molecular graph GNN", "smiles": "SMILES transformer"}


def read_history(path):
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def threshold_time(history, key, threshold):
    for row in history:
        if row[key] >= threshold:
            return int(row["step"]), round(row["seconds"], 2)
    return None


def pca_probe(hidden, train, test, y):
    mean = hidden[train].mean(0)
    centered = hidden - mean
    # PCA trained on train representations, then project held-out molecules.
    _, s, vt = np.linalg.svd(centered[train], full_matrices=False)
    xy = centered @ vt[:2].T
    ratio = (s[:2] ** 2 / np.sum(s ** 2)).tolist()
    # A fixed ridge linear probe tests whether the gap is linearly accessible.
    scale = centered[train].std(0)
    scale[scale < 1e-6] = 1
    X = centered / scale
    X = np.column_stack((X, np.ones(len(X))))
    target = np.log(y)
    gram = X[train].T @ X[train]
    reg = np.eye(gram.shape[0]) * 10
    reg[-1, -1] = 0
    w = np.linalg.solve(gram + reg, X[train].T @ target[train])
    pred = X @ w
    r2 = 1 - np.sum((pred[test] - target[test]) ** 2) / np.sum((target[test] - target[test].mean()) ** 2)
    return xy, ratio, float(r2)


def main():
    available = [k for k in KINDS if (RUNS / k / "final_probe.npz").exists()]
    if not available:
        raise RuntimeError("No completed runs")
    plot_dir = HERE / "plots_20k_seed0"
    plot_dir.mkdir(exist_ok=True)
    histories = {k: read_history(RUNS / k / "history.csv") for k in available}
    summaries = {}
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex="col")
    for k in available:
        rows = histories[k]
        x = [r["step"] for r in rows]
        for col, split in enumerate(("train", "test")):
            for row, (metric, title) in enumerate((("accuracy_rel_1pct", "1% relative accuracy"),
                                                   ("accuracy_rel_5pct", "5% relative accuracy"),
                                                   ("mae", "MAE (|β|)"))):
                axes[row, col].plot(x, [r[f"{split}_{metric}"] for r in rows],
                                    color=COLORS[k], label=NAMES[k], lw=1.7)
                axes[row, col].set_ylabel(title)
                axes[row, col].set_title(split.capitalize())
        final = rows[-1]
        summaries[k] = {"final_step": int(final["step"]), "elapsed_seconds": final["seconds"],
                        "parameters": json.loads((RUNS / k / "config.json").read_text())["parameters"],
                        "final": {q: v for q, v in final.items() if q.startswith("train_") or q.startswith("test_")},
                        "test_5pct_first": threshold_time(rows, "test_accuracy_rel_5pct", .5),
                        "test_1pct_first": threshold_time(rows, "test_accuracy_rel_1pct", .5),
                        "train_5pct_first": threshold_time(rows, "train_accuracy_rel_5pct", .5)}
    for ax in axes[-1]:
        ax.set_xlabel("Gradient steps")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Learned matrix, graph, and SMILES representations; spectral-group holdout")
    fig.tight_layout()
    fig.savefig(plot_dir / "training_dynamics.png", dpi=180)
    fig.savefig(plot_dir / "training_dynamics.svg")
    plt.close(fig)

    cols = min(3, len(available))
    rows_count = int(np.ceil(len(available) / cols))
    fig, axes_grid = plt.subplots(rows_count, cols, figsize=(4 * cols, 4 * rows_count),
                                  sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes_grid).ravel()
    for ax, k in zip(axes, available):
        with np.load(RUNS / k / "final_probe.npz") as d:
            test = d["test"]
            ax.scatter(d["true"][test], d["pred"][test], s=4, alpha=.12, color=COLORS[k])
        ax.plot([0, 2.1], [0, 2.1], "k--", lw=1)
        ax.set_title(NAMES[k])
        ax.set_xlabel("True gap (|β|)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted gap (|β|)")
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.savefig(plot_dir / "heldout_predictions.png", dpi=180)
    plt.close(fig)

    fig, axes_grid = plt.subplots(rows_count, cols, figsize=(4.5 * cols, 4 * rows_count),
                                  constrained_layout=True)
    axes = np.atleast_1d(axes_grid).ravel()
    for ax, k in zip(axes, available):
        with np.load(RUNS / k / "final_probe.npz") as d:
            hidden = d["hidden"].astype(np.float64)
            train, test = d["train"], d["test"]
            y = d["true"].astype(np.float64)
            xy, ratio, r2 = pca_probe(hidden, train, test, y)
            ax.scatter(xy[train, 0], xy[train, 1], s=2, color="lightgray", alpha=.2)
            color = ax.scatter(xy[test, 0], xy[test, 1], c=y[test], s=4, alpha=.4,
                               cmap="viridis", vmin=.2, vmax=2.0)
            summaries[k]["pca_variance_ratio_first_two"] = ratio
            summaries[k]["hidden_linear_probe_test_r2_loggap"] = r2
            summaries[k]["pc1_gap_pearson_test"] = float(np.corrcoef(xy[test, 0], y[test])[0, 1])
            summaries[k]["pc1_site_count_pearson_test"] = float(np.corrcoef(xy[test, 0], d["site_counts"][test])[0, 1])
            summaries[k]["pc1_pi_electrons_pearson_test"] = float(np.corrcoef(xy[test, 0], d["pi_electrons"][test])[0, 1])
        ax.set_title(f"{NAMES[k]}\nPC1+2 {sum(ratio):.1%}, linear probe R² {r2:.2f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.colorbar(color, ax=axes.tolist(), label="Gap (|β|)", shrink=.7)
    fig.savefig(plot_dir / "hidden_pca.png", dpi=180)
    plt.close(fig)
    (HERE / "summary_20k_seed0.json").write_text(json.dumps(summaries, indent=2))
    for k, v in summaries.items():
        print(k, "test1%", round(v["final"]["test_accuracy_rel_1pct"], 4),
              "test5%", round(v["final"]["test_accuracy_rel_5pct"], 4),
              "testMAE", round(v["final"]["test_mae"], 5),
              "hidden R2", round(v["hidden_linear_probe_test_r2_loggap"], 4))

    # These controls receive eigenvalues that were diagonalized before training.
    # Keep their curves and metrics outside the raw-matrix representation figure.
    control_summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for k in SPECTRAL_CONTROLS:
        if not (RUNS / k / "history.csv").exists():
            continue
        h = read_history(RUNS / k / "history.csv")
        x = [r["step"] for r in h]
        for col, split in enumerate(("train", "test")):
            for row, (metric, label) in enumerate((("accuracy_rel_1pct", "1% relative-gap accuracy"),
                                                   ("mae", "Gap MAE (|β|)"))):
                axes[row, col].plot(x, [r[f"{split}_{metric}"] for r in h],
                                    color=COLORS[k], label=NAMES[k], lw=2)
                axes[row, col].set_title(f"{split.capitalize()} {label}")
        control_summary[k] = {
            "input": ("precomputed full eigenvalue spectrum plus HOMO/LUMO position markers"
                      if k == "matrix_spectral" else
                      "precomputed HOMO and LUMO eigenvalues selected before the neural network"),
            "final": {q: v for q, v in h[-1].items()
                      if q.startswith("train_") or q.startswith("test_")},
        }
    for ax in axes[-1]:
        ax.set_xlabel("Gradient steps")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Precomputed orbital-energy controls — not raw-matrix models")
    fig.tight_layout()
    fig.savefig(plot_dir / "precomputed_eigenvalue_controls.png", dpi=180)
    plt.close(fig)
    (HERE / "spectral_controls_20k_seed0.json").write_text(json.dumps(control_summary, indent=2))


if __name__ == "__main__":
    main()
