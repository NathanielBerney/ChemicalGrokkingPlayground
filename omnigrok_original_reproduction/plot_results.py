#!/usr/bin/env python3
"""Plot Omnigrok QM9 grokking curves and fixed-norm landscape results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
BLUE = "#2563eb"
ORANGE = "#ea580c"


def read_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def grokking_pair(total_size, metric):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, layout="constrained")
    summary = {}
    for ax, alpha in zip(axes, (3, 1)):
        run = ROOT / "runs" / f"grokking_total{total_size}_alpha{alpha}"
        if not (run / "last.pt").exists():
            plt.close(fig)
            return None
        if metric == "author":
            values = read_csv(run / "batch20_metrics.csv")
            train_key, test_key = "author_scaled_train_mse", "author_scaled_test_mse"
            ylabel = "Notebook plotted quantity"
        else:
            values = read_csv(run / "full_metrics.csv")
            train_key, test_key = "train_mse", "test_mse"
            ylabel = "Full-set MSE (polarizability²)"
        ax.plot(values["step"], np.maximum(values[train_key], 1e-12),
                color=BLUE, label="train", lw=1.5)
        ax.plot(values["step"], np.maximum(values[test_key], 1e-12),
                color=ORANGE, label="test", lw=1.5)
        ax.set(xscale="log", yscale="log", ylabel=ylabel,
               title=f"Initialization scale α={alpha}; {total_size//2} train / {total_size//2} test")
        ax.grid(alpha=.25, which="both")
        ax.legend()
        summary[str(alpha)] = {"last_step": int(values["step"][-1]),
                               "final_train": float(values[train_key][-1]),
                               "final_test": float(values[test_key][-1]),
                               "best_test": float(values[test_key].min())}
    axes[-1].set_xlabel("Optimizer step")
    fig.suptitle("Omnigrok QM9 · isotropic polarizability", fontsize=15)
    name = f"figure05d_{'notebook_metric' if metric == 'author' else 'true_mse'}_total{total_size}.png"
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)
    return summary


def log_edges(values):
    log = np.log10(np.asarray(values, dtype=float))
    mids = (log[1:] + log[:-1]) / 2
    return 10 ** np.r_[log[0] - (mids[0] - log[0]), mids,
                       log[-1] + (log[-1] - mids[-1])]


def landscape_results():
    rows = []
    for path in (ROOT / "runs" / "landscape").glob("size_*.json"):
        rows.append(json.loads(path.read_text()))
    return rows


def write_landscape_csv(rows):
    if not rows:
        return
    fields = ["job_index", "total_size", "train_size", "test_size", "alpha",
              "epochs", "steps", "initial_weight_norm", "final_weight_norm",
              "train_mse", "test_mse", "author_scaled_train_mse",
              "author_scaled_test_mse"]
    with (OUT / "landscape_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["job_index"]):
            writer.writerow({key: row.get(key, row["best"].get(key)) for key in fields})


def landscape_heatmap(rows, metric):
    sizes = sorted({row["total_size"] for row in rows if row["job_index"] < 462})
    alphas = sorted({row["alpha"] for row in rows if row["job_index"] < 462})
    if len(sizes) < 2 or len(alphas) < 2:
        return
    size_idx = {size: i for i, size in enumerate(sizes)}
    alpha_idx = {round(alpha, 6): i for i, alpha in enumerate(alphas)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for ax, (key, title) in zip(axes, (("train", "Train"), ("test", "Test"))):
        grid = np.full((len(alphas), len(sizes)), np.nan)
        field = f"author_scaled_{key}_mse" if metric == "author" else f"{key}_mse"
        for row in rows:
            if row["total_size"] in size_idx and round(row["alpha"], 6) in alpha_idx:
                grid[alpha_idx[round(row["alpha"], 6)], size_idx[row["total_size"]]] = row["best"][field]
        image = ax.pcolormesh(log_edges(alphas), log_edges(sizes),
                              np.log10(np.maximum(grid.T, 1e-12)),
                              cmap="viridis", shading="flat")
        ax.set(xscale="log", yscale="log", xlabel="Relative initial weight norm α",
               ylabel="First N QM9 molecules (50% train)", title=f"Best {title.lower()} loss")
        fig.colorbar(image, ax=ax, label="log₁₀ MSE" if metric == "true" else
                     "log₁₀ notebook plotted quantity")
    done = sum(row["job_index"] < 462 for row in rows)
    fig.suptitle(f"Omnigrok QM9 · fixed-norm landscape ({done}/462 grid points)", fontsize=15)
    fig.savefig(OUT / f"figure05ab_landscape_{metric}.png", dpi=180)
    plt.close(fig)


def landscape_cuts(rows, metric):
    available = sorted({row["total_size"] for row in rows})
    sizes = [size for size in (100, 3000) if size in available]
    if len(sizes) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, size in zip(axes, sizes):
        subset = sorted((row for row in rows if row["total_size"] == size),
                        key=lambda row: row["alpha"])
        for key, color, label in (("train", BLUE, "train"), ("test", ORANGE, "test")):
            field = f"author_scaled_{key}_mse" if metric == "author" else f"{key}_mse"
            ax.plot([row["alpha"] for row in subset],
                    [max(row["best"][field], 1e-12) for row in subset],
                    color=color, marker="o", ms=3, label=label)
        ax.set(xscale="log", yscale="log", xlabel="Relative weight norm α",
               ylabel="Best MSE" if metric == "true" else "Notebook plotted quantity",
               title=f"First {size:,} molecules; {size//2:,} train")
        ax.grid(alpha=.25, which="both")
        ax.legend()
    fig.suptitle("Omnigrok QM9 · fixed-norm loss slices", fontsize=15)
    fig.savefig(OUT / f"figure05c_landscape_cuts_{metric}.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    summary = {"grokking": {}, "landscape_grid_points": 0}
    for size in (400, 100):
        summary["grokking"][str(size)] = {}
        for metric in ("author", "true"):
            result = grokking_pair(size, metric)
            if result:
                summary["grokking"][str(size)][metric] = result
    rows = landscape_results()
    write_landscape_csv(rows)
    summary["landscape_grid_points"] = sum(row["job_index"] < 462 for row in rows)
    summary["landscape_cut_points"] = sum(row["job_index"] >= 462 for row in rows)
    for metric in ("author", "true"):
        landscape_heatmap(rows, metric)
        landscape_cuts(rows, metric)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
