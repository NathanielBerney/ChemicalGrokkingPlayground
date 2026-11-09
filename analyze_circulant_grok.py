#!/usr/bin/env python3
"""Summarize one n=42 HOMO/LUMO grokking run and plot its training dynamics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze(run_dir: Path, dataset: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    with (run_dir / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No completed evaluations")
    step = np.array([int(r["step"]) for r in rows])
    train = np.array([[float(r[f"train_{target}_rmse"]) for target in ("homo", "lumo")]
                      for r in rows])
    test = np.array([[float(r[f"test_{target}_rmse"]) for target in ("homo", "lumo")]
                     for r in rows])
    gini = np.array([float(r["row_weight_fourier_gini"]) for r in rows])
    with np.load(dataset, allow_pickle=False) as data:
        row = data["first_rows"].astype(np.float64)
        eig = np.sort(np.fft.fft(row, axis=1).real, axis=1)
        true = data["eigenvalues"]
        fft_rmse = np.sqrt(np.mean((eig[:, 20:22] - true[:, 20:22]) ** 2, axis=0))
        order = np.random.default_rng(int(config["seed"])).permutation(len(true))
        cut = int(round(float(config["train_fraction"]) * len(true)))
        train_mean_target = true[order[:cut], 20:22].mean(axis=0)
        mean_baseline_rmse = np.sqrt(np.mean(
            (true[order[cut:], 20:22] - train_mean_target) ** 2, axis=0))
    train_mean = train.mean(axis=1)
    test_mean = test.mean(axis=1)
    learned_train = np.flatnonzero(train_mean < 0.05)
    overfit = np.flatnonzero((train_mean < 0.05) & (test_mean > 0.4))
    grok = np.flatnonzero((train_mean < 0.05) & (test_mean < 0.1))
    summary = {
        "evaluations": len(rows),
        "last_step": int(step[-1]),
        "first_train_rmse_below_0_05": int(step[learned_train[0]]) if len(learned_train) else None,
        "first_overfit_checkpoint": int(step[overfit[0]]) if len(overfit) else None,
        "first_test_rmse_below_0_1_after_train_fit": int(step[grok[0]]) if len(grok) else None,
        "best_test_rmse_mean": float(test_mean.min()),
        "best_test_step": int(step[test_mean.argmin()]),
        "last_train_rmse": train[-1].tolist(),
        "last_test_rmse": test[-1].tolist(),
        "first_fourier_weight_gini": float(gini[0]),
        "last_fourier_weight_gini": float(gini[-1]),
        "exact_fft_baseline_rmse": fft_rmse.tolist(),
        "train_mean_test_baseline_rmse": mean_baseline_rmse.tolist(),
    }
    (run_dir / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, layout="constrained",
                              gridspec_kw={"height_ratios": [2, 1]})
    for j, (target, color) in enumerate((("HOMO", "#2563eb"), ("LUMO", "#dc6a32"))):
        axes[0].plot(step, train[:, j], color=color, ls="--", label=f"train {target}")
        axes[0].plot(step, test[:, j], color=color, label=f"test {target}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Energy RMSE (|β| units)")
    axes[0].set_title("n=42 circulant: train/test HOMO–LUMO learning curves")
    axes[0].axhline(float(mean_baseline_rmse.mean()), color="#64748b", ls=":",
                    label="test train-mean baseline")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].plot(step, gini, color="#8b2bb6")
    axes[1].set_ylabel("Row-weight Fourier Gini")
    axes[1].set_xlabel("Optimizer step (full-batch epoch)")
    axes[1].grid(alpha=0.25)
    fig.savefig(run_dir / "learning_curves.png", dpi=170)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("circulant_grokking_runs/seed0_tmux"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/circulant_n42.npz"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir, args.dataset), indent=2))


if __name__ == "__main__":
    main()
