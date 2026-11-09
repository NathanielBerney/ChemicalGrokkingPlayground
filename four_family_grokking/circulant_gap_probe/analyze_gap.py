#!/usr/bin/env python3
"""Plot strict train/test gap accuracy and held-out error by ring size."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_gap import (ROOT, VARIANTS, GapBitTransformer, PositionAwareGapTransformer,
                       ScalarGapMLP, FlatBitsMLP,
                       load_data, predict_all, score)


def load_metrics(variant):
    with (ROOT / "runs" / variant / "metrics.csv").open() as handle:
        return list(csv.DictReader(handle))


def series(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def final_prediction(variant, device, n, run_root=None):
    run_root = ROOT / "runs" if run_root is None else Path(run_root)
    checkpoint = torch.load(run_root / variant / "last.pt", map_location=device,
                            weights_only=False)
    model = (ScalarGapMLP() if variant == "log_scalar" else
             FlatBitsMLP() if variant == "log_flatbits" else
             PositionAwareGapTransformer() if variant == "log_posbits" else
             GapBitTransformer()).to(device)
    model.load_state_dict(checkpoint["model"])
    if variant == "log_scalar":
        value = (np.log(n) - checkpoint["input_mean"]) / checkpoint["input_std"]
        x = torch.as_tensor(value[:, None], dtype=torch.float32, device=device)
    else:
        bits = ((n[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.int64)
        x = torch.as_tensor(bits.astype(np.float32) if variant == "log_flatbits" else bits,
                            device=device)
    pred = predict_all(model, x, variant, checkpoint["target_center"],
                       checkpoint["target_scale"])
    return pred, checkpoint["train_indices"], checkpoint["test_indices"]


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    n, gap = load_data()
    available = [v for v in VARIANTS if (ROOT / "runs" / v / "last.pt").exists()]
    # This baseline learns only a slope and intercept from training examples.
    # It uses no analytic Hückel formula when fitting.
    _, reference_train, reference_test = final_prediction("log_scalar", device, n)
    slope, intercept = np.polyfit(np.log(n[reference_train]), np.log(gap[reference_train]), 1)
    log_linear_pred = np.exp(intercept + slope * np.log(n))
    extended_n = np.arange(40_006, 60_003, 4, dtype=np.int64)
    extended_truth = 4 * np.sin(np.pi / extended_n)
    extended_log_linear_pred = np.exp(intercept + slope * np.log(extended_n))
    fig, axes = plt.subplots(2, len(available), figsize=(5 * len(available), 8), layout="constrained")
    by_size, size_axes = plt.subplots(len(available), 2,
                                     figsize=(13, 3.6 * len(available)), layout="constrained")
    report = {}
    for column, variant in enumerate(available):
        rows = load_metrics(variant)
        step = series(rows, "step")
        config = json.loads((ROOT / "runs" / variant / "config.json").read_text())
        for row, pct in enumerate((1, 5)):
            ax = axes[row, column]
            key = f"accuracy_rel_{pct}pct"
            ax.plot(step, series(rows, "train_" + key), label="train", color="#2563eb")
            ax.plot(step, series(rows, "test_" + key), label="held-out", color="#ea580c")
            ax.axhline(config["constant_baseline_test"][key], ls=":", color="#64748b",
                       label="constant baseline")
            ax.axhline(score(log_linear_pred, gap, reference_test)[key], ls="--",
                       color="#059669", label="log-linear baseline")
            ax.set(xlabel="Full-batch step", ylabel="Accuracy", ylim=(0, 1.02),
                   title=f"{variant.replace('_', ' ')} · within {pct}% relative error")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        pred, train, test = final_prediction(variant, device, n)
        extended_pred, _, _ = final_prediction(variant, device, extended_n)
        err = np.abs(pred - gap) / gap
        ax = size_axes[column, 0]
        order = np.argsort(n)
        ax.plot(n[order], gap[order], color="black", lw=1.3, label="exact gap")
        ax.scatter(n[test], np.maximum(pred[test], 1e-9), s=2, alpha=.2,
                   color="#ea580c", label="held-out prediction")
        ax.set(xscale="log", yscale="log", xlabel="Ring size n", ylabel="Gap / |β|",
               title=variant.replace("_", " "))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax = size_axes[column, 1]
        bins = np.geomspace(n.min(), n.max() + 1, 16)
        centers = np.sqrt(bins[:-1] * bins[1:])
        median_error, accuracy_1pct, count = [], [], []
        for left, right in zip(bins[:-1], bins[1:]):
            ids = test[(n[test] >= left) & (n[test] < right)]
            count.append(len(ids))
            median_error.append(float(np.median(err[ids])) if len(ids) else np.nan)
            accuracy_1pct.append(float(np.mean(err[ids] < .01)) if len(ids) else np.nan)
        ax.plot(centers, median_error, marker="o", color="#ea580c",
                label="held-out median error")
        ax.axhline(.01, color="black", ls="--", label="1% threshold")
        ax.set(xscale="log", yscale="log", xlabel="Ring size n", ylabel="Relative gap error")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        report[variant] = {
            "last_step": int(step[-1]),
            "train_final": score(pred, gap, train),
            "test_final": score(pred, gap, test),
            "test_n_le_1000": score(pred, gap, test[n[test] <= 1000]),
            "extrapolation_40006_to_60002": score(
                extended_pred, extended_truth, np.arange(len(extended_n))),
            "test_best_1pct": float(np.max(series(rows, "test_accuracy_rel_1pct"))),
            "test_best_5pct": float(np.max(series(rows, "test_accuracy_rel_5pct"))),
            "constant_baseline_test": config["constant_baseline_test"],
            "size_bin_centers": centers.tolist(),
            "size_bin_test_counts": count,
            "size_bin_median_relative_error": median_error,
            "size_bin_accuracy_1pct": accuracy_1pct,
        }
    fig.suptitle("Circulant HOMO–LUMO gap · strict train/test accuracy", fontsize=16)
    fig.savefig(ROOT / "gap_accuracy.png", dpi=175)
    plt.close(fig)
    by_size.suptitle("Circulant gap · held-out predictions versus ring size", fontsize=16)
    by_size.savefig(ROOT / "gap_by_size.png", dpi=175)
    plt.close(by_size)
    report["log_linear_baseline"] = {
        "slope": float(slope), "intercept": float(intercept),
        "train": score(log_linear_pred, gap, reference_train),
        "test": score(log_linear_pred, gap, reference_test),
        "extrapolation_40006_to_60002": score(extended_log_linear_pred, extended_truth,
                                                np.arange(len(extended_n)))
    }
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({v: {"test_1pct": report[v]["test_final"]["accuracy_rel_1pct"],
                          "best_1pct": report[v]["test_best_1pct"],
                          "test_5pct": report[v]["test_final"]["accuracy_rel_5pct"]}
                      for v in available}, indent=2))


if __name__ == "__main__":
    main()
