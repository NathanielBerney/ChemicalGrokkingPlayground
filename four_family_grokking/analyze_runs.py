#!/usr/bin/env python3
"""Summarize four full-batch grokking runs and render train/test comparisons."""

import csv
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FAMILIES = ("circulant", "mobius", "linear", "pericyclic")


def read_family(family, run_root):
    folder = run_root / family
    if not (folder / "metrics.csv").exists():
        return None
    with (folder / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    config = json.loads((folder / "config.json").read_text())
    def series(name):
        return np.array([float(r[name]) if r[name] else np.nan for r in rows])
    return folder, config, rows, series


def analyze(run_root=ROOT / "runs", plots=ROOT / "plots", families=FAMILIES):
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(families), 2, figsize=(13, 3.5 * len(families)),
                             layout="constrained")
    fig_loss, loss_axes = plt.subplots(len(families), 1, figsize=(9, 3 * len(families)),
                                      layout="constrained")
    report = {}
    for row_i, family in enumerate(families):
        data = read_family(family, run_root)
        if data is None:
            continue
        folder, config, rows, series = data
        step = series("step")
        train_acc, test_acc = series("train_accuracy"), series("test_accuracy")
        train_loss, test_loss = series("train_loss"), series("test_loss")
        first_train = np.flatnonzero(train_acc >= .99)
        first_test = np.flatnonzero(test_acc >= .99)
        is_complete = (folder / "last.pt").exists()
        item = {"complete": is_complete, "last_step": int(step[-1]),
                "first_train_accuracy_99pct_step": int(step[first_train[0]]) if len(first_train) else None,
                "first_test_accuracy_99pct_step": int(step[first_test[0]]) if len(first_test) else None,
                "last_train_accuracy": float(train_acc[-1]),
                "last_test_accuracy": float(test_acc[-1]),
                "best_test_accuracy": float(np.max(test_acc)),
                "last_train_loss": float(train_loss[-1]),
                "last_test_loss": float(test_loss[-1]),
                "last_elapsed_seconds": float(series("elapsed_seconds")[-1]),
                "baselines": config["baselines"]}
        ax = axes[row_i, 0]
        ax.plot(step, train_acc, label="train", color="#2563eb")
        ax.plot(step, test_acc, label="test", color="#ea580c")
        baseline = (config["baselines"].get("zero_test_abs_accuracy")
                    if family != "pericyclic" else config["baselines"]["majority_test_accuracy"])
        ax.axhline(baseline, color="#64748b", ls=":", label="naive baseline")
        if family == "pericyclic" and step[-1] > 40_000:
            ax.axvline(40_000, color="#7c3aed", ls="--", alpha=.65, label="original endpoint")
        ax.set(ylabel="Accuracy", title=f"{family.capitalize()} · " +
               ("both energies within 0.05 |β|" if family != "pericyclic" else "thermal allowed"),
               ylim=(0, 1.03))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.set(xlim=(0, step[-1]), xlabel="Full-batch optimizer step")
        ax2 = axes[row_i, 1]
        if family == "pericyclic":
            ax2.plot(step, train_loss, label="train BCE", color="#2563eb")
            ax2.plot(step, test_loss, label="test BCE", color="#ea580c")
            ax2.set(ylabel="Cross-entropy", title="Classification loss")
            ax2.set_yscale("log")
        else:
            train_rel = series("train_relative_accuracy")
            test_rel = series("test_relative_accuracy")
            ax2.plot(step, train_rel, label="train", color="#2563eb")
            ax2.plot(step, test_rel, label="test", color="#ea580c")
            ax2.axhline(config["baselines"]["zero_test_relative_accuracy"],
                        color="#64748b", ls=":", label="zero baseline")
            ax2.set(ylabel="Accuracy", title="Both energies within 5% relative error", ylim=(0, 1.03))
            item.update({"last_train_relative_accuracy": float(train_rel[-1]),
                         "last_test_relative_accuracy": float(test_rel[-1]),
                         "best_test_relative_accuracy": float(np.max(test_rel)),
                         "first_train_relative_accuracy_99pct_step": int(step[np.flatnonzero(train_rel >= .99)[0]])
                         if np.any(train_rel >= .99) else None,
                         "first_test_relative_accuracy_99pct_step": int(step[np.flatnonzero(test_rel >= .99)[0]])
                         if np.any(test_rel >= .99) else None,
                         "last_train_rmse": float(series("train_rmse")[-1]),
                         "last_test_rmse": float(series("test_rmse")[-1]),
                         "last_test_small_n_rmse": float(series("test_small_n_rmse")[-1]),
                         "last_test_small_n_relative_accuracy": float(series("test_small_n_relative_accuracy")[-1])})
        ax2.grid(alpha=.2)
        ax2.legend(fontsize=8)
        ax2.set(xlim=(0, step[-1]), xlabel="Full-batch optimizer step")
        lax = loss_axes[row_i]
        lax.plot(step, np.maximum(train_loss, 1e-12), label="train", color="#2563eb")
        lax.plot(step, np.maximum(test_loss, 1e-12), label="test", color="#ea580c")
        if family == "pericyclic" and step[-1] > 40_000:
            lax.axvline(40_000, color="#7c3aed", ls="--", alpha=.65)
        lax.set(ylabel="Loss", title=family.capitalize())
        lax.set_yscale("log")
        lax.grid(alpha=.2)
        lax.legend(fontsize=8)
        lax.set(xlim=(0, step[-1]), xlabel="Full-batch optimizer step")
        if family == "pericyclic":
            one, each = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
            each[0].plot(step, train_acc, label="train", color="#2563eb")
            each[0].plot(step, test_acc, label="test", color="#ea580c")
            each[0].axhline(baseline, color="#64748b", ls=":", label="majority baseline")
            if step[-1] > 40_000:
                each[0].axvline(40_000, color="#7c3aed", ls="--", alpha=.65,
                                label="original endpoint")
            each[0].set(xlabel="Step", ylabel="Accuracy", ylim=(0, 1.03), title="Allowed / forbidden")
            each[1].plot(step, train_loss, label="train", color="#2563eb")
            each[1].plot(step, test_loss, label="test", color="#ea580c")
            if step[-1] > 40_000:
                each[1].axvline(40_000, color="#7c3aed", ls="--", alpha=.65)
            each[1].set(xlabel="Step", ylabel="Cross-entropy", yscale="log", title="Loss")
        else:
            one, each = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
            each[0].plot(step, train_acc, label="train", color="#2563eb")
            each[0].plot(step, test_acc, label="test", color="#ea580c")
            each[0].axhline(baseline, color="#64748b", ls=":", label="zero baseline")
            each[0].set(xlabel="Step", ylabel="Accuracy", ylim=(0, 1.03),
                        title="Both energies within 0.05 |β|")
            each[1].plot(step, train_rel, label="train", color="#2563eb")
            each[1].plot(step, test_rel, label="test", color="#ea580c")
            each[1].set(xlabel="Step", ylabel="Accuracy", ylim=(0, 1.03),
                        title="Both energies within 5% relative")
            each[2].plot(step, series("train_rmse"), label="train", color="#2563eb")
            each[2].plot(step, series("test_rmse"), label="test", color="#ea580c")
            each[2].set(xlabel="Step", ylabel="RMSE / |β|", yscale="log", title="Energy error")
        for individual in each:
            individual.grid(alpha=.2)
            individual.legend(fontsize=8)
        one.suptitle(f"{family.capitalize()} · train versus held-out")
        one.savefig(plots / f"{family}_learning_curves.png", dpi=175)
        plt.close(one)
        report[family] = item
    fig.suptitle("Grokking probes · train versus held-out", fontsize=16)
    fig.savefig(plots / "accuracy_dashboard.png", dpi=175)
    plt.close(fig)
    fig_loss.suptitle("Grokking probes · loss", fontsize=16)
    fig_loss.savefig(plots / "loss_dashboard.png", dpi=175)
    plt.close(fig_loss)
    summary_name = "summary_log.json" if run_root.name == "runs_log" else "summary.json"
    (ROOT / summary_name).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--plots", type=Path, default=ROOT / "plots")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_root, args.plots, args.families), indent=2))
