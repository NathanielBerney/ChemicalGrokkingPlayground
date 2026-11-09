#!/usr/bin/env python3
"""Check size dependence and orbital-gauge invariance of finished models."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_family import (ROOT, BitSizeTransformer, MatrixTransformer,
                          load_data, predictions)


HERE = Path(__file__).resolve().parent


def model_and_split(family, device, run_root="runs"):
    checkpoint = torch.load(HERE / run_root / family / "last.pt",
                            map_location=device, weights_only=False)
    model = (MatrixTransformer() if family == "pericyclic" else BitSizeTransformer()).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["train_indices"], checkpoint["test_indices"], checkpoint


def energy_size_diagnostics(device, run_root="runs"):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    report = {}
    for row, family in enumerate(("circulant", "mobius", "linear")):
        _, x, mask, _, n, target = load_data(family, device)
        model, train_idx, test_idx, checkpoint = model_and_split(family, device, run_root)
        output = predictions(model, x, mask)
        transformed = output * checkpoint["target_std"] + checkpoint["target_mean"]
        pred = (np.exp(np.clip(transformed, -30, 5)) * np.array([-1, 1])
                if run_root == "runs_log" else transformed)
        exact = np.abs(target[:, 1])
        estimated = np.abs(pred[:, 1])
        order = np.argsort(n)
        ax = axes[row, 0]
        ax.plot(n[order], exact[order], color="black", lw=1.5, label="exact LUMO")
        ax.scatter(n[test_idx], estimated[test_idx], s=2, alpha=.2, label="held-out prediction")
        ax.set(xscale="log", yscale="log", xlabel="Carbon atoms n", ylabel="|LUMO| / |β|",
               title=f"{family.capitalize()} · frontier energy versus size")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        ax = axes[row, 1]
        bins = np.geomspace(n.min(), n.max() + 1, 15)
        centers = np.sqrt(bins[:-1] * bins[1:])
        relative_error = np.max(np.abs(pred - target) / np.maximum(np.abs(target), 1e-8), axis=1)
        median = []
        within_5 = []
        counts = []
        for low, high in zip(bins[:-1], bins[1:]):
            ids = test_idx[(n[test_idx] >= low) & (n[test_idx] < high)]
            median.append(float(np.median(relative_error[ids])) if len(ids) else np.nan)
            within_5.append(float(np.mean(relative_error[ids] < .05)) if len(ids) else np.nan)
            counts.append(len(ids))
        ax.plot(centers, median, marker="o", label="median relative error")
        ax.axhline(.05, color="black", ls="--", label="5% target")
        ax.set(xscale="log", yscale="log", xlabel="Carbon atoms n", ylabel="Relative error",
               title="Held-out error by molecular size")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        report[family] = {"size_bin_centers": centers.tolist(),
                          "median_relative_error": median,
                          "accuracy_within_5pct": within_5,
                          "test_count_per_bin": counts}
    fig.suptitle("Frontier-energy models · size-dependent generalization" +
                 (" · log-magnitude target" if run_root == "runs_log" else ""))
    suffix = "_log" if run_root == "runs_log" else ""
    fig.savefig(HERE / "plots" / f"energy_size_diagnostics{suffix}.png", dpi=175)
    plt.close(fig)
    return report


def pericyclic_invariance(device, repeats=4):
    _, x, mask, _, n, y = load_data("pericyclic", device)
    model, train_idx, test_idx, _ = model_and_split("pericyclic", device)
    H = x[test_idx].cpu().numpy()
    mask_test = mask[test_idx]
    n_test = n[test_idx]
    y_test = y[test_idx]
    raw = predictions(model, x[test_idx], mask_test)
    raw_label = (raw >= 0).astype(np.uint8)
    rng = np.random.default_rng(5001)
    gauge_acc = []
    rotation_acc = []
    gauge_agreement = []
    rotation_agreement = []
    for rep in range(repeats):
        signs = rng.choice(np.array([-1, 1], dtype=np.float32), size=(len(H), 8))
        gauged = H * signs[:, :, None] * signs[:, None, :]
        logits = predictions(model, torch.from_numpy(gauged).to(device), mask_test)
        labels = (logits >= 0).astype(np.uint8)
        gauge_acc.append(float(np.mean(labels == y_test)))
        gauge_agreement.append(float(np.mean(labels == raw_label)))
        rotated = np.zeros_like(H)
        for i, size in enumerate(n_test):
            order = np.roll(np.arange(size), int(rng.integers(size)))
            if rng.integers(2):
                order = order[::-1]
            rotated[i, :size, :size] = H[i][np.ix_(order, order)]
        logits = predictions(model, torch.from_numpy(rotated).to(device), mask_test)
        labels = (logits >= 0).astype(np.uint8)
        rotation_acc.append(float(np.mean(labels == y_test)))
        rotation_agreement.append(float(np.mean(labels == raw_label)))
    with (ROOT / "pericyclic_thermal_dataset" / "metadata.csv").open() as handle:
        metadata = list(csv.DictReader(handle))
    family_scores = {}
    for family in sorted({metadata[i]["reaction_family"] for i in test_idx}):
        chosen = np.array([j for j, i in enumerate(test_idx) if metadata[i]["reaction_family"] == family])
        family_scores[family] = {"count": len(chosen),
                                 "accuracy": float(np.mean(raw_label[chosen] == y_test[chosen]))}
    size_scores = {}
    for size in sorted(np.unique(n_test)):
        chosen = np.flatnonzero(n_test == size)
        size_scores[str(int(size))] = {"count": len(chosen),
                                       "accuracy": float(np.mean(raw_label[chosen] == y_test[chosen]))}
    phase_scores = {}
    for phase in (-1, 1):
        chosen = np.array([j for j, i in enumerate(test_idx)
                           if int(metadata[i]["gauge_invariant_cycle_phase"]) == phase])
        phase_scores[str(phase)] = {"count": len(chosen),
                                    "accuracy": float(np.mean(raw_label[chosen] == y_test[chosen]))}
    report = {"original_test_accuracy": float(np.mean(raw_label == y_test)),
              "gauge_accuracy_mean": float(np.mean(gauge_acc)),
              "gauge_accuracy_each": gauge_acc,
              "gauge_prediction_agreement_mean": float(np.mean(gauge_agreement)),
              "cyclic_rotation_reversal_accuracy_mean": float(np.mean(rotation_acc)),
              "cyclic_rotation_reversal_accuracy_each": rotation_acc,
              "cyclic_rotation_prediction_agreement_mean": float(np.mean(rotation_agreement)),
              "accuracy_by_reaction_family": family_scores,
              "accuracy_by_active_sites": size_scores,
              "accuracy_by_cycle_phase": phase_scores}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    families = list(family_scores)
    values = [family_scores[family]["accuracy"] for family in families]
    axes[0].barh(np.arange(len(families)), values, color="#2563eb")
    axes[0].set_yticks(np.arange(len(families)),
                      [name.replace("_", " ") for name in families])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Held-out accuracy", xlim=(0, 1.05), title="By reaction family")
    axes[0].grid(axis="x", alpha=.2)
    variants = ["Original", "Gauge transformed", "Rotated / reversed"]
    scores = [report["original_test_accuracy"], report["gauge_accuracy_mean"],
              report["cyclic_rotation_reversal_accuracy_mean"]]
    axes[1].bar(variants, scores, color=["#2563eb", "#c026d3", "#059669"])
    axes[1].set(ylabel="Held-out accuracy", ylim=(0, 1.05),
                title="Equivalent orbital descriptions")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Pericyclic model · rule generalization checks")
    fig.savefig(HERE / "plots" / "pericyclic_generalization.png", dpi=175)
    plt.close(fig)
    return report


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    (HERE / "plots").mkdir(exist_ok=True)
    report = {"energy_size": energy_size_diagnostics(device),
              "pericyclic_invariance": pericyclic_invariance(device)}
    if all((HERE / "runs_log" / family / "last.pt").exists()
           for family in ("circulant", "mobius", "linear")):
        report["energy_size_log"] = energy_size_diagnostics(device, "runs_log")
    (HERE / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pericyclic_invariance"], indent=2))


if __name__ == "__main__":
    main()
