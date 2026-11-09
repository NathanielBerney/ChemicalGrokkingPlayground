#!/usr/bin/env python3
"""Checkpoint ablations for the N=400 Omnigrok QM9 animation.

Mean-replace pooled graph and final hidden channels, compare matched random
groups, and fit train-only linear probes. Original checkpoints are read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from reproduce_qm9 import DATA_ROOT, ROOT, ExampleNet


OUT = ROOT / "animations" / "ablations" / "total400"
STATES = ((3, 10000), (3, 20000), (3, 40000), (3, 87500), (1, 40000))
RANDOM_TRIALS = 20


@torch.inference_mode()
def encode(model, loader, device):
    pooled_rows = []
    for batch in loader:
        batch = batch.to(device)
        nodes = F.relu(model.conv1(batch.x, batch.edge_index, batch.edge_attr))
        nodes = F.relu(model.conv2(nodes, batch.edge_index, batch.edge_attr))
        pooled_rows.append(global_add_pool(nodes, batch.batch).cpu().numpy())
    return np.concatenate(pooled_rows).astype(np.float64)


def head_parameters(model):
    return tuple(parameter.detach().cpu().numpy().astype(np.float64)
                 for parameter in (model.fc_1.weight, model.fc_1.bias,
                                   model.out.weight, model.out.bias))


def head(pooled, parameters):
    first_weight, first_bias, out_weight, out_bias = parameters
    hidden = np.maximum(pooled @ first_weight.T + first_bias, 0)
    prediction = hidden @ out_weight.T + out_bias
    return hidden, prediction.ravel()


def mse_scores(prediction, truth, train, test):
    error = (prediction - truth) ** 2
    return {"train_mse": float(error[train].mean()),
            "test_mse": float(error[test].mean()),
            "test_mae": float(np.abs(prediction[test] - truth[test]).mean())}


def ridge_predict(features, target, fit, penalty):
    center = features[fit].mean(axis=0)
    scale = np.maximum(features[fit].std(axis=0), 1e-6)
    standardized = (features - center) / scale
    target_center = target[fit].mean()
    matrix = standardized[fit].T @ standardized[fit]
    weights = np.linalg.solve(
        matrix + penalty * np.eye(features.shape[1]),
        standardized[fit].T @ (target[fit] - target_center),
    )
    return standardized @ weights + target_center


def train_only_probe(features, truth, train, test):
    folds = np.array_split(np.random.default_rng(2026).permutation(train), 5)
    candidates = []
    for penalty in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0):
        errors = []
        for validation in folds:
            fit = np.setdiff1d(train, validation, assume_unique=True)
            prediction = ridge_predict(features, truth, fit, penalty)
            errors.append(np.mean((prediction[validation] - truth[validation]) ** 2))
        candidates.append((float(np.mean(errors)), penalty))
    validation_mse, penalty = min(candidates)
    prediction = ridge_predict(features, truth, train, penalty)
    return {"selected_penalty": penalty, "train_cv_mse": validation_mse,
            **mse_scores(prediction, truth, train, test)}


def channel_ablations(layer, features, prediction, truth, train, test,
                      parameters, pooled):
    pooled_mean = pooled[train].mean(axis=0)
    hidden, _ = head(pooled, parameters)
    hidden_mean = hidden[train].mean(axis=0)
    out_weight = parameters[2].ravel()
    out_bias = float(parameters[3][0])

    def intervene(channels):
        if layer == "pooled16":
            modified = pooled.copy()
            modified[:, channels] = pooled_mean[channels]
            return head(modified, parameters)[1]
        modified = hidden.copy()
        modified[:, channels] = hidden_mean[channels]
        return modified @ out_weight + out_bias

    baseline = mse_scores(prediction, truth, train, test)
    single_effect = np.array([
        mse_scores(intervene([channel]), truth, train, test)["train_mse"]
        - baseline["train_mse"]
        for channel in range(features.shape[1])
    ])
    ranked = np.argsort(single_effect)[::-1]
    rng = np.random.default_rng(2026)
    rows = []
    retained_rows = []
    for count in (1, features.shape[1] // 4, features.shape[1] // 2):
        selections = [("top_train_impact", 0, ranked[:count]),
                      ("bottom_train_impact", 0, ranked[-count:])]
        selections.extend(("random", trial,
                           rng.choice(features.shape[1], count, replace=False))
                          for trial in range(RANDOM_TRIALS))
        for selection, trial, channels in selections:
            rows.append({
                "layer": layer,
                "channels_removed": int(count),
                "selection": selection,
                "random_trial": trial,
                **mse_scores(intervene(channels), truth, train, test),
            })
        retained = [("top_train_impact", 0, ranked[:count]),
                    ("bottom_train_impact", 0, ranked[-count:])]
        retained.extend(("random", trial,
                         rng.choice(features.shape[1], count, replace=False))
                        for trial in range(RANDOM_TRIALS))
        for selection, trial, channels in retained:
            removed = np.setdiff1d(np.arange(features.shape[1]), channels)
            retained_rows.append({
                "layer": layer,
                "channels_retained": int(count),
                "selection": selection,
                "random_trial": trial,
                **mse_scores(intervene(removed), truth, train, test),
            })
    return rows, retained_rows, single_effect.tolist()


def run_state(alpha, step, loader, device, cached, truth, train, test):
    checkpoint_path = (ROOT / "runs" / f"grokking_total400_alpha{alpha}_animation"
                       / "snapshots" / f"step_{step:06d}.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["step"]) != step:
        raise ValueError(f"Checkpoint step mismatch: {checkpoint_path}")
    model = ExampleNet().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    pooled = encode(model, loader, device)
    parameters = head_parameters(model)
    hidden, prediction = head(pooled, parameters)
    index = int(np.flatnonzero(cached["steps"] == step)[0])
    expected = cached[f"alpha{alpha}_prediction"][index]
    max_difference = float(np.max(np.abs(prediction - expected)))
    if max_difference > 1e-3:
        raise ValueError(f"Ablation baseline differs from GIF predictions: {max_difference}")
    baseline = mse_scores(prediction, truth, train, test)
    rows = []
    retained_rows = []
    effects = {}
    for layer, features in (("pooled16", pooled), ("readout32", hidden)):
        layer_rows, layer_retained, effects[layer] = channel_ablations(
            layer, features, prediction, truth, train, test, parameters, pooled)
        rows.extend(layer_rows)
        retained_rows.extend(layer_retained)
    probes = {
        "pooled16_linear": train_only_probe(pooled, truth, train, test),
        "readout32_linear": train_only_probe(hidden, truth, train, test),
        "output_affine": train_only_probe(prediction[:, None], truth, train, test),
    }
    return {
        "alpha": alpha,
        "step": step,
        "checkpoint": str(checkpoint_path),
        "max_difference_from_gif_prediction": max_difference,
        "baseline": baseline,
        "activity": {
            "pooled_channels_active_on_any_graph": int(
                np.sum(np.any(pooled > 1e-8, axis=0))),
            "readout_channels_active_on_any_graph": int(
                np.sum(np.any(hidden > 1e-8, axis=0))),
            "pooled_channel_active_fraction": np.mean(pooled > 1e-8, axis=0).tolist(),
            "readout_channel_active_fraction": np.mean(hidden > 1e-8, axis=0).tolist(),
        },
        "single_channel_train_mse_effect": effects,
        "channel_ablations": rows,
        "retained_channel_tests": retained_rows,
        "train_only_probes": probes,
    }


def draw_dashboard(results, output):
    chosen = [row for row in results if row["step"] == 40000]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for column, state in enumerate(sorted(chosen, key=lambda row: row["alpha"])):
        for row_index, layer in enumerate(("pooled16", "readout32")):
            ax = axes[row_index, column]
            baseline = state["baseline"]["test_mse"]
            records = [row for row in state["channel_ablations"]
                       if row["layer"] == layer]
            sizes = sorted({row["channels_removed"] for row in records})
            for selection, color in (("top_train_impact", "#dc2626"),
                                     ("bottom_train_impact", "#2563eb"),
                                     ("random", "#64748b")):
                means, deviations = [], []
                for size in sizes:
                    values = [row["test_mse"] / baseline for row in records
                              if row["channels_removed"] == size
                              and row["selection"] == selection]
                    means.append(np.mean(values))
                    deviations.append(np.std(values))
                ax.errorbar(sizes, means,
                            yerr=deviations if selection == "random" else None,
                            marker="o", color=color, capsize=3,
                            label=selection.replace("_", " "))
            ax.axhline(1, color="#111827", ls="--", lw=1)
            ax.set(xlabel="Channels mean-replaced",
                   ylabel="Test MSE / unablated test MSE",
                   title=f"α={state['alpha']} · {layer} · step 40,000",
                   xticks=sizes)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("QM9 channel ablations ranked using training MSE only")
    fig.savefig(output / "channel_ablation_step40000.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for column, state in enumerate(sorted(chosen, key=lambda row: row["alpha"])):
        for row_index, layer in enumerate(("pooled16", "readout32")):
            ax = axes[row_index, column]
            baseline = state["baseline"]["test_mse"]
            records = [row for row in state["retained_channel_tests"]
                       if row["layer"] == layer]
            sizes = sorted({row["channels_retained"] for row in records})
            for selection, color in (("top_train_impact", "#dc2626"),
                                     ("bottom_train_impact", "#2563eb"),
                                     ("random", "#64748b")):
                means, deviations = [], []
                for size in sizes:
                    values = [row["test_mse"] / baseline for row in records
                              if row["channels_retained"] == size
                              and row["selection"] == selection]
                    means.append(np.mean(values))
                    deviations.append(np.std(values))
                ax.errorbar(sizes, means,
                            yerr=deviations if selection == "random" else None,
                            marker="o", color=color, capsize=3,
                            label=selection.replace("_", " "))
            ax.axhline(1, color="#111827", ls="--", lw=1)
            ax.set(xlabel="Channels retained",
                   ylabel="Test MSE / unablated test MSE",
                   title=f"α={state['alpha']} · {layer} · step 40,000",
                   xticks=sizes)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Sufficiency of channels ranked using training MSE only")
    fig.savefig(output / "channel_sufficiency_step40000.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    records = [row for row in results if row["alpha"] == 3]
    steps = [row["step"] for row in records]
    ax.plot(steps, [row["baseline"]["test_mse"] for row in records],
            marker="o", label="trained head", color="#334155")
    for name, color in (("pooled16_linear", "#2563eb"),
                        ("readout32_linear", "#059669"),
                        ("output_affine", "#ea580c")):
        ax.plot(steps, [row["train_only_probes"][name]["test_mse"]
                        for row in records], marker="o", color=color,
                label=name.replace("_", " "))
    ax.set(xlabel="Optimizer step", ylabel="Full held-out MSE",
           title="α=3 frozen-representation readout probes")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(output / "readout_probes.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    states = sorted(chosen, key=lambda row: row["alpha"])
    locations = np.arange(3)
    width = .36
    for offset, state, color in ((-width / 2, states[0], "#2563eb"),
                                 (width / 2, states[1], "#7c3aed")):
        baseline = state["baseline"]["test_mse"]
        values = [baseline]
        for count in (4, 8):
            values.append(next(row["test_mse"] for row in state["retained_channel_tests"]
                               if row["layer"] == "pooled16"
                               and row["selection"] == "top_train_impact"
                               and row["channels_retained"] == count))
        bars = ax.bar(locations + offset, values, width, color=color,
                      label=f"α={state['alpha']}")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(xticks=locations, xticklabels=("Original", "Retain top 4", "Retain top 8"),
           ylim=(0, 1.38), ylabel="Held-out MSE",
           title="Step 40,000: pooled graph channels ranked on training set")
    ax.grid(alpha=.2, axis="y")
    ax.set_axisbelow(True)
    ax.legend()
    fig.savefig(output / "pooled_channel_compactness_step40000.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    dataset = QM9(root=str(DATA_ROOT))[:400]
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with np.load(ROOT / "animations/analysis_total400.npz") as saved:
        cached = {key: saved[key] for key in saved.files}
    truth = cached["truth"]
    train = np.flatnonzero(cached["train_mask"])
    test = np.flatnonzero(~cached["train_mask"])
    if len(train) != 200 or len(test) != 200:
        raise ValueError("Expected a 200/200 split")
    for alpha in (1, 3):
        config = json.loads((ROOT / "runs" / f"grokking_total400_alpha{alpha}_animation"
                             / "config.json").read_text())
        if set(config["train_indices"]) != set(train):
            raise ValueError(f"Archived train split differs for alpha={alpha}")
    results = [run_state(alpha, step, loader, device, cached, truth, train, test)
               for alpha, step in STATES]
    report = {
        "dataset": "first 400 QM9 molecules, isotropic polarizability",
        "split": "200 training / 200 test, archived seed 0",
        "mean_target_test_mse": float(np.mean((truth[test] - truth[train].mean()) ** 2)),
        "states": results,
        "methods": [
            "The model has no token embedding table or classifier logits.",
            "Pooled graph channels and final hidden channels are replaced by their training-set means before the frozen downstream head.",
            "Individual channels are ranked by train-MSE increase; groups compare top, bottom, and 20 equal-size random selections.",
            "Sufficiency tests retain only the top-ranked, bottom-ranked, or random channels and replace all others by training means.",
            "Ridge probes use only training targets, with five-fold train-only cross-validation for the penalty.",
            "The exact cached GIF predictions and archived 200/200 split are checked before scoring.",
        ],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    draw_dashboard(results, args.outdir)
    for row in results:
        print(f"alpha={row['alpha']} step={row['step']}: "
              f"train MSE {row['baseline']['train_mse']:.4g}, "
              f"test MSE {row['baseline']['test_mse']:.4g}")
    print(f"Saved {args.outdir}")


if __name__ == "__main__":
    main()
