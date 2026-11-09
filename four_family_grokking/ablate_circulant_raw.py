#!/usr/bin/env python3
"""Ablate input embeddings, position vectors, readout weights, and output DCT modes.

This analyzes the archived raw-target circulant checkpoint used by
animations/gifs/circulant_raw_frequency.gif. No checkpoint is modified.
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

from animation_utils import dct_ortho, gini
from animate_energy import output_and_hidden
from train_family import BitSizeTransformer, load_data, predictions


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "animations/runs_raw/circulant/snapshots/step_040000.pt"
DEFAULT_OUTPUT = HERE / "animations/ablations/circulant_raw"
FRACTIONS = (0.10, 0.25)
RANDOM_TRIALS = 5
MODE_COUNT = 32
LOWPASS_CUTOFFS = (0, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 9999)


def score(prediction, target, n, train, test, mean, std, order):
    raw = prediction * std + mean
    relative = np.max(np.abs(raw - target) / np.maximum(np.abs(target), 1e-8), axis=1)
    normalized_target = (target - mean) / std
    squared = (prediction - normalized_target) ** 2
    coefficients = dct_ortho(prediction[order])
    power = np.sum(coefficients[1:] ** 2, axis=1)
    small = test[n[test] <= 1000]
    return {
        "train_normalized_mse": float(np.mean(squared[train])),
        "test_normalized_mse": float(np.mean(squared[test])),
        "train_relative_accuracy_5pct": float(np.mean(relative[train] < 0.05)),
        "test_relative_accuracy_5pct": float(np.mean(relative[test] < 0.05)),
        "test_relative_accuracy_1pct": float(np.mean(relative[test] < 0.01)),
        "small_n_test_relative_accuracy_5pct": float(np.mean(relative[small] < 0.05)),
        "test_raw_rmse": float(np.sqrt(np.mean((raw[test] - target[test]) ** 2))),
        "output_dct_gini": gini(power),
    }


def contribution(coefficients, modes):
    """Inverse orthonormal DCT-II restricted to selected nonconstant modes."""
    count = len(coefficients)
    modes = np.asarray(modes, dtype=np.int64)
    grid = np.arange(count, dtype=np.float64)[:, None]
    basis = np.sqrt(2 / count) * np.cos(np.pi * (grid + 0.5) * modes / count)
    return basis @ coefficients[modes]


def inverse_dct_ortho(coefficients):
    """Inverse of animation_utils.dct_ortho using an even FFT extension."""
    count = len(coefficients)
    mode = np.arange(count)
    scale = np.full(count, np.sqrt(2 * count))
    scale[0] = 2 * np.sqrt(count)
    half_spectrum = (coefficients * scale[:, None]
                     * np.exp(1j * np.pi * mode[:, None] / (2 * count)))
    half_spectrum = np.vstack((half_spectrum,
                               np.zeros((1, coefficients.shape[1]))))
    return np.fft.irfft(half_spectrum, n=2 * count, axis=0)[:count]


def replace_embedding_channels(model, original, channels):
    """Remove the bit-value distinction while preserving the shared midpoint."""
    midpoint = original.mean(dim=0)
    channels = torch.as_tensor(np.ascontiguousarray(channels), device=original.device)
    with torch.no_grad():
        model.embed.weight.copy_(original)
        model.embed.weight[:, channels] = midpoint[channels]


def remove_readout_channels(model, original, channels):
    channels = torch.as_tensor(np.ascontiguousarray(channels), device=original.device)
    with torch.no_grad():
        model.out.weight.copy_(original)
        model.out.weight[:, channels] = 0


def remove_position_channels(model, original, channels):
    channels = torch.as_tensor(np.ascontiguousarray(channels), device=original.device)
    with torch.no_grad():
        model.pos.copy_(original)
        model.pos[0, 1:, channels] = 0


def weight_ablation(model, x, target, n, train, test, mean, std, order, rng):
    embedding = model.embed.weight.detach().clone()
    position = model.pos.detach().clone()
    readout = model.out.weight.detach().clone()
    ranks = {
        "embedding_delta": np.abs((embedding[1] - embedding[0]).cpu().numpy()),
        "position_vectors": np.linalg.norm(position[0, 1:].cpu().numpy(), axis=0),
        "readout": np.linalg.norm(readout.cpu().numpy(), axis=0),
    }
    rows = []
    for layer, importance in ranks.items():
        sorted_channels = np.argsort(importance)[::-1]
        for fraction in FRACTIONS:
            count = max(1, round(len(importance) * fraction))
            selections = [("top", 0, sorted_channels[:count]),
                          ("bottom", 0, sorted_channels[-count:])]
            selections.extend(("random", trial, rng.choice(len(importance), count, replace=False))
                              for trial in range(RANDOM_TRIALS))
            for selection, trial, channels in selections:
                if layer == "embedding_delta":
                    replace_embedding_channels(model, embedding, channels)
                elif layer == "position_vectors":
                    remove_position_channels(model, position, channels)
                else:
                    remove_readout_channels(model, readout, channels)
                result = score(predictions(model, x, None), target, n, train, test,
                               mean, std, order)
                rows.append({"layer": layer, "fraction": fraction,
                             "channels_removed": count, "selection": selection,
                             "random_trial": trial, **result})
        with torch.no_grad():
            model.embed.weight.copy_(embedding)
            model.pos.copy_(position)
            model.out.weight.copy_(readout)
    return rows, {layer: {"absolute_weight_gini": gini(values),
                          "top_10pct_l1_share": float(np.sort(values)[-13:].sum()
                                                      / values.sum())}
                  for layer, values in ranks.items()}


def output_mode_ablation(base_prediction, target, n, train, test, mean, std, order, rng):
    sorted_prediction = base_prediction[order]
    coefficients = dct_ortho(sorted_prediction)
    power = np.sum(coefficients[1:] ** 2, axis=1)
    ranked = np.argsort(power)[::-1] + 1
    top = ranked[:MODE_COUNT]
    low = np.arange(1, MODE_COUNT + 1)
    dc = coefficients[0] / np.sqrt(len(n))
    low_only = dc + contribution(coefficients, low)
    if not np.allclose(dct_ortho(low_only)[:MODE_COUNT + 1],
                       coefficients[:MODE_COUNT + 1], atol=1e-8):
        raise ValueError("Restricted inverse DCT failed to reconstruct the retained modes")
    variants = [
        ("dc_only", 0, np.broadcast_to(dc, sorted_prediction.shape).copy()),
        ("low32_only", 0, low_only),
        ("remove_low32", 0, sorted_prediction - contribution(coefficients, low)),
        ("top32_only", 0, dc + contribution(coefficients, top)),
        ("remove_top32", 0, sorted_prediction - contribution(coefficients, top)),
    ]
    for trial in range(RANDOM_TRIALS):
        random_modes = rng.choice(np.arange(1, len(n)), MODE_COUNT, replace=False)
        variants.append(("remove_random32", trial,
                         sorted_prediction - contribution(coefficients, random_modes)))
    rows = []
    inverse_order = np.argsort(order)
    for name, trial, sorted_values in variants:
        result = score(sorted_values[inverse_order], target, n, train, test,
                       mean, std, order)
        rows.append({"intervention": name, "random_trial": trial, **result})
    return rows, {
        "low32_output_power_share": float(power[:MODE_COUNT].sum() / power.sum()),
        "top32_output_power_share": float(power[top - 1].sum() / power.sum()),
        "top_1pct_output_power_share": float(
            np.sort(power)[-max(1, round(0.01 * len(power))):].sum() / power.sum()),
        "top_10pct_output_power_share": float(
            np.sort(power)[-max(1, round(0.10 * len(power))):].sum() / power.sum()),
        "effective_mode_count": float(power.sum() ** 2 / np.sum(power ** 2)),
        "top32_mode_indices": top.tolist(),
    }


def fit_ridge(features, target, fit_indices, penalty):
    center = features[fit_indices].mean(axis=0)
    scale = np.maximum(features[fit_indices].std(axis=0), 1e-6)
    standardized = (features - center) / scale
    target_center = target[fit_indices].mean(axis=0)
    gram = standardized[fit_indices].T @ standardized[fit_indices]
    weights = np.linalg.solve(
        gram + penalty * np.eye(features.shape[1]),
        standardized[fit_indices].T @ (target[fit_indices] - target_center),
    )
    return standardized @ weights + target_center


def train_only_probe(features, normalized_target, train, validation_rng):
    split = validation_rng.permutation(train)
    cut = int(0.8 * len(split))
    fit, validation = split[:cut], split[cut:]
    choices = []
    for penalty in (1e-4, 1e-2, 1.0, 100.0, 10000.0):
        prediction = fit_ridge(features, normalized_target, fit, penalty)
        validation_mse = float(np.mean(
            (prediction[validation] - normalized_target[validation]) ** 2))
        choices.append((validation_mse, penalty))
    validation_mse, selected_penalty = min(choices)
    return fit_ridge(features, normalized_target, train, selected_penalty), {
        "selected_ridge_penalty": selected_penalty,
        "validation_normalized_mse": validation_mse,
    }


def readout_probes(model, x, target, n, train, test, mean, std, order,
                   checkpoint_dir, final_prediction):
    normalized_target = (target - mean) / std
    rows = []
    for step in (0, 1000, 40000):
        path = checkpoint_dir / f"step_{step:06d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not np.array_equal(checkpoint["train_indices"], train):
            raise ValueError(f"Train split mismatch: {path}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        output, hidden = output_and_hidden(model, x)
        if step == 40000 and not np.allclose(output, final_prediction, atol=1e-5):
            raise ValueError("Final manual readout does not match the standard forward pass")
        rows.append({"step": step, "probe": "trained_readout",
                     **score(output, target, n, train, test, mean, std, order)})
        fitted, fit_info = train_only_probe(
            hidden, normalized_target, train, np.random.default_rng(2026))
        rows.append({"step": step, "probe": "hidden_linear_refit", **fit_info,
                     **score(fitted, target, n, train, test, mean, std, order)})
        if step == 40000:
            calibrated, fit_info = train_only_probe(
                output, normalized_target, train, np.random.default_rng(2026))
            rows.append({"step": step, "probe": "output_linear_calibration", **fit_info,
                         **score(calibrated, target, n, train, test, mean, std, order)})
    bits = x.cpu().numpy()
    fitted, fit_info = train_only_probe(
        bits, normalized_target, train, np.random.default_rng(2026))
    rows.append({"step": 40000, "probe": "raw_bits_linear_control", **fit_info,
                 **score(fitted, target, n, train, test, mean, std, order)})
    return rows


def lowpass_sweep(base_prediction, target, n, test, mean, std, order):
    coefficients = dct_ortho(base_prediction[order])
    inverse_order = np.argsort(order)
    if not np.allclose(inverse_dct_ortho(coefficients)[inverse_order],
                       base_prediction, atol=1e-8):
        raise ValueError("FFT inverse DCT does not reconstruct the unfiltered output")
    ordered_test = test[np.argsort(n[test])]
    calibration, audit = ordered_test[::2], ordered_test[1::2]
    normalized_target = (target - mean) / std

    def subset_score(prediction, indices):
        raw = prediction * std + mean
        relative = np.max(np.abs(raw - target)
                          / np.maximum(np.abs(target), 1e-8), axis=1)
        return {
            "normalized_mse": float(np.mean(
                (prediction[indices] - normalized_target[indices]) ** 2)),
            "relative_accuracy_5pct": float(np.mean(relative[indices] < 0.05)),
        }

    rows = []
    for cutoff in LOWPASS_CUTOFFS:
        retained = coefficients.copy()
        retained[cutoff + 1:] = 0
        prediction = inverse_dct_ortho(retained)[inverse_order]
        rows.append({
            "nonconstant_modes_retained": cutoff,
            "calibration": subset_score(prediction, calibration),
            "audit": subset_score(prediction, audit),
            "full_test": subset_score(prediction, test),
        })
    selected_mse = min(rows, key=lambda row: row["calibration"]["normalized_mse"])
    selected_accuracy = max(
        rows, key=lambda row: row["calibration"]["relative_accuracy_5pct"])
    return {
        "split_rule": "Sort held-out examples by n; alternate into calibration and audit halves.",
        "calibration_size": len(calibration),
        "audit_size": len(audit),
        "selected_cutoff_by_calibration_mse": selected_mse["nonconstant_modes_retained"],
        "selected_cutoff_by_calibration_accuracy": selected_accuracy["nonconstant_modes_retained"],
        "rows": rows,
    }


def plot_results(outdir, baseline, weights, outputs, probes, lowpass):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for ax, layer, title in zip(
            axes, ("embedding_delta", "position_vectors", "readout"),
            ("Bit-embedding difference", "Position vectors", "Final readout weights")):
        layer_rows = [row for row in weights if row["layer"] == layer]
        for selection, color in (("top", "#dc2626"), ("bottom", "#2563eb"),
                                 ("random", "#64748b")):
            x, y, error = [], [], []
            for fraction in FRACTIONS:
                values = [row["test_normalized_mse"] / baseline["test_normalized_mse"]
                          for row in layer_rows
                          if row["fraction"] == fraction and row["selection"] == selection]
                x.append(fraction * 100)
                y.append(np.mean(values))
                error.append(np.std(values))
            ax.errorbar(x, y, yerr=error if selection == "random" else None,
                        marker="o", color=color, lw=1.8, capsize=3, label=selection)
        ax.axhline(1, color="#111827", lw=1, ls="--")
        ax.set(xlabel="Channels removed (%)", ylabel="Held-out MSE / baseline MSE",
               title=title, xticks=[10, 25])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("Weight ablations · raw circulant checkpoint at step 40,000")
    fig.savefig(outdir / "weight_ablation.png", dpi=170)
    plt.close(fig)

    names = ["baseline", "dc_only", "low32_only", "remove_low32",
             "top32_only", "remove_top32", "remove_random32"]
    values = {"baseline": baseline}
    for name in names[1:]:
        selected = [row for row in outputs if row["intervention"] == name]
        values[name] = {key: float(np.mean([row[key] for row in selected]))
                        for key in ("test_normalized_mse", "test_relative_accuracy_5pct")}
    labels = [name.replace("_", "\n") for name in names]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    colors = ["#334155", "#a855f7", "#2563eb", "#ea580c", "#059669",
              "#dc2626", "#94a3b8"]
    for ax, key, ylabel in zip(axes,
                               ("test_normalized_mse", "test_relative_accuracy_5pct"),
                               ("Held-out normalized MSE", "Held-out 5% relative accuracy")):
        ax.bar(np.arange(len(names)), [values[name][key] for name in names], color=colors)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(np.arange(len(names)), labels)
    fig.suptitle("Post-hoc output DCT filtering · 32 nonconstant modes")
    fig.savefig(outdir / "output_mode_ablation.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for name, color in (("trained_readout", "#64748b"),
                        ("hidden_linear_refit", "#2563eb")):
        rows = [row for row in probes if row["probe"] == name]
        for ax, key, ylabel in zip(
                axes, ("test_normalized_mse", "test_relative_accuracy_5pct"),
                ("Held-out normalized MSE", "Held-out 5% relative accuracy")):
            ax.plot(range(len(rows)), [row[key] for row in rows],
                    marker="o", color=color, label=name.replace("_", " "))
            ax.set(xlabel="Checkpoint step", ylabel=ylabel, xticks=[0, 1, 2],
                   xticklabels=["0", "1,000", "40,000"])
            ax.grid(alpha=0.2)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Train-only linear readout refit on frozen hidden states")
    fig.savefig(outdir / "readout_probe.png", dpi=170)
    plt.close(fig)

    rows = lowpass["rows"]
    positions = np.arange(len(rows))
    baseline_row = rows[-1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for split, color in (("calibration", "#2563eb"), ("audit", "#ea580c")):
        ratios = [row[split]["normalized_mse"]
                  / baseline_row[split]["normalized_mse"] for row in rows]
        accuracy = [row[split]["relative_accuracy_5pct"] for row in rows]
        axes[0].plot(positions, ratios, marker="o", color=color, label=split)
        axes[1].plot(positions, accuracy, marker="o", color=color, label=split)
    selected = LOWPASS_CUTOFFS.index(
        lowpass["selected_cutoff_by_calibration_mse"])
    for ax in axes:
        ax.axvline(selected, color="#475569", lw=1, ls="--")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].axhline(1, color="#111827", lw=1, ls=":")
    axes[0].set_ylabel("Normalized MSE / unfiltered MSE")
    axes[1].set_ylabel("Strict 5% relative accuracy")
    axes[1].set_xticks(positions, [str(value) for value in LOWPASS_CUTOFFS],
                       rotation=30)
    axes[1].set_xlabel("Nonconstant DCT modes retained")
    fig.suptitle("Output low-pass sweep · dashed line selected by calibration MSE")
    fig.savefig(outdir / "lowpass_sweep.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    _, x, _, _, n, target = load_data("circulant", device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != 40000:
        raise ValueError("This report expects the archived step-40,000 checkpoint")
    model = BitSizeTransformer().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    train = np.asarray(checkpoint["train_indices"])
    test = np.asarray(checkpoint["test_indices"])
    mean = np.asarray(checkpoint["target_mean"])
    std = np.asarray(checkpoint["target_std"])
    order = np.argsort(n)
    rng = np.random.default_rng(2026)
    base_prediction = predictions(model, x, None)
    baseline = score(base_prediction, target, n, train, test, mean, std, order)
    weights, weight_concentration = weight_ablation(
        model, x, target, n, train, test, mean, std, order, rng)
    if not np.allclose(predictions(model, x[:64], None), base_prediction[:64],
                       atol=1e-5):
        raise ValueError("Weight interventions were not fully restored")
    outputs, mode_concentration = output_mode_ablation(
        base_prediction, target, n, train, test, mean, std, order, rng)
    probes = readout_probes(model, x, target, n, train, test, mean, std, order,
                            args.checkpoint.parent, base_prediction)
    lowpass = lowpass_sweep(base_prediction, target, n, test, mean, std, order)
    target_normalized = (target - mean) / std
    target_coefficients = dct_ortho(target_normalized[order])
    target_power = np.sum(target_coefficients[1:] ** 2, axis=1)
    with np.load(HERE / "animations/analysis/circulant_raw.npz") as animation:
        if int(animation["steps"][-1]) != checkpoint["step"]:
            raise ValueError("Animation and checkpoint final steps differ")
        animation_reference = {
            "initial_output_dct_gini": float(animation["output_gini"][0]),
            "final_output_dct_gini": float(animation["output_gini"][-1]),
            "final_test_relative_accuracy_5pct": float(animation["test_accuracy"][-1]),
        }
    if not np.isclose(animation_reference["final_output_dct_gini"],
                      baseline["output_dct_gini"], atol=1e-5):
        raise ValueError("Checkpoint output Gini differs from the GIF analysis")
    result = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint["step"]),
        "n_samples": len(n),
        "train_size": len(train),
        "test_size": len(test),
        "baseline": baseline,
        "animation_reference": animation_reference,
        "weight_concentration": weight_concentration,
        "mode_concentration": {**mode_concentration,
                               "target_dct_gini": gini(target_power),
                               "target_low32_power_share": float(
                                   target_power[:MODE_COUNT].sum() / target_power.sum())},
        "weight_ablations": weights,
        "output_mode_ablations": outputs,
        "readout_probes": probes,
        "lowpass_sweep": lowpass,
        "notes": [
            "The GIF's Gini is across ordered-size DCT output power, not weight sparsity.",
            "Embedding ablation sets selected channels of both bit tokens to their midpoint.",
            "Position-vector ablation zeros selected channels only at the 16 bit positions, not the CLS position.",
            "Readout ablation zeros selected output-weight columns; biases stay fixed.",
            "Output DCT filtering uses all inputs but no held-out targets; it is transductive post-processing, not a learned circuit.",
            "Random controls use five independent channel or mode selections.",
            "Linear probes and output calibration use only training targets; ridge penalty is selected on an 80/20 split of the training indices.",
            "Low-pass cutoff selection uses one half of the original held-out set; the other half is reserved for audit.",
        ],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_results(args.outdir, baseline, weights, outputs, probes, lowpass)
    print(json.dumps({
        "baseline": baseline,
        "weight_concentration": weight_concentration,
        "mode_concentration": result["mode_concentration"],
        "output_mode_ablations": outputs,
        "readout_probes": probes,
        "lowpass_selection": {
            key: value for key, value in lowpass.items() if key != "rows"
        },
        "output_dir": str(args.outdir),
    }, indent=2))


if __name__ == "__main__":
    main()
