#!/usr/bin/env python3
"""Nanda-inspired functional Fourier and Gini probes for gap checkpoints.

The DCT axis here is ordered molecule size, not ring-site orbital position.
These probes describe the model's learned scalar function and hidden states;
they do not identify a Fourier circuit in the Hamiltonian eigenbasis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_gap import (GapBitTransformer, PositionAwareGapTransformer, ScalarGapMLP,
                       load_data, score)  # noqa: E402

VARIANTS = ("log_bits", "log_posbits", "log_scalar")
COLORS = {"log_bits": "#7c3aed", "log_posbits": "#2563eb", "log_scalar": "#059669"}
LOW_MODES = 32


def dct_ortho(values):
    """Orthonormal DCT-II along axis zero, implemented with NumPy FFT."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    even = np.concatenate((values, values[::-1]), axis=0)
    coefficients = np.fft.rfft(even, axis=0)[:n]
    shape = (n,) + (1,) * (values.ndim - 1)
    phase = np.exp(-1j * np.pi * np.arange(n).reshape(shape) / (2 * n))
    result = .5 * np.real(coefficients * phase)
    result[0] *= np.sqrt(1 / n)
    result[1:] *= np.sqrt(2 / n)
    return result


def gini(power):
    values = np.sort(np.asarray(power, dtype=np.float64).ravel())
    total = values.sum()
    if total <= 1e-20:
        return 0.0
    count = len(values)
    return float(((2 * np.arange(1, count + 1) - count - 1) @ values)
                 / (count * total))


def build_model(variant, checkpoint, device):
    if variant == "log_bits":
        model = GapBitTransformer()
    elif variant == "log_posbits":
        model = PositionAwareGapTransformer()
    else:
        model = ScalarGapMLP()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.inference_mode()
def output_and_features(model, variant, n, checkpoint, device, batch_size=2048):
    if variant == "log_scalar":
        values = (np.log(n) - checkpoint["input_mean"]) / checkpoint["input_std"]
        x = torch.as_tensor(values[:, None], dtype=torch.float32, device=device)
    else:
        bits = ((n[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.int64)
        x = torch.as_tensor(bits, device=device)
    outputs, features = [], []
    for first in range(0, len(x), batch_size):
        batch = x[first:first + batch_size]
        if variant == "log_scalar":
            hidden = model.layers[:-1](batch)
            normalized = model.layers[-1](hidden).squeeze(-1)
        else:
            inner = model.inner
            embedded = torch.cat((inner.cls.expand(len(batch), -1, -1),
                                  inner.embed(batch)), dim=1) + inner.pos
            update, _ = inner.attn(embedded, embedded, embedded, need_weights=False)
            stream = embedded + update
            stream = stream + inner.ff(stream)
            hidden = stream[:, 0]
            normalized = inner.out(hidden).squeeze(-1)
        # This manual readout must agree with the model's actual forward path.
        if not torch.allclose(normalized, model(batch), atol=2e-5, rtol=2e-5):
            raise AssertionError("Feature readout differs from model forward")
        outputs.append(normalized.cpu().numpy())
        features.append(hidden.cpu().numpy())
    raw = np.concatenate(outputs) * checkpoint["target_scale"] + checkpoint["target_center"]
    return raw, np.concatenate(features)


def low_pass(coefficients, count=LOW_MODES):
    n = len(coefficients)
    k = np.arange(1, count + 1)
    j = np.arange(n)[:, None]
    basis = np.sqrt(2 / n) * np.cos(np.pi * (j + .5) * k / n)
    low = coefficients[0] / np.sqrt(n) + basis @ coefficients[1:count + 1]
    return low


def transform_report(prediction, features, n_values, truth, train, test, target_dct):
    n = len(truth)
    output_dct = dct_ortho(prediction)
    feature_dct = dct_ortho(features)
    output_power = output_dct[1:] ** 2
    hidden_power = np.sum(feature_dct[1:] ** 2, axis=1)
    error_power = (output_dct - target_dct) ** 2
    low_error = float(np.sum(error_power[:LOW_MODES + 1]) / n)
    high_error = float(np.sum(error_power[LOW_MODES + 1:]) / n)
    low_target_power = np.sum(target_dct[1:LOW_MODES + 1] ** 2)
    high_target_power = np.sum(target_dct[LOW_MODES + 1:] ** 2)
    low_mode_relative_rmse = float(np.sqrt(np.sum(error_power[1:LOW_MODES + 1])
                                            / low_target_power))
    high_mode_relative_rmse = float(np.sqrt(np.sum(error_power[LOW_MODES + 1:])
                                             / high_target_power))
    centered_pred = output_dct[1:]
    centered_target = target_dct[1:]
    denominator = np.linalg.norm(centered_pred) * np.linalg.norm(centered_target)
    alignment = float(centered_pred @ centered_target / denominator) if denominator else 0.0
    gap_pred = np.exp(np.clip(prediction, -30, 5))
    gap_truth = np.exp(truth)
    small_1000 = test[n_values[test] <= 1000]
    small_100 = test[n_values[test] <= 100]
    low_signal = low_pass(output_dct)
    high_only = prediction - (low_signal - output_dct[0] / np.sqrt(n))
    def ablation(signal):
        gaps = np.exp(np.clip(signal, -30, 5))
        return {"train": score(gaps, gap_truth, train),
                "test": score(gaps, gap_truth, test)}
    periodic_power = np.abs(np.fft.rfft(prediction - prediction.mean())[1:]) ** 2
    return {
        "train_accuracy_1pct": score(gap_pred, gap_truth, train)["accuracy_rel_1pct"],
        "test_accuracy_1pct": score(gap_pred, gap_truth, test)["accuracy_rel_1pct"],
        "test_accuracy_5pct": score(gap_pred, gap_truth, test)["accuracy_rel_5pct"],
        "test_n_le_1000_accuracy_1pct": score(gap_pred, gap_truth, small_1000)["accuracy_rel_1pct"],
        "test_n_le_100_accuracy_1pct": score(gap_pred, gap_truth, small_100)["accuracy_rel_1pct"],
        "output_cosine_gini": gini(output_power),
        "hidden_cosine_gini": gini(hidden_power),
        "output_periodic_fft_gini": gini(periodic_power),
        "output_low32_power_fraction": float(output_power[:LOW_MODES].sum()
                                              / max(output_power.sum(), 1e-20)),
        "hidden_low32_power_fraction": float(hidden_power[:LOW_MODES].sum()
                                              / max(hidden_power.sum(), 1e-20)),
        "target_cosine_alignment": alignment,
        "low_mode_relative_rmse": low_mode_relative_rmse,
        "high_mode_relative_rmse": high_mode_relative_rmse,
        "low32_log_mse_contribution": low_error,
        "remaining_log_mse_contribution": high_error,
        "total_log_mse": low_error + high_error,
        "ablations": {"full": ablation(prediction), "low32_only": ablation(low_signal),
                      "remove_low32": ablation(high_only)},
        "output_dct": output_dct,
        "hidden_power": hidden_power,
    }


def analyze_one(variant, device, n, truth, target_dct):
    files = sorted((HERE / "runs" / variant / "snapshots").glob("step_*.pt"))
    if not files:
        raise FileNotFoundError(f"No snapshots for {variant}")
    records = []
    final_detail = None
    for path in files:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model(variant, checkpoint, device)
        prediction, features = output_and_features(model, variant, n, checkpoint, device)
        info = transform_report(prediction, features, n, truth,
                                checkpoint["train_indices"], checkpoint["test_indices"], target_dct)
        step = int(checkpoint["step"])
        record = {"step": step, **{key: value for key, value in info.items()
                                    if key not in ("ablations", "output_dct", "hidden_power")}}
        records.append(record)
        if step == 40_000:
            final_detail = {"ablations": info["ablations"],
                            "output_dct": info["output_dct"],
                            "hidden_power": info["hidden_power"],
                            "prediction": prediction}
        print(f"{variant}: analyzed step {step}", flush=True)
    if final_detail is None:
        raise ValueError(f"{variant} has no 40,000-step snapshot")
    return records, final_detail


def plot_progress(all_records, target_gini, target_low32):
    fig, axes = plt.subplots(3, 3, figsize=(17, 11), layout="constrained")
    for row, variant in enumerate(VARIANTS):
        records = all_records[variant]
        steps = np.array([r["step"] for r in records])
        ax = axes[row, 0]
        ax.plot(steps, [r["train_accuracy_1pct"] for r in records],
                label="train", color="#2563eb")
        ax.plot(steps, [r["test_accuracy_1pct"] for r in records],
                label="held-out", color="#ea580c")
        ax.set(xlabel="Step", ylabel="Within 1%", ylim=(0, 1.03),
               title=f"{variant} · strict gap accuracy")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax = axes[row, 1]
        ax.plot(steps, [r["output_cosine_gini"] for r in records],
                label="output-function Gini", color="#7c3aed")
        ax.plot(steps, [r["hidden_cosine_gini"] for r in records],
                label="hidden-state Gini", color="#059669")
        ax.axhline(target_gini, ls=":", color="black", label="target-function Gini")
        gini_values = ([target_gini] + [r["output_cosine_gini"] for r in records]
                       + [r["hidden_cosine_gini"] for r in records])
        floor, ceiling = min(gini_values), max(gini_values)
        margin = max((ceiling - floor) * .12, .0002)
        ax.set(xlabel="Step", ylabel="Cosine-power Gini (zoomed)",
               ylim=(floor - margin, min(1.0002, ceiling + margin)),
               title="Spectral concentration across size")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax = axes[row, 2]
        ax.plot(steps, [r["low32_log_mse_contribution"] for r in records],
                label="modes 0–32", color="#2563eb")
        ax.plot(steps, [r["remaining_log_mse_contribution"] for r in records],
                label="remaining modes", color="#ea580c")
        ax.set(xlabel="Step", ylabel="Log-gap MSE contribution", yscale="log",
               title=f"Orthogonal error split; target low32 = {target_low32:.1%} variance")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Gap-model Fourier progress · cosine basis across ordered ring sizes", fontsize=17)
    fig.savefig(HERE / "fourier_progress.png", dpi=175)
    plt.close(fig)


def plot_final_spectra(n, truth, target_dct, details):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for row, variant in enumerate(VARIANTS):
        detail = details[variant]
        output_dct = detail["output_dct"]
        k = np.arange(1, 501)
        ax = axes[row, 0]
        ax.loglog(k, target_dct[1:501] ** 2, label="exact log-gap", color="black")
        ax.loglog(k, output_dct[1:501] ** 2, label="model", color=COLORS[variant], alpha=.8)
        ax.axvline(LOW_MODES, ls="--", color="#64748b", label="32-mode split")
        ax.set(xlabel="Cosine mode k across size index", ylabel="Power", title=variant)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax = axes[row, 1]
        low = low_pass(output_dct)
        ax.plot(n, truth, color="black", lw=1.3, label="exact log-gap")
        ax.plot(n, detail["prediction"], color=COLORS[variant], alpha=.75,
                label="model")
        ax.plot(n, low, color="#f97316", ls="--", lw=1,
                label="model modes 0–32")
        ax.set(xscale="log", xlabel="Ring size n", ylabel="log(gap / |β|)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Final gap functions and their cosine Fourier spectra", fontsize=17)
    fig.savefig(HERE / "final_spectra.png", dpi=175)
    plt.close(fig)


def plot_ablation(details):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    labels = [v.replace("_", " ") for v in VARIANTS]
    x = np.arange(len(VARIANTS))
    for ax, key, title in zip(axes, ("accuracy_rel_1pct", "accuracy_rel_5pct"),
                              ("Within 1% relative gap error", "Within 5% relative gap error")):
        for shift, name, color in ((-.26, "full", "#2563eb"),
                                   (0, "low32_only", "#059669"),
                                   (.26, "remove_low32", "#ea580c")):
            values = [details[v]["ablations"][name]["test"][key] for v in VARIANTS]
            ax.bar(x + shift, values, width=.25, color=color,
                   label=name.replace("_", " "))
        ax.set_xticks(x, labels)
        ax.set(ylabel="Held-out accuracy", ylim=(0, 1.03), title=title)
        ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Filtering output-function cosine modes · not a weight ablation")
    fig.savefig(HERE / "mode_filtering.png", dpi=175)
    plt.close(fig)


def plot_alignment(all_records, target_low32):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), layout="constrained")
    for variant in VARIANTS:
        rows = all_records[variant]
        step = [r["step"] for r in rows]
        axes[0].plot(step, [max(1 - r["target_cosine_alignment"], 1e-10) for r in rows],
                     label=variant.replace("_", " "), color=COLORS[variant])
        axes[1].plot(step, [r["output_low32_power_fraction"] for r in rows],
                     label=variant.replace("_", " "), color=COLORS[variant])
        axes[2].plot(step, [r["low_mode_relative_rmse"] for r in rows],
                     label=variant.replace("_", " ") + " low", color=COLORS[variant])
        axes[2].plot(step, [r["high_mode_relative_rmse"] for r in rows],
                     label=variant.replace("_", " ") + " high", color=COLORS[variant],
                     ls="--")
    axes[1].axhline(target_low32, color="black", ls=":", label="exact target")
    axes[0].set(title="One minus coefficient cosine similarity",
                xlabel="Step", ylabel="Misalignment (log scale)", yscale="log")
    axes[1].set(title="Fraction of nonconstant power in modes 1–32",
                xlabel="Step", ylabel="Power fraction (zoomed)", ylim=(.95, 1.005))
    axes[2].set(title="Error relative to target power in each band",
                xlabel="Step", ylabel="Relative mode RMSE", yscale="log")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Spectral alignment is more informative than concentration alone")
    fig.savefig(HERE / "spectral_alignment.png", dpi=175)
    plt.close(fig)


def plot_small_rings(all_records):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, variant in zip(axes, VARIANTS):
        rows = all_records[variant]
        step = [r["step"] for r in rows]
        ax.plot(step, [r["test_accuracy_1pct"] for r in rows],
                label="all held-out sizes", color="#2563eb")
        ax.plot(step, [r["test_n_le_1000_accuracy_1pct"] for r in rows],
                label="held-out n ≤ 1000", color="#ea580c")
        ax.set(xlabel="Step", ylabel="Within 1% relative gap error", ylim=(0, 1.03),
               title=variant.replace("_", " "))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Strict gap accuracy hides concentrated errors on small rings")
    fig.savefig(HERE / "small_ring_accuracy.png", dpi=175)
    plt.close(fig)


def plot_nanda_reference(all_records):
    path = HERE.parents[2] / "nanda_original_reproduction/run_seed0/metrics.csv"
    with path.open() as handle:
        nanda = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    nanda_step = [float(r["step"]) for r in nanda]
    axes[0].plot(nanda_step, [float(r["embedding_fourier_gini"]) for r in nanda],
                 label="embedding Gini")
    axes[0].plot(nanda_step, [float(r["neuron_logit_fourier_gini"]) for r in nanda],
                 label="neuron-logit Gini")
    axes[0].set(title="Nanda reproduction · Fourier over mod-113 classes",
                xlabel="Step", ylabel="Power Gini", ylim=(0, 1.03))
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    for variant in VARIANTS:
        rows = all_records[variant]
        axes[1].plot([r["step"] for r in rows], [r["output_cosine_gini"] for r in rows],
                     label=variant.replace("_", " "), color=COLORS[variant])
    axes[1].set(title="Gap probes · model output over ordered ring sizes",
                xlabel="Step", ylabel="Cosine-power Gini (zoomed)")
    gap_ginis = [r["output_cosine_gini"] for variant in VARIANTS
                 for r in all_records[variant]]
    axes[1].set_ylim(min(gap_ginis) - .001, min(1.0002, max(gap_ginis) + .001))
    axes[1].grid(alpha=.2); axes[1].legend(fontsize=8)
    fig.suptitle("Fourier Gini trajectories · axes and meanings differ")
    fig.savefig(HERE / "nanda_gini_reference.png", dpi=175)
    plt.close(fig)


def plot_two_axes(n):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    site = np.arange(42)
    axes[0].plot(site, np.cos(2 * np.pi * 10 * site / 42), label="ring-site DFT k=10")
    axes[0].plot(site, np.cos(2 * np.pi * 11 * site / 42), label="ring-site DFT k=11")
    axes[0].set(xlabel="Orbital site j on one n=42 ring", ylabel="Mode amplitude",
                title="Physical Fourier basis of a circulant H")
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    index = np.arange(len(n))
    for mode in (1, 4, 16):
        basis = np.sqrt(2 / len(n)) * np.cos(np.pi * mode * (index + .5) / len(n))
        axes[1].plot(n, basis, label=f"size DCT mode {mode}")
    axes[1].set(xlabel="Ring size n across 10,000 graphs", ylabel="Mode amplitude",
                title="Analysis basis of the learned gap function")
    axes[1].grid(alpha=.2); axes[1].legend(fontsize=8)
    fig.suptitle("Two different Fourier axes · orbital sites versus molecule sizes")
    fig.savefig(HERE / "two_fourier_axes.png", dpi=175)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    n, gap = load_data()
    truth = np.log(gap)
    target_dct = dct_ortho(truth)
    target_power = target_dct[1:] ** 2
    target_gini = gini(target_power)
    target_low32 = float(target_power[:LOW_MODES].sum() / target_power.sum())
    records, details = {}, {}
    for variant in VARIANTS:
        records[variant], details[variant] = analyze_one(variant, device, n, truth, target_dct)
    plot_progress(records, target_gini, target_low32)
    plot_final_spectra(n, truth, target_dct, details)
    plot_ablation(details)
    plot_alignment(records, target_low32)
    plot_small_rings(records)
    plot_nanda_reference(records)
    plot_two_axes(n)
    reference_checkpoint = torch.load(HERE / "runs/log_bits/snapshots/step_040000.pt",
                                      map_location="cpu", weights_only=False)
    target_low_signal = low_pass(target_dct)
    target_high_signal = truth - (target_low_signal - target_dct[0] / np.sqrt(len(n)))
    reference_test = reference_checkpoint["test_indices"]
    report = {"axis": "DCT-II across ordered n=6,10,...,40002",
              "physical_ring_site_Fourier_basis": "separate analytic basis; not learned from this input",
              "low_mode_cutoff": LOW_MODES,
              "target_cosine_gini": target_gini,
              "target_low32_power_fraction": target_low32,
              "target_only_filtering": {
                  "low32_only": score(np.exp(np.clip(target_low_signal, -30, 5)),
                                      gap, reference_test),
                  "remove_low32": score(np.exp(np.clip(target_high_signal, -30, 5)),
                                        gap, reference_test)},
              "runs": {variant: {"progress": records[variant],
                                 "final_ablations": details[variant]["ablations"],
                                 "final_output_dct_power_first_500":
                                     (details[variant]["output_dct"][1:501] ** 2).tolist()}
                       for variant in VARIANTS}}
    (HERE / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"target_gini": target_gini,
                      "target_low32_power_fraction": target_low32,
                      "final": {v: records[v][-1] for v in VARIANTS}}, indent=2))


if __name__ == "__main__":
    main()
