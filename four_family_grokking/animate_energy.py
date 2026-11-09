#!/usr/bin/env python3
"""Animate embeddings and size-axis Fourier spectra for the three energy families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from animation_utils import dct_ortho, gini, metrics_table, pca_basis, save_gif
from train_family import BitSizeTransformer, load_data


HERE = Path(__file__).resolve().parent
OUT = HERE / "animations"
EDGES = np.array([1, 4, 16, 64, 256, 1024, 4096, 10000])
LABELS = [f"{lo}–{hi-1}" for lo, hi in zip(EDGES[:-1], EDGES[1:])]
BLUE = "#2563eb"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
GREEN = "#059669"


@torch.inference_mode()
def output_and_hidden(model, x, batch_size=2048):
    output, hidden = [], []
    for first in range(0, len(x), batch_size):
        bits = x[first:first + batch_size]
        stream = torch.cat((model.cls.expand(len(bits), -1, -1), model.embed(bits)), dim=1) + model.pos
        update, _ = model.attn(stream, stream, stream, need_weights=False)
        stream = stream + update
        stream = stream + model.ff(stream)
        features = stream[:, 0]
        prediction = model.out(features)
        output.append(prediction.cpu().numpy())
        hidden.append(features.cpu().numpy())
    return np.concatenate(output), np.concatenate(hidden)


def effective_tokens(model):
    position = model.pos.detach().cpu().numpy()[0, 1:]
    values = model.embed.weight.detach().cpu().numpy()
    return (position[:, None, :] + values[None, :, :]).reshape(32, 128)


def project_current(rows, basis):
    centered = rows - rows.mean(axis=0)
    points = centered @ basis
    radius = float(np.sqrt(np.mean(np.sum(points ** 2, axis=1))))
    return points / max(radius, 1e-12), radius


def band_density(power):
    return np.array([np.mean(power[lo - 1:hi - 1])
                     for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def band_fraction(power):
    total = max(power.sum(), 1e-30)
    return np.array([power[lo - 1:hi - 1].sum() / total
                     for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def load_archived(family, objective):
    run = OUT / ("runs_log" if objective == "log" else "runs_raw") / family
    config = json.loads((run / "config.json").read_text())
    files = sorted((run / "snapshots").glob("step_*.pt"))
    if not (run / "last.pt").exists() or len(files) < 2:
        raise ValueError(f"Incomplete archived run: {run}")
    steps = np.array([int(path.stem.split("_")[-1]) for path in files])
    if steps[0] != 0 or steps[-1] != config["steps"] or np.any(np.diff(steps) != 1000):
        raise ValueError(f"Missing or unordered snapshots: {run}")
    return run, config, files, steps


def analyze(family, objective, device):
    run, config, files, steps = load_archived(family, objective)
    _, x, _, _, n, y = load_data(family, device)
    order = np.argsort(n)
    chosen = np.linspace(0, len(n) - 1, 2200, dtype=int)
    model = BitSizeTransformer().to(device).eval()
    final = torch.load(files[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(final["model"])
    _, final_hidden = output_and_hidden(model, x)
    hidden_basis, hidden_fraction = pca_basis(final_hidden)
    token_basis, token_fraction = pca_basis(effective_tokens(model))
    mean = np.asarray(final["target_mean"])
    std = np.asarray(final["target_std"])
    transformed = np.log(np.abs(y)) if objective == "log" else y
    target = (transformed - mean) / std
    target_coeff = dct_ortho(target[order])
    target_power = np.sum(target_coeff[1:] ** 2, axis=1)
    train_indices = np.asarray(final["train_indices"])
    test_indices = np.asarray(final["test_indices"])
    train_mask = np.zeros(len(n), dtype=bool)
    train_mask[train_indices] = True
    records = {key: [] for key in ("token_points", "hidden_points", "token_radius",
                                  "hidden_radius", "output_density", "hidden_fraction",
                                  "output_gini", "hidden_gini", "low_error", "high_error",
                                  "train_accuracy", "test_accuracy", "train_loss", "test_loss")}
    for path, step in zip(files, steps):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint["step"]) != step or not np.array_equal(checkpoint["train_indices"], train_indices):
            raise ValueError(f"Snapshot metadata mismatch: {path}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        prediction, hidden = output_and_hidden(model, x)
        if step in (0, steps[-1]):
            with torch.inference_mode():
                direct = model(x[:64]).cpu().numpy()
            if not np.allclose(prediction[:64], direct, rtol=1e-5, atol=2e-5):
                raise ValueError(f"Manual hidden readout disagrees with forward: {path}")
        token_points, token_radius = project_current(effective_tokens(model), token_basis)
        hidden_points, hidden_radius = project_current(hidden[chosen], hidden_basis)
        coefficients = dct_ortho(prediction[order])
        output_power = np.sum(coefficients[1:] ** 2, axis=1)
        hidden_coeff = dct_ortho(hidden[order])
        hidden_power = np.sum(hidden_coeff[1:] ** 2, axis=1)
        error_power = np.sum((coefficients - target_coeff) ** 2, axis=1) / (len(n) * 2)
        low_error = float(error_power[:33].sum())
        high_error = float(error_power[33:].sum())
        actual_error = float(np.mean((prediction - target) ** 2))
        if not np.isclose(low_error + high_error, actual_error, rtol=1e-7, atol=1e-7):
            raise ValueError(f"DCT error power does not reconstruct MSE at step {step}")
        raw = (np.exp(np.clip(prediction * std + mean, -30, 5)) * np.array([-1, 1])
               if objective == "log" else prediction * std + mean)
        correct = np.all(np.abs(raw - y) / np.maximum(np.abs(y), 1e-8) < .05, axis=1)
        squared = np.mean((prediction - target) ** 2, axis=1)
        values = {"token_points": token_points, "hidden_points": hidden_points,
                  "token_radius": token_radius, "hidden_radius": hidden_radius,
                  "output_density": band_density(output_power),
                  "hidden_fraction": band_fraction(hidden_power),
                  "output_gini": gini(output_power), "hidden_gini": gini(hidden_power),
                  "low_error": low_error, "high_error": high_error,
                  "train_accuracy": float(correct[train_indices].mean()),
                  "test_accuracy": float(correct[test_indices].mean()),
                  "train_loss": float(squared[train_indices].mean()),
                  "test_loss": float(squared[test_indices].mean())}
        for key, value in values.items():
            records[key].append(value)
    records = {key: np.asarray(value) for key, value in records.items()}
    metrics = metrics_table(run / "metrics.csv")
    for index in (1, len(steps) // 2, len(steps) - 1):
        row = np.where(metrics["step"] == steps[index])[0]
        if len(row) != 1:
            raise ValueError(f"No metric row for step {steps[index]}")
        for key in ("train_loss", "test_loss", "train_relative_accuracy", "test_relative_accuracy"):
            record_key = key.replace("relative_", "")
            tolerance = .003 if "accuracy" in key else 2e-4
            if not np.isclose(records[record_key][index], metrics[key][row[0]],
                              rtol=1e-4, atol=tolerance):
                raise ValueError(f"Metrics mismatch: {family} {objective} step {steps[index]} {key}")
    records.update({"steps": steps, "n_sample": n[chosen], "sample_train": train_mask[chosen],
                    "target_density": band_density(target_power),
                    "target_gini": gini(target_power),
                    "hidden_pca_fraction": hidden_fraction,
                    "token_pca_fraction": token_fraction})
    return records


def scale_limits(points, minimum=3.0):
    span = max(minimum, float(np.percentile(np.abs(points), 99.8)) * 1.1)
    return -span, span


def draw_embeddings(index, family, objective, data):
    steps = data["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=100,
                            gridspec_kw={"height_ratios": [1.3, 1]}, layout="constrained")
    tokens = data["token_points"][index]
    ax = axes[0, 0]
    for bit in range(16):
        ax.plot(tokens[2 * bit:2 * bit + 2, 0], tokens[2 * bit:2 * bit + 2, 1],
                color="#94a3b8", lw=1)
    for value, marker in ((0, "o"), (1, "^")):
        ids = np.arange(value, 32, 2)
        ax.scatter(tokens[ids, 0], tokens[ids, 1], c=np.repeat(np.arange(15, -1, -1), 2)[ids],
                   cmap="viridis", vmin=0, vmax=15, marker=marker, s=46,
                   edgecolors="black", linewidths=.3, label=f"bit {value}")
    ax.set(xlim=scale_limits(data["token_points"][:, :, 0]),
           ylim=scale_limits(data["token_points"][:, :, 1]),
           xlabel="Final input-token PC1 / frame RMS", ylabel="Final input-token PC2 / frame RMS",
           title=f"Effective bit tokens · final PC1+2 {data['token_pca_fraction']:.1%}\n"
                 f"raw projected radius {data['token_radius'][index]:.3g}")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax = axes[0, 1]
    points = data["hidden_points"][index]
    scatter = ax.scatter(points[:, 0], points[:, 1], c=np.log10(data["n_sample"]),
                         cmap="viridis", s=7, alpha=.6, edgecolors="none")
    ax.set(xlim=scale_limits(data["hidden_points"][:, :, 0]),
           ylim=scale_limits(data["hidden_points"][:, :, 1]),
           xlabel="Final readout PC1 / frame RMS", ylabel="Final readout PC2 / frame RMS",
           title=f"CLS readout states · final PC1+2 {data['hidden_pca_fraction']:.1%}\n"
                 f"raw projected radius {data['hidden_radius'][index]:.3g}")
    ax.grid(alpha=.2)
    fig.colorbar(scatter, ax=ax, shrink=.8, label="log₁₀(n)")
    visible = slice(1, index + 1)
    ax = axes[1, 0]
    for key, color, label in (("train_accuracy", BLUE, "train"),
                              ("test_accuracy", ORANGE, "test")):
        ax.plot(steps[visible] + 1, data[key][visible], color=color, label=label)
        ax.scatter(1, data[key][0], marker="x", color=color, s=24)
    ax.set(xscale="log", xlim=(1, steps[-1] * 1.06), ylim=(0, 1.03),
           xlabel="Optimizer step + 1", ylabel="Both energies within 5%", title="Strict relative accuracy")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for key, color, label in (("train_loss", BLUE, "train"), ("test_loss", ORANGE, "test")):
        ax.plot(steps[visible] + 1, np.maximum(data[key][visible], 1e-12),
                color=color, label=label)
        ax.scatter(1, max(data[key][0], 1e-12), marker="x", color=color, s=24)
    all_loss = np.concatenate([data["train_loss"], data["test_loss"]])
    ax.set(xscale="log", yscale="log", xlim=(1, steps[-1] * 1.06),
           ylim=(max(1e-12, all_loss.min() * .5), all_loss.max() * 2),
           xlabel="Optimizer step + 1", ylabel="Normalized target MSE", title="Fit")
    ax.grid(alpha=.2, which="both"); ax.legend(fontsize=8)
    fig.suptitle(f"{family.capitalize()} · {objective} target · embedding evolution · step {int(steps[index]):,}\n"
                 "Each PCA basis is fixed to the final checkpoint; coordinates are centered and RMS-scaled per frame",
                 fontsize=12)
    return fig


def draw_frequency(index, family, objective, data):
    steps = data["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=100, layout="constrained")
    ax = axes[0, 0]
    x = np.arange(len(LABELS))
    current = np.maximum(data["output_density"][index], 1e-20)
    target = np.maximum(data["target_density"], 1e-20)
    ax.bar(x, current, color=BLUE, alpha=.8, label="model")
    ax.plot(x, target, color=ORANGE, marker="o", lw=1.5, label="target")
    ax.set(xticks=x, xticklabels=LABELS, yscale="log",
           ylim=(max(1e-20, min(current.min(), target.min()) * .4),
                 max(current.max(), target.max()) * 2),
           xlabel="Ordered-size DCT mode range", ylabel="Power per mode",
           title=f"Output · Gini {data['output_gini'][index]:.3f} (target {data['target_gini']:.3f})")
    ax.tick_params(axis="x", labelrotation=30); ax.grid(alpha=.2, axis="y", which="both")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.bar(x, data["hidden_fraction"][index], color=PURPLE)
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 1),
           xlabel="Ordered-size DCT mode range", ylabel="Share of nonconstant power",
           title=f"128D readout state · Gini {data['hidden_gini'][index]:.3f}")
    ax.tick_params(axis="x", labelrotation=30); ax.grid(alpha=.2, axis="y")
    ax = axes[1, 0]
    visible = slice(1, index + 1)
    series = ((data["low_error"] + data["high_error"], ORANGE, "all modes"),
              (data["low_error"], BLUE, "modes 0–32"),
              (data["high_error"], PURPLE, "modes 33+") )
    for values, color, label in series:
        ax.plot(steps[visible] + 1, np.maximum(values[visible], 1e-14),
                color=color, label=label)
        ax.scatter(1, max(values[0], 1e-14), marker="x", color=color, s=24)
    all_error = np.concatenate([values for values, _, _ in series])
    ax.set(xscale="log", yscale="log", xlim=(1, steps[-1] * 1.06),
           ylim=(max(1e-14, all_error[all_error > 0].min() * .5), all_error.max() * 2),
           xlabel="Optimizer step + 1", ylabel="Normalized full-grid MSE contribution",
           title="Parseval error by frequency")
    ax.grid(alpha=.2, which="both"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for key, color, label in (("train_accuracy", BLUE, "train"),
                              ("test_accuracy", ORANGE, "test")):
        ax.plot(steps[visible] + 1, data[key][visible], color=color, label=label)
        ax.scatter(1, data[key][0], marker="x", color=color, s=24)
    ax.set(xscale="log", xlim=(1, steps[-1] * 1.06), ylim=(0, 1.03),
           xlabel="Optimizer step + 1", ylabel="Both energies within 5%",
           title="Held-out behavior")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"{family.capitalize()} · {objective} target · frequency evolution · step {int(steps[index]):,}\n"
                 "DCT along sorted molecule size; step 0 is a separate marker; history axes stay fixed",
                 fontsize=12)
    return fig


def make_one(family, objective, fps, device):
    data = analyze(family, objective, device)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "gifs").mkdir(parents=True, exist_ok=True)
    name = f"{family}_{objective}"
    np.savez_compressed(OUT / "analysis" / f"{name}.npz", **data)
    save_gif(OUT / "gifs" / f"{name}_embeddings.gif", len(data["steps"]), fps,
             lambda i: draw_embeddings(i, family, objective, data))
    save_gif(OUT / "gifs" / f"{name}_frequency.gif", len(data["steps"]), fps,
             lambda i: draw_frequency(i, family, objective, data))
    summary = {"family": family, "objective": objective,
            "frames": len(data["steps"]), "last_step": int(data["steps"][-1]),
            "final_test_relative_accuracy": float(data["test_accuracy"][-1]),
            "final_output_gini": float(data["output_gini"][-1]),
            "final_hidden_gini": float(data["hidden_gini"][-1]),
            "final_hidden_pc12_fraction": data["hidden_pca_fraction"],
            "final_token_pc12_fraction": data["token_pca_fraction"]}
    (OUT / "analysis" / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize_saved():
    rows = []
    for family in ("circulant", "mobius", "linear"):
        for objective in ("raw", "log"):
            path = OUT / "analysis" / f"{family}_{objective}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path) as data:
                rows.append({"family": family, "objective": objective,
                             "frames": len(data["steps"]),
                             "last_step": int(data["steps"][-1]),
                             "final_test_relative_accuracy": float(data["test_accuracy"][-1]),
                             "final_output_gini": float(data["output_gini"][-1]),
                             "final_hidden_gini": float(data["hidden_gini"][-1]),
                             "final_hidden_pc12_fraction": float(data["hidden_pca_fraction"]),
                             "final_token_pc12_fraction": float(data["token_pca_fraction"])})
    rows.sort(key=lambda row: (row["family"], row["objective"]))
    (OUT / "energy_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def render_saved(family, objective, fps):
    path = OUT / "analysis" / f"{family}_{objective}.npz"
    with np.load(path) as saved:
        data = {key: (float(saved[key]) if saved[key].ndim == 0 else saved[key])
                for key in saved.files}
    name = f"{family}_{objective}"
    save_gif(OUT / "gifs" / f"{name}_embeddings.gif", len(data["steps"]), fps,
             lambda i: draw_embeddings(i, family, objective, data))
    save_gif(OUT / "gifs" / f"{name}_frequency.gif", len(data["steps"]), fps,
             lambda i: draw_frequency(i, family, objective, data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=("circulant", "mobius", "linear"),
                        default=["circulant", "mobius", "linear"])
    parser.add_argument("--objectives", nargs="+", choices=("raw", "log"), default=["raw", "log"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--summary-only", action="store_true",
                        help="Rebuild the combined summary from saved analysis arrays")
    parser.add_argument("--render-only", action="store_true",
                        help="Regenerate GIFs from saved analysis arrays without reloading models")
    args = parser.parse_args()
    if args.summary_only:
        print(json.dumps(summarize_saved(), indent=2))
        return
    if args.fps < 1:
        parser.error("fps must be positive")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    if args.render_only:
        for family in args.families:
            for objective in args.objectives:
                render_saved(family, objective, args.fps)
        return
    summary_path = OUT / "energy_summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    summary = {(row["family"], row["objective"]): row for row in existing}
    for family in args.families:
        for objective in args.objectives:
            row = make_one(family, objective, args.fps, torch.device(args.device))
            summary[(family, objective)] = row
            summary_path.write_text(json.dumps(sorted(summary.values(),
                                                       key=lambda r: (r["family"], r["objective"])),
                                               indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
