#!/usr/bin/env python3
"""Animate pericyclic representations and cyclic-reference site spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from animation_utils import metrics_table, pca_basis, save_gif
from plot_fourier_bases import gauged_pericyclic_basis
from train_family import MatrixTransformer, load_data, predictions


HERE = Path(__file__).resolve().parent
OUT = HERE / "animations"
BAND_BOUNDS = np.array([0, .25, .5, .75, 1.000001])
BAND_LABELS = ("0–.25", ".25–.5", ".5–.75", ".75–1")
BLUE = "#2563eb"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"


@torch.inference_mode()
def readout_and_site_tokens(model, x, site_mask, site=False, batch_size=512):
    readouts, sites = [], []
    for first in range(0, len(x), batch_size):
        h = x[first:first + batch_size]
        mask = site_mask[first:first + batch_size]
        stream = torch.cat((model.cls.expand(len(h), -1, -1), model.row_embed(h)), dim=1) + model.pos
        padding = torch.cat((torch.zeros((len(h), 1), dtype=torch.bool, device=h.device), ~mask), dim=1)
        update, _ = model.attn(stream, stream, stream, key_padding_mask=padding, need_weights=False)
        stream = stream + update
        stream = stream + model.ff(stream)
        readouts.append(stream[:, 0].cpu().numpy())
        if site:
            sites.append(stream[:, 1:].cpu().numpy())
    return np.concatenate(readouts), np.concatenate(sites) if site else None


def projection(rows, basis):
    centered = rows - rows.mean(axis=0)
    points = centered @ basis
    radius = float(np.sqrt(np.mean(np.sum(points ** 2, axis=1))))
    return points / max(radius, 1e-12), radius


def spectral_bases(H, sizes, selected):
    bases = []
    for index in selected:
        size = int(sizes[index])
        Q, phase = gauged_pericyclic_basis(H[index, :size, :size])
        shift = 0 if phase > 0 else .5
        modes = np.arange(size) + shift
        relative_frequency = 2 * np.minimum(modes, size - modes) / size
        bands = np.digitize(relative_frequency, BAND_BOUNDS[1:])
        bases.append((Q.conj().T, bands, size))
    return bases


def site_spectrum(tokens, bases, labels):
    power = np.zeros((2, 4), dtype=np.float64)
    rms = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=int)
    for feature, (Q_t, bands, size), label in zip(tokens, bases, labels):
        coefficients = Q_t @ feature[:size]
        energy = np.sum(np.abs(coefficients) ** 2, axis=1)
        total = float(energy.sum())
        category = int(label)
        if total > 0:
            for band in range(4):
                power[category, band] += energy[bands == band].sum() / total
        rms[category] += np.sqrt(total / (size * feature.shape[1]))
        counts[category] += 1
    if np.any(counts == 0):
        raise ValueError("A class has no site-spectrum examples")
    return power / counts[:, None], rms / counts


def load_archived():
    run = OUT / "runs" / "pericyclic"
    config = json.loads((run / "config.json").read_text())
    files = sorted((run / "snapshots").glob("step_*.pt"))
    if not (run / "last.pt").exists() or len(files) < 2:
        raise ValueError(f"Incomplete archived run: {run}")
    steps = np.array([int(path.stem.split("_")[-1]) for path in files])
    if steps[0] != 0 or steps[-1] != config["steps"] or np.any(np.diff(steps) != 2000):
        raise ValueError("Missing or unordered pericyclic snapshots")
    return run, files, steps


def analyze(device):
    run, files, steps = load_archived()
    _, x, mask, _, sizes, labels = load_data("pericyclic", device)
    rng = np.random.default_rng(0)
    hidden_indices = np.sort(rng.choice(len(x), 2400, replace=False))
    site_indices = np.concatenate([rng.choice(np.flatnonzero(labels == category), 256,
                                               replace=False) for category in (0, 1)])
    H = x[site_indices].cpu().numpy()
    site_sizes = sizes[site_indices]
    site_bases = spectral_bases(H, site_sizes, np.arange(len(site_indices)))
    model = MatrixTransformer().to(device).eval()
    final = torch.load(files[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(final["model"])
    final_hidden, _ = readout_and_site_tokens(model, x[hidden_indices], mask[hidden_indices])
    hidden_pc, hidden_fraction = pca_basis(final_hidden)
    output_weight = model.out.weight.detach().cpu().numpy().ravel()
    readout_axis = output_weight - np.dot(output_weight, hidden_pc[:, 0]) * hidden_pc[:, 0]
    readout_axis /= np.linalg.norm(readout_axis)
    if np.corrcoef(final_hidden @ readout_axis, labels[hidden_indices])[0, 1] < 0:
        readout_axis *= -1
    hidden_basis = np.stack((hidden_pc[:, 0], readout_axis), axis=1)
    row_basis, row_fraction = pca_basis(model.row_embed.weight.detach().cpu().numpy().T)
    pos_basis, pos_fraction = pca_basis(model.pos.detach().cpu().numpy()[0, 1:])
    train_indices = np.asarray(final["train_indices"])
    test_indices = np.asarray(final["test_indices"])
    data = {key: [] for key in ("row_points", "pos_points", "hidden_points",
                                "row_radius", "pos_radius", "hidden_radius",
                                "site_power", "site_rms", "train_accuracy", "test_accuracy",
                                "train_loss", "test_loss")}
    for path, step in zip(files, steps):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint["step"]) != step or not np.array_equal(checkpoint["train_indices"], train_indices):
            raise ValueError(f"Snapshot metadata mismatch: {path}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        logits = predictions(model, x, mask)
        hidden, _ = readout_and_site_tokens(model, x[hidden_indices], mask[hidden_indices])
        _, tokens = readout_and_site_tokens(model, x[site_indices], mask[site_indices], site=True)
        if step in (0, steps[-1]):
            with torch.inference_mode():
                direct = model(x[hidden_indices[:64]], mask[hidden_indices[:64]])
                manual = model.out(torch.as_tensor(hidden[:64], device=device)).squeeze(-1)
            if not torch.allclose(direct, manual, atol=2e-5, rtol=1e-5):
                raise ValueError(f"Manual hidden readout disagrees with forward: {path}")
        row_points, row_radius = projection(model.row_embed.weight.detach().cpu().numpy().T,
                                            row_basis)
        pos_points, pos_radius = projection(model.pos.detach().cpu().numpy()[0, 1:], pos_basis)
        hidden_points, hidden_radius = projection(hidden, hidden_basis)
        site_power, site_rms = site_spectrum(tokens, site_bases, labels[site_indices])
        if not np.allclose(site_power.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f"Site frequency bins do not sum to one at step {step}")
        correct = (logits >= 0) == labels
        loss = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
        values = {"row_points": row_points, "pos_points": pos_points,
                  "hidden_points": hidden_points, "row_radius": row_radius,
                  "pos_radius": pos_radius, "hidden_radius": hidden_radius,
                  "site_power": site_power, "site_rms": site_rms,
                  "train_accuracy": float(correct[train_indices].mean()),
                  "test_accuracy": float(correct[test_indices].mean()),
                  "train_loss": float(loss[train_indices].mean()),
                  "test_loss": float(loss[test_indices].mean())}
        for key, value in values.items():
            data[key].append(value)
    data = {key: np.asarray(value) for key, value in data.items()}
    metrics = metrics_table(run / "metrics.csv")
    for index in (1, len(steps) // 2, len(steps) - 1):
        row = np.where(metrics["step"] == steps[index])[0]
        if len(row) != 1:
            raise ValueError(f"No metric row for step {steps[index]}")
        for key in ("train_accuracy", "test_accuracy", "train_loss", "test_loss"):
            if not np.isclose(data[key][index], metrics[key][row[0]], rtol=1e-4, atol=2e-4):
                raise ValueError(f"Metrics mismatch: pericyclic step {steps[index]} {key}")
    data.update({"steps": steps, "sample_labels": labels[hidden_indices],
                 "sample_sizes": sizes[hidden_indices], "site_sample_indices": site_indices,
                 "hidden_pc1_fraction": float(hidden_fraction),
                 "row_pca_fraction": float(row_fraction), "pos_pca_fraction": float(pos_fraction)})
    return data


def axis_limits(points, minimum=2.7):
    span = max(minimum, float(np.percentile(np.abs(points), 99.7)) * 1.1)
    return -span, span


def draw_embeddings(index, data):
    steps = data["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=100, layout="constrained")
    for ax, key, title, fraction in ((axes[0, 0], "row", "Matrix-column projection weights",
                                     data["row_pca_fraction"]),
                                    (axes[0, 1], "pos", "Site-position weights",
                                     data["pos_pca_fraction"])):
        points = data[f"{key}_points"][index]
        ax.plot(points[:, 0], points[:, 1], color="#94a3b8", lw=1)
        ax.scatter(points[:, 0], points[:, 1], c=np.arange(8), cmap="viridis", s=43)
        for j, pair in enumerate(points):
            ax.annotate(str(j), pair, xytext=(4, 2), textcoords="offset points", fontsize=8)
        ax.set(xlim=axis_limits(data[f"{key}_points"][:, :, 0]),
               ylim=axis_limits(data[f"{key}_points"][:, :, 1]),
               xlabel="Final PC1 / frame RMS", ylabel="Final PC2 / frame RMS",
               title=f"{title} · final PC1+2 {fraction:.1%}\n"
                     f"raw projected radius {data[f'{key}_radius'][index]:.3g}")
        ax.grid(alpha=.2)
    ax = axes[1, 0]
    points = data["hidden_points"][index]
    for category, color, name in ((0, BLUE, "forbidden"), (1, ORANGE, "allowed")):
        selected = data["sample_labels"] == category
        ax.scatter(points[selected, 0], points[selected, 1], color=color, s=7,
                   alpha=.35, edgecolors="none", label=name)
    ax.set(xlim=axis_limits(data["hidden_points"][:, :, 0]),
           ylim=axis_limits(data["hidden_points"][:, :, 1]),
           xlabel="Final hidden PC1 / frame RMS", ylabel="Final readout direction / frame RMS",
           title=f"CLS readout states · final PC1 {data['hidden_pc1_fraction']:.1%} variance\n"
                 f"raw projected radius {data['hidden_radius'][index]:.3g}")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax = axes[1, 1]
    visible = slice(1, index + 1)
    for key, color, label in (("train_accuracy", BLUE, "train"),
                              ("test_accuracy", ORANGE, "test")):
        ax.plot(steps[visible] + 1, data[key][visible], color=color, label=label)
        ax.scatter(1, data[key][0], marker="x", color=color, s=24)
    ax.set(xscale="log", xlim=(1, steps[-1] * 1.06), ylim=(0, 1.03),
           xlabel="Optimizer step + 1", ylabel="Thermal label accuracy", title="Classification")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"Pericyclic · embedding evolution · step {int(steps[index]):,}\n"
                 "Final PCA/readout axes stay fixed; each frame is centered and divided by its RMS radius",
                 fontsize=12)
    return fig


def draw_frequency(index, data):
    steps = data["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=100, layout="constrained")
    for category, color, name in ((0, BLUE, "forbidden"), (1, ORANGE, "allowed")):
        ax = axes[0, category]
        ax.bar(np.arange(4), data["site_power"][index, category], color=color)
        ax.set(xticks=np.arange(4), xticklabels=BAND_LABELS, ylim=(0, 1),
               xlabel="Relative cyclic wavenumber / π", ylabel="Mean share of site-token power",
               title=f"{name.capitalize()} · raw site RMS {data['site_rms'][index, category]:.3g}")
        ax.grid(alpha=.2, axis="y")
    visible = slice(1, index + 1)
    ax = axes[1, 0]
    for category, color, name in ((0, BLUE, "forbidden"), (1, ORANGE, "allowed")):
        high = data["site_power"][:, category, 2:].sum(axis=1)
        ax.plot(steps[visible] + 1, high[visible], color=color, label=name)
        ax.scatter(1, high[0], marker="x", color=color, s=24)
    ax.set(xscale="log", xlim=(1, steps[-1] * 1.06), ylim=(0, 1),
           xlabel="Optimizer step + 1", ylabel="Power at relative wavenumber ≥ 0.5",
           title="Higher-mode share over training")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for key, color, label in (("train_accuracy", BLUE, "train"),
                              ("test_accuracy", ORANGE, "test")):
        ax.plot(steps[visible] + 1, data[key][visible], color=color, label=label)
        ax.scatter(1, data[key][0], marker="x", color=color, s=24)
    ax.set(xscale="log", xlim=(1, steps[-1] * 1.06), ylim=(0, 1.03),
           xlabel="Optimizer step + 1", ylabel="Thermal label accuracy",
           title="Classification")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"Pericyclic · cyclic-reference site spectrum · step {int(steps[index]):,}\n"
                 "Gauged periodic/anti-periodic site modes; step 0 is a separate marker; reference basis only",
                 fontsize=12)
    return fig


def render_saved(fps):
    with np.load(OUT / "analysis" / "pericyclic.npz") as saved:
        data = {key: (float(saved[key]) if saved[key].ndim == 0 else saved[key])
                for key in saved.files}
    save_gif(OUT / "gifs" / "pericyclic_embeddings.gif", len(data["steps"]), fps,
             lambda i: draw_embeddings(i, data))
    save_gif(OUT / "gifs" / "pericyclic_frequency.gif", len(data["steps"]), fps,
             lambda i: draw_frequency(i, data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--render-only", action="store_true",
                        help="Regenerate GIFs from saved analysis arrays without reloading checkpoints")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    if args.render_only:
        render_saved(args.fps)
        return
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    data = analyze(torch.device(args.device))
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "gifs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "analysis" / "pericyclic.npz", **data)
    save_gif(OUT / "gifs" / "pericyclic_embeddings.gif", len(data["steps"]), args.fps,
             lambda i: draw_embeddings(i, data))
    save_gif(OUT / "gifs" / "pericyclic_frequency.gif", len(data["steps"]), args.fps,
             lambda i: draw_frequency(i, data))
    summary = {"frames": len(data["steps"]), "last_step": int(data["steps"][-1]),
               "final_test_accuracy": float(data["test_accuracy"][-1]),
               "final_hidden_pc1_fraction": data["hidden_pc1_fraction"],
               "final_row_pc12_fraction": data["row_pca_fraction"],
               "final_position_pc12_fraction": data["pos_pca_fraction"],
               "site_examples_per_label": 256}
    (OUT / "pericyclic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
