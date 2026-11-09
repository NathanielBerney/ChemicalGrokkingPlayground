#!/usr/bin/env python3
"""Put modular, gap, and pericyclic PCA structures side by side."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_family_pca import HERE, ROOT, load_model, readout_features, pca


def modular_panel(ax):
    path = HERE.parents[1] / "nanda_original_reproduction/figure_run_seed0/figure_snapshots/step_040000.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    tokens = state["W_E"][:, :113].numpy().T
    coords, _, variance = pca(tokens)
    power = np.sum(np.abs(np.fft.rfft(coords, axis=0)) ** 2, axis=1)
    power[0] = 0
    frequency = int(np.argmax(power))
    phase = (frequency * np.arange(113)) % 113
    order = np.argsort(phase)
    ax.plot(coords[order, 0], coords[order, 1], color="#94a3b8", lw=.7)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=phase, cmap="twilight",
                         vmin=0, vmax=113, s=15)
    ax.set(title=f"Modular addition · k={frequency} ring\nPC1+PC2={variance[:2].sum():.1%}",
           xlabel="PC1", ylabel="PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.2)
    return scatter


def gap_panel(ax):
    path = ROOT / "circulant_gap_probe/fourier_progress/runs/log_posbits/snapshots/step_040000.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    position = state["inner.pos"][0, 1:].numpy()
    table = state["inner.embed.table.weight"].numpy().reshape(16, 2, 128)
    tokens = (position[:, None, :] + table).reshape(32, 128)
    coords, _, variance = pca(tokens)
    for position_index in range(16):
        pair = coords[2 * position_index:2 * position_index + 2]
        ax.plot(pair[:, 0], pair[:, 1], color="#94a3b8", lw=.8, alpha=.7)
    color = np.repeat(np.arange(15, -1, -1), 2)
    for value, marker in ((0, "o"), (1, "^")):
        ids = np.arange(value, 32, 2)
        scatter = ax.scatter(coords[ids, 0], coords[ids, 1], c=color[ids], cmap="viridis",
                             vmin=0, vmax=15, marker=marker, s=29,
                             edgecolor="black", linewidth=.2)
    ax.set(title=f"Circulant gap · paired bit vectors\nPC1+PC2={variance[:2].sum():.1%}",
           xlabel="PC1", ylabel="PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.2)
    return scatter


def pericyclic_panel(ax):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model, _ = load_model("pericyclic", device)
    from sys import path as sys_path
    sys_path.insert(0, str(ROOT))
    from train_family import load_data
    _, matrix, mask, _, _, labels = load_data("pericyclic", device)
    features = readout_features(model, "pericyclic", matrix, mask)
    coords, basis, variance = pca(features)
    centered = features - features.mean(axis=0)
    readout = model.out.weight.detach().cpu().numpy().ravel()
    projection = basis @ readout
    component = int(np.argmax(projection ** 2))
    signal = centered @ basis[component]
    if np.corrcoef(signal, labels)[0, 1] < 0:
        signal = -signal
    sample = np.random.default_rng(0).choice(len(labels), 3000, replace=False)
    for value, color, name in ((0, "#2563eb", "forbidden"),
                               (1, "#ea580c", "allowed")):
        selected = sample[labels[sample] == value]
        ax.scatter(coords[selected, 0], signal[selected], s=4, alpha=.35,
                   color=color, label=name)
    ax.set(title=f"Pericyclic rule · PC{component+1} separates labels\n"
                 f"PC{component+1} variance={variance[component]:.1%}",
           xlabel="PC1", ylabel=f"Readout-aligned PC{component+1}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    modular_colors = modular_panel(axes[0])
    gap_colors = gap_panel(axes[1])
    pericyclic_panel(axes[2])
    fig.colorbar(modular_colors, ax=axes[0], shrink=.6, label="41·a mod 113")
    fig.colorbar(gap_colors, ax=axes[1], shrink=.6, label="log₂(bit place value)")
    fig.suptitle("PCA reveals different learned geometries across tasks", fontsize=17)
    fig.savefig(HERE / "nanda_vs_chemical_pca.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
