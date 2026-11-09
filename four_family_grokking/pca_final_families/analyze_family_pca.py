#!/usr/bin/env python3
"""Final-checkpoint PCA views of all four chemical benchmark models."""

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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from train_family import (BitSizeTransformer, MatrixTransformer, load_data)  # noqa: E402

FAMILIES = ("circulant", "mobius", "linear", "pericyclic")


def pca(rows):
    matrix = np.asarray(rows, dtype=np.float64)
    mean = matrix.mean(axis=0)
    _, singular, basis = np.linalg.svd(matrix - mean, full_matrices=False)
    variance = singular ** 2 / np.sum(singular ** 2)
    return (matrix - mean) @ basis[:2].T, basis, variance


def load_model(family, device):
    root = "runs" if family == "pericyclic" else "runs_log"
    checkpoint = torch.load(ROOT / root / family / "last.pt",
                            map_location=device, weights_only=False)
    model = (MatrixTransformer() if family == "pericyclic"
             else BitSizeTransformer()).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def readout_features(model, family, x, mask, batch_size=2048):
    chunks = []
    for first in range(0, len(x), batch_size):
        batch = x[first:first + batch_size]
        site_mask = None if mask is None else mask[first:first + batch_size]
        if family == "pericyclic":
            tokens = model.row_embed(batch)
            stream = torch.cat((model.cls.expand(len(batch), -1, -1), tokens), dim=1) + model.pos
            padding = torch.cat((torch.zeros((len(batch), 1), dtype=torch.bool,
                                             device=batch.device), ~site_mask), dim=1)
            update, _ = model.attn(stream, stream, stream,
                                   key_padding_mask=padding, need_weights=False)
        else:
            stream = torch.cat((model.cls.expand(len(batch), -1, -1),
                                model.embed(batch)), dim=1) + model.pos
            update, _ = model.attn(stream, stream, stream, need_weights=False)
        stream = stream + update
        stream = stream + model.ff(stream)
        chunks.append(stream[:, 0].cpu().numpy())
    return np.concatenate(chunks)


def effective_bits(model):
    position = model.pos.detach().cpu().numpy()[0, 1:]
    embedding = model.embed.weight.detach().cpu().numpy()
    return (position[:, None, :] + embedding[None, :, :]).reshape(32, 128)


def plot_energy_hidden(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for ax, family in zip(axes, FAMILIES[:3]):
        info = data[family]
        n = info["n"]
        coords = info["coordinates"]
        sample = np.linspace(0, len(n) - 1, 2500, dtype=int)
        scatter = ax.scatter(coords[sample, 0], coords[sample, 1],
                             c=np.log10(n[sample]), cmap="viridis", s=5, alpha=.55)
        ax.set(xlabel="Hidden PC1", ylabel="Hidden PC2",
               title=f"{family.capitalize()} · PC1+PC2={info['variance'][:2].sum():.1%}")
        ax.grid(alpha=.2)
    fig.colorbar(scatter, ax=axes, shrink=.7, label="log₁₀ carbon count n")
    fig.suptitle("Final HOMO/LUMO model readout features · log-target runs")
    fig.savefig(HERE / "frontier_hidden_pca.png", dpi=180)
    plt.close(fig)


def plot_energy_embeddings(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for ax, family in zip(axes, FAMILIES[:3]):
        tokens = data[family]["tokens"]
        coords, _, variance = pca(tokens)
        for position in range(16):
            pair = coords[2 * position:2 * position + 2]
            ax.plot(pair[:, 0], pair[:, 1], color="#94a3b8", lw=.8, alpha=.7)
        color = np.repeat(np.arange(15, -1, -1), 2)
        for value, marker in ((0, "o"), (1, "^")):
            ids = np.arange(value, 32, 2)
            scatter = ax.scatter(coords[ids, 0], coords[ids, 1], c=color[ids],
                                 cmap="viridis", vmin=0, vmax=15, marker=marker,
                                 s=29, edgecolor="black", linewidth=.2)
        ax.set(xlabel="Input PC1", ylabel="Input PC2",
               title=f"{family.capitalize()} · PC1+PC2={variance[:2].sum():.1%}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
    fig.colorbar(scatter, ax=axes, shrink=.7, label="log₂ bit place value")
    fig.suptitle("Final input embeddings · shared 0/1 token plus learned bit position\n"
                 "Circle = bit 0; triangle = bit 1; paired tokens joined")
    fig.savefig(HERE / "frontier_input_embedding_pca.png", dpi=180)
    plt.close(fig)


def plot_pericyclic_hidden(data):
    info = data["pericyclic"]
    coords, labels, sizes = info["coordinates"], info["labels"], info["n"]
    sample = np.random.default_rng(0).choice(len(coords), 4000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for label, color, name in ((0, "#2563eb", "forbidden"),
                               (1, "#ea580c", "allowed")):
        chosen = sample[labels[sample] == label]
        axes[0].scatter(coords[chosen, 0], coords[chosen, 1], s=5, alpha=.35,
                        color=color, label=name)
    axes[0].set(xlabel="Hidden PC1", ylabel="Hidden PC2",
                title="Colored by thermal label")
    axes[0].legend(fontsize=8)
    for size, color in ((4, "#7c3aed"), (6, "#059669"), (8, "#f97316")):
        chosen = sample[sizes[sample] == size]
        axes[1].scatter(coords[chosen, 0], coords[chosen, 1], s=5, alpha=.35,
                        color=color, label=f"{size} active sites")
    axes[1].set(xlabel="Hidden PC1", ylabel="Hidden PC2",
                title="Colored by transition-state size")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"Pericyclic final readout features · PC1+PC2={info['variance'][:2].sum():.1%}")
    fig.savefig(HERE / "pericyclic_hidden_pca.png", dpi=180)
    plt.close(fig)


def plot_pericyclic_input_weights(model):
    rows = model.row_embed.weight.detach().cpu().numpy().T
    positions = model.pos.detach().cpu().numpy()[0, 1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, matrix, title in ((axes[0], rows, "Row-projection columns"),
                              (axes[1], positions, "Site-position vectors")):
        coords, _, variance = pca(matrix)
        ax.plot(coords[:, 0], coords[:, 1], color="#94a3b8", lw=1)
        ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(8), cmap="viridis", s=40)
        for index, (x, y) in enumerate(coords):
            ax.annotate(str(index), (x, y), xytext=(4, 3),
                        textcoords="offset points", fontsize=8)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"{title} · PC1+PC2={variance[:2].sum():.1%}")
        ax.grid(alpha=.2)
    fig.suptitle("Pericyclic input weights · eight matrix-column and site positions")
    fig.savefig(HERE / "pericyclic_input_weight_pca.png", dpi=180)
    plt.close(fig)


def plot_pericyclic_signal(data):
    info = data["pericyclic"]
    coords, labels = info["coordinates"], info["labels"]
    signal = info["readout_pc_signal"]
    pc_number = info["readout_pc_number"]
    sample = np.random.default_rng(0).choice(len(coords), 4000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for label, color, name in ((0, "#2563eb", "forbidden"),
                               (1, "#ea580c", "allowed")):
        chosen = sample[labels[sample] == label]
        axes[0].scatter(coords[chosen, 0], signal[chosen], s=5, alpha=.3,
                        color=color, label=name)
        axes[1].hist(signal[labels == label], bins=60, density=True, alpha=.45,
                     color=color, label=name)
    axes[0].set(xlabel="Hidden PC1", ylabel=f"Readout-aligned hidden PC{pc_number}",
                title=f"PC{pc_number} carries the classification direction")
    axes[1].set(xlabel=f"Readout-aligned hidden PC{pc_number}", ylabel="Density",
                title="Label distributions")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Pericyclic label structure appears outside the top-variance PCA plane")
    fig.savefig(HERE / "pericyclic_signal_pca.png", dpi=180)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    data = {}
    report = {}
    for family in FAMILIES:
        model, checkpoint = load_model(family, device)
        _, x, mask, _, n, labels = load_data(family, device)
        features = readout_features(model, family, x, mask)
        coords, basis, variance = pca(features)
        weights = model.out.weight.detach().cpu().numpy()
        readout_fraction = float(np.sum((weights @ basis[:2].T) ** 2)
                                 / np.sum(weights ** 2))
        data[family] = {"coordinates": coords, "variance": variance,
                        "n": n, "labels": labels,
                        "tokens": effective_bits(model) if family != "pericyclic" else None}
        item = {"model_source": "runs/pericyclic/last.pt" if family == "pericyclic"
                else f"runs_log/{family}/last.pt",
                "hidden_pc1_pc2_variance_fraction": float(variance[:2].sum()),
                "hidden_pc1_variance_fraction": float(variance[0]),
                "readout_weight_fraction_in_pc1_pc2": readout_fraction,
                "checkpoint_step": int(checkpoint["step"])}
        if family != "pericyclic":
            item["hidden_pc1_log_n_correlation"] = float(np.corrcoef(coords[:, 0], np.log(n))[0, 1])
            item["input_pc1_pc2_variance_fraction"] = float(pca(data[family]["tokens"])[2][:2].sum())
        else:
            item["hidden_pc1_label_correlation"] = float(np.corrcoef(coords[:, 0], labels)[0, 1])
            item["hidden_pc2_label_correlation"] = float(np.corrcoef(coords[:, 1], labels)[0, 1])
            readout = weights.ravel()
            readout_power_by_pc = (basis @ readout) ** 2 / np.sum(readout ** 2)
            strongest = int(np.argmax(readout_power_by_pc))
            full_coordinates = (features - features.mean(axis=0)) @ basis.T
            correlation = float(np.corrcoef(full_coordinates[:, strongest], labels)[0, 1])
            data[family]["readout_pc_number"] = strongest + 1
            data[family]["readout_pc_signal"] = (np.sign(correlation)
                                                 * full_coordinates[:, strongest])
            item["strongest_readout_pc"] = strongest + 1
            item["readout_weight_fraction_in_strongest_pc"] = float(readout_power_by_pc[strongest])
            item["hidden_variance_fraction_in_strongest_pc"] = float(variance[strongest])
            item["strongest_pc_label_correlation"] = correlation
        report[family] = item
        print(f"{family}: PCA complete", flush=True)
        if family == "pericyclic":
            plot_pericyclic_input_weights(model)
    plot_energy_hidden(data)
    plot_energy_embeddings(data)
    plot_pericyclic_hidden(data)
    plot_pericyclic_signal(data)
    (HERE / "pca_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
