#!/usr/bin/env python3
"""PCA of chemical gap-model weights, effective bit tokens, and hidden states."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_fourier import (HERE, VARIANTS, build_model, output_and_features)
from sys import path as sys_path
sys_path.insert(0, str(HERE.parent))
from train_gap import load_data  # noqa: E402


def snapshots(variant):
    result = []
    for path in sorted((HERE / "runs" / variant / "snapshots").glob("step_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        result.append((int(data["step"]), data))
    if len(result) != 45 or result[0][0] != 0 or result[-1][0] != 40_000:
        raise ValueError(f"Incomplete snapshot series for {variant}")
    return result


def pca_fit(rows):
    matrix = np.asarray(rows, dtype=np.float64)
    mean = matrix.mean(axis=0)
    _, singular, basis = np.linalg.svd(matrix - mean, full_matrices=False)
    variance = singular ** 2 / np.sum(singular ** 2)
    return mean, basis, variance


def effective_tokens(state, variant):
    position = state["inner.pos"][0, 1:, :].numpy()
    if variant == "log_bits":
        shared = state["inner.embed.weight"].numpy()
        return (position[:, None, :] + shared[None, :, :]).reshape(32, 128)
    table = state["inner.embed.table.weight"].numpy().reshape(16, 2, 128)
    return (position[:, None, :] + table).reshape(32, 128)


def weight_trajectory(states):
    keys = [key for key, value in states[0][1]["model"].items()
            if value.is_floating_point()]
    matrix = np.stack([np.concatenate([record["model"][key].numpy().ravel()
                                       for key in keys])
                       for _, record in states]).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    centered = matrix - matrix.mean(axis=0)
    gram = centered @ centered.T
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = np.maximum(eigenvalue[order], 0)
    scores = eigenvector[:, order] * np.sqrt(eigenvalue)
    variance = eigenvalue / eigenvalue.sum()
    for col in range(2):
        if scores[-1, col] < scores[0, col]:
            scores[:, col] *= -1
    return scores, variance, norms, int(matrix.shape[1])


def accuracy_series(variant):
    with (HERE / "runs" / variant / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return (np.asarray([float(row["step"]) for row in rows]),
            np.asarray([float(row["test_accuracy_rel_1pct"]) for row in rows]))


def plot_embedding_evolution(all_states):
    variants = ("log_bits", "log_posbits")
    selected = (0, 10_000, 20_000, 40_000)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    report = {}
    for row, variant in enumerate(variants):
        lookup = {step: record["model"] for step, record in all_states[variant]}
        final = effective_tokens(lookup[40_000], variant)
        mean, basis, variance = pca_fit(final)
        projections = [(effective_tokens(lookup[step], variant) - mean) @ basis[:2].T
                       for step in selected]
        limit = max(np.max(np.abs(points)) for points in projections) * 1.08
        if limit == 0:
            limit = 1
        for col, (step, coords) in enumerate(zip(selected, projections)):
            ax = axes[row, col]
            for bit_position in range(16):
                pair = coords[2 * bit_position:2 * bit_position + 2]
                ax.plot(pair[:, 0], pair[:, 1], color="#94a3b8", lw=.8, alpha=.6)
            color = np.repeat(np.arange(15, -1, -1), 2)
            for value, marker in ((0, "o"), (1, "^")):
                ids = np.arange(value, 32, 2)
                scatter = ax.scatter(coords[ids, 0], coords[ids, 1], c=color[ids],
                                     cmap="viridis", vmin=0, vmax=15,
                                     marker=marker, s=26, edgecolor="black", linewidth=.2)
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit),
                   xlabel="Final PC1", ylabel="Final PC2",
                   title=f"{variant.replace('_', ' ')} · step {step:,}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.2)
        report[variant] = {
            "effective_token_pc1_pc2_final_variance_fraction": float(variance[:2].sum()),
            "effective_token_pc1_pc2_initial_variance_fraction":
                float(pca_fit(effective_tokens(lookup[0], variant))[2][:2].sum()),
            "bit_delta_norm_by_position_msb_first":
                np.linalg.norm((final.reshape(16, 2, 128)[:, 1]
                                - final.reshape(16, 2, 128)[:, 0]), axis=1).tolist(),
        }
    fig.colorbar(scatter, ax=axes, shrink=.65, label="Bit significance: log₂(place value)")
    fig.suptitle("Gap transformers · effective 0/1 token embeddings in final PCA planes\n"
                 "Circle = bit 0; triangle = bit 1; paired tokens joined",
                 fontsize=15)
    fig.savefig(HERE / "chemical_embedding_pca.png", dpi=180)
    plt.close(fig)
    return report


def plot_final_embeddings(all_states):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, variant in zip(axes, ("log_bits", "log_posbits")):
        state = all_states[variant][-1][1]["model"]
        tokens = effective_tokens(state, variant)
        mean, basis, variance = pca_fit(tokens)
        coords = (tokens - mean) @ basis[:2].T
        for position in range(16):
            pair = coords[2 * position:2 * position + 2]
            ax.plot(pair[:, 0], pair[:, 1], color="#94a3b8", lw=1, alpha=.7)
        bit_significance = np.repeat(np.arange(15, -1, -1), 2)
        for value, marker in ((0, "o"), (1, "^")):
            ids = np.arange(value, 32, 2)
            scatter = ax.scatter(coords[ids, 0], coords[ids, 1],
                                 c=bit_significance[ids], cmap="viridis", vmin=0, vmax=15,
                                 marker=marker, s=36, edgecolor="black", linewidth=.3)
        for position in (0, 3, 7, 11, 14, 15):
            x, y = coords[2 * position + 1]
            ax.annotate(f"2^{15-position}", (x, y), xytext=(4, 3),
                        textcoords="offset points", fontsize=7)
        ax.set(xlabel="PC1", ylabel="PC2",
               title=f"{variant.replace('_', ' ')} · PC1+PC2={variance[:2].sum():.1%}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
    fig.colorbar(scatter, ax=axes, shrink=.7, label="log₂(bit place value)")
    fig.suptitle("Final effective bit embeddings · zoomed PCA projection\n"
                 "Circle = bit 0; triangle = bit 1; pairs joined", fontsize=15)
    fig.savefig(HERE / "chemical_embedding_pca_final_zoom.png", dpi=180)
    plt.close(fig)


def plot_bit_effect_norms(embedding_report):
    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    exponent = np.arange(15, -1, -1)
    for variant, color in (("log_bits", "#7c3aed"), ("log_posbits", "#2563eb")):
        values = embedding_report[variant]["bit_delta_norm_by_position_msb_first"]
        ax.plot(exponent, values, marker="o", label=variant.replace("_", " "), color=color)
    ax.axvspan(-.5, 1.5, color="#94a3b8", alpha=.15, label="constant bits in this dataset")
    ax.set(xlabel="Bit place exponent j in 2^j", ylabel="Norm of embedding(bit=1) − embedding(bit=0)",
           title="Learned bit effects by binary place value")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(HERE / "chemical_bit_effect_norms.png", dpi=180)
    plt.close(fig)


def plot_weight_trajectories(all_states):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
    report = {}
    for row, variant in enumerate(VARIANTS):
        states = all_states[variant]
        steps = np.asarray([step for step, _ in states])
        coordinates, variance, norms, count = weight_trajectory(states)
        metric_steps, metric_test = accuracy_series(variant)
        accuracy = np.interp(steps, metric_steps, metric_test)
        for col, color_values, cmap, label in ((0, steps, "viridis", "Step"),
                                               (1, accuracy, "plasma", "Held-out 1% accuracy")):
            ax = axes[row, col]
            ax.plot(coordinates[:, 0], coordinates[:, 1], color="#94a3b8", lw=1)
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=color_values,
                                 cmap=cmap, s=30, vmin=0,
                                 vmax=40_000 if col == 0 else 1)
            ax.scatter(coordinates[0, 0], coordinates[0, 1], color="black", marker="x",
                       s=50, label="initial")
            ax.scatter(coordinates[-1, 0], coordinates[-1, 1], color="black", marker="*",
                       s=100, label="final")
            ax.set(xlabel="Weight PC1", ylabel="Weight PC2",
                   title=f"{variant.replace('_', ' ')} · PC1+PC2={variance[:2].sum():.1%}")
            ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.colorbar(scatter, ax=ax, shrink=.7, label=label)
        report[variant] = {
            "parameter_count": count,
            "weight_pc1_pc2_variance_fraction": float(variance[:2].sum()),
            "weight_pc1_variance_fraction": float(variance[0]),
            "initial_weight_norm": float(norms[0]),
            "final_weight_norm": float(norms[-1]),
            "weight_norm_ratio_final_to_initial": float(norms[-1] / norms[0]),
            "coordinates": [{"step": int(step), "pc1": float(x), "pc2": float(y),
                             "test_accuracy_1pct": float(a), "weight_norm": float(norm)}
                            for step, (x, y), a, norm in zip(steps, coordinates[:, :2],
                                                              accuracy, norms)],
        }
    fig.suptitle("Gap models · PCA trajectories of full checkpoint weight vectors", fontsize=16)
    fig.savefig(HERE / "chemical_weight_trajectory_pca.png", dpi=180)
    plt.close(fig)
    return report


def plot_hidden_pca(all_states, device, n):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    report = {}
    sample = np.linspace(0, len(n) - 1, 2500, dtype=int)
    for ax, variant in zip(axes, VARIANTS):
        checkpoint = all_states[variant][-1][1]
        model = build_model(variant, checkpoint, device)
        _, features = output_and_features(model, variant, n, checkpoint, device)
        mean, basis, variance = pca_fit(features)
        coordinates = (features[sample] - mean) @ basis[:2].T
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=np.log10(n[sample]),
                             cmap="viridis", s=5, alpha=.55)
        ax.set(xlabel="Hidden PC1", ylabel="Hidden PC2",
               title=f"{variant.replace('_', ' ')} · PC1+PC2={variance[:2].sum():.1%}")
        ax.grid(alpha=.2)
        report[variant] = {"hidden_pc1_pc2_variance_fraction": float(variance[:2].sum()),
                           "hidden_pc1_variance_fraction": float(variance[0])}
    fig.colorbar(scatter, ax=axes, shrink=.7, label="log₁₀ ring size n")
    fig.suptitle("Final readout features · PCA colored by molecular size", fontsize=16)
    fig.savefig(HERE / "chemical_hidden_pca.png", dpi=180)
    plt.close(fig)
    return report


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    all_states = {variant: snapshots(variant) for variant in VARIANTS}
    n, _ = load_data()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    embedding = plot_embedding_evolution(all_states)
    plot_final_embeddings(all_states)
    plot_bit_effect_norms(embedding)
    weight = plot_weight_trajectories(all_states)
    hidden = plot_hidden_pca(all_states, device, n)
    summary = {"embedding": embedding, "weight_trajectory": weight, "hidden": hidden,
               "interpretation": "PCA is descriptive; full-weight trajectories can reflect weight decay."}
    (HERE / "pca_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"embedding": embedding,
                      "weight_trajectory": {v: {k: value for k, value in result.items()
                                                if k != "coordinates"}
                                            for v, result in weight.items()},
                      "hidden": hidden}, indent=2))


if __name__ == "__main__":
    main()
