#!/usr/bin/env python3
"""PCA of the local mod-113 reproduction's embeddings and weight trajectory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAPSHOTS = ROOT / "figure_run_seed0/figure_snapshots"
P = 113


def load_snapshots():
    result = []
    for path in sorted(SNAPSHOTS.glob("step_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        result.append((int(data["step"]), data["model"]))
    if not result or result[0][0] != 0 or result[-1][0] != 40_000:
        raise ValueError("Expected the complete seed-0 snapshot sequence")
    return result


def pca_fit(rows):
    matrix = np.asarray(rows, dtype=np.float64)
    mean = matrix.mean(axis=0)
    _, singular, components = np.linalg.svd(matrix - mean, full_matrices=False)
    explained = singular ** 2 / np.sum(singular ** 2)
    return mean, components, explained


def weight_trajectory(snapshots):
    keys = [key for key, value in snapshots[0][1].items() if value.is_floating_point()]
    weights = np.stack([np.concatenate([state[key].numpy().ravel() for key in keys])
                        for _, state in snapshots]).astype(np.float64)
    centered = weights - weights.mean(axis=0)
    gram = centered @ centered.T
    eigval, eigvec = np.linalg.eigh(gram)
    order = np.argsort(eigval)[::-1]
    eigval = np.maximum(eigval[order], 0)
    scores = eigvec[:, order] * np.sqrt(eigval)
    variance = eigval / eigval.sum()
    # Fix arbitrary PCA sign so the trajectory is oriented consistently.
    for col in range(2):
        if scores[-1, col] < scores[0, col]:
            scores[:, col] *= -1
    return scores, variance, int(weights.shape[1])


def embedding_k_fraction(embedding, frequency):
    centered = embedding - embedding.mean(axis=0)
    transformed = np.fft.rfft(centered, axis=0)
    power = np.sum(np.abs(transformed) ** 2, axis=1)
    power[0] = 0
    return float(power[frequency] / power[1:].sum())


def curve_data():
    with (ROOT / "figure_run_seed0/metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    step = np.asarray([float(row["step"]) for row in rows])
    test = np.asarray([float(row["test_accuracy"]) for row in rows])
    train = np.asarray([float(row["train_accuracy"]) for row in rows])
    return step, train, test


def draw_embedding_evolution(snapshots, final_mean, final_basis, frequency, explained):
    chosen = (0, 5_000, 10_000, 13_000, 40_000)
    lookup = {step: state for step, state in snapshots}
    token_ids = np.arange(P)
    phase = (frequency * token_ids) % P
    phase_order = np.argsort(phase)
    projections = [((lookup[step]["W_E"][:, :P].numpy().T - final_mean) @ final_basis[:2].T)
                   for step in chosen]
    limit = max(np.max(np.abs(points)) for points in projections) * 1.08
    fig, axes = plt.subplots(1, len(chosen), figsize=(18, 4.2), layout="constrained")
    for ax, step, points in zip(axes, chosen, projections):
        ax.plot(points[phase_order, 0], points[phase_order, 1],
                color="#94a3b8", lw=.7, alpha=.7)
        scatter = ax.scatter(points[:, 0], points[:, 1], c=phase, cmap="twilight",
                             vmin=0, vmax=P, s=19, zorder=2)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Final PC1",
               ylabel="Final PC2", title=f"step {step:,}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
    fig.colorbar(scatter, ax=axes, shrink=.65, label=f"Phase k·a mod 113; k={frequency}")
    fig.suptitle("Nanda reproduction · learned token embeddings in the final PCA plane\n"
                 f"Final PC1+PC2 explain {explained[:2].sum():.1%} of embedding variance",
                 fontsize=15)
    fig.savefig(HERE / "embedding_pca_evolution.png", dpi=180)
    plt.close(fig)


def draw_frequency_pairs(final_embedding, mean, basis, explained):
    projected = (final_embedding - mean) @ basis.T
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    report = []
    for pair, ax in enumerate(axes):
        one, two = 2 * pair, 2 * pair + 1
        fft = np.fft.rfft(projected[:, [one, two]], axis=0)
        power = np.sum(np.abs(fft) ** 2, axis=1)
        power[0] = 0
        frequency = int(np.argmax(power))
        purity = float(power[frequency] / power.sum())
        phase = (frequency * np.arange(P)) % P
        order = np.argsort(phase)
        ax.plot(projected[order, one], projected[order, two],
                color="#94a3b8", lw=.7, alpha=.7)
        scatter = ax.scatter(projected[:, one], projected[:, two], c=phase,
                             cmap="twilight", vmin=0, vmax=P, s=22)
        ax.set(xlabel=f"PC{one+1}", ylabel=f"PC{two+1}",
               title=f"k={frequency}; pair power at k={purity:.1%}\n"
                     f"variance={explained[[one, two]].sum():.1%}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
        report.append({"pair": [one + 1, two + 1], "dominant_modular_frequency": frequency,
                       "pair_frequency_power_fraction": purity,
                       "embedding_variance_fraction": float(explained[[one, two]].sum())})
    fig.colorbar(scatter, ax=axes, shrink=.65, label="Dominant-frequency phase")
    fig.suptitle("Nanda reproduction · several PCA planes reveal Fourier-frequency structure",
                 fontsize=15)
    fig.savefig(HERE / "embedding_pca_frequency_pairs.png", dpi=180)
    plt.close(fig)
    return report


def draw_trajectory(steps, scores, variance, metric_step, test_accuracy, first_test_99):
    test_at_snapshot = np.interp(steps, metric_step, test_accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax in axes:
        ax.plot(scores[:, 0], scores[:, 1], color="#94a3b8", lw=1)
        ax.scatter(scores[0, 0], scores[0, 1], color="black", marker="x", s=60,
                   label="initial")
        ax.scatter(scores[-1, 0], scores[-1, 1], color="black", marker="*", s=110,
                   label="final")
        ax.set(xlabel="Weight PC1", ylabel="Weight PC2")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    a = axes[0].scatter(scores[:, 0], scores[:, 1], c=steps, cmap="viridis", s=35)
    b = axes[1].scatter(scores[:, 0], scores[:, 1], c=test_at_snapshot,
                        cmap="plasma", vmin=0, vmax=1, s=35)
    axes[0].set_title("Colored by optimizer step")
    axes[1].set_title("Colored by held-out accuracy")
    fig.colorbar(a, ax=axes[0], shrink=.7, label="Step")
    fig.colorbar(b, ax=axes[1], shrink=.7, label="Test accuracy")
    transition = int(np.argmin(np.abs(steps - first_test_99)))
    for ax in axes:
        ax.annotate("first ≥99% test", (scores[transition, 0], scores[transition, 1]),
                    xytext=(10, 14), textcoords="offset points",
                    arrowprops={"arrowstyle": "->", "color": "#334155"}, fontsize=8)
    fig.suptitle("Nanda reproduction · PCA of full checkpoint weight vectors\n"
                 f"PC1+PC2 explain {variance[:2].sum():.1%} of trajectory variance",
                 fontsize=15)
    fig.savefig(HERE / "weight_trajectory_pca.png", dpi=180)
    plt.close(fig)


def draw_structure_progress(steps, fractions, train, test, metric_step, frequency):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    axes[0].plot(metric_step, train, label="train", color="#2563eb")
    axes[0].plot(metric_step, test, label="test", color="#ea580c")
    axes[0].set(xlabel="Step", ylabel="Accuracy", ylim=(0, 1.03),
                title="Modular addition accuracy")
    axes[1].plot(steps, fractions, color="#7c3aed", marker="o", ms=3)
    axes[1].set(xlabel="Step", ylabel="Fraction of embedding Fourier power",
                title=f"Power at final PC-pair frequency k={frequency}")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Emergence of PCA-visible Fourier structure during grokking")
    fig.savefig(HERE / "pca_structure_progress.png", dpi=180)
    plt.close(fig)


def main():
    snapshots = load_snapshots()
    steps = np.asarray([step for step, _ in snapshots])
    final_embedding = snapshots[-1][1]["W_E"][:, :P].numpy().T
    mean, basis, explained = pca_fit(final_embedding)
    projected = (final_embedding - mean) @ basis[:2].T
    power = np.sum(np.abs(np.fft.rfft(projected, axis=0)) ** 2, axis=1)
    power[0] = 0
    frequency = int(np.argmax(power))
    if frequency == 0:
        raise ValueError("No nonconstant embedding Fourier frequency")
    draw_embedding_evolution(snapshots, mean, basis, frequency, explained)
    pairs = draw_frequency_pairs(final_embedding, mean, basis, explained)
    trajectory, trajectory_variance, n_parameters = weight_trajectory(snapshots)
    metric_step, train_accuracy, test_accuracy = curve_data()
    first_test_99 = int(metric_step[np.flatnonzero(test_accuracy >= .99)[0]])
    draw_trajectory(steps, trajectory, trajectory_variance,
                    metric_step, test_accuracy, first_test_99)
    fractions = [embedding_k_fraction(state["W_E"][:, :P].numpy().T, frequency)
                 for _, state in snapshots]
    draw_structure_progress(steps, fractions, train_accuracy, test_accuracy,
                            metric_step, frequency)
    report = {"source_run": "nanda_original_reproduction/figure_run_seed0",
              "snapshot_count": len(snapshots), "first_test_accuracy_99pct_step": first_test_99,
              "embedding_final_pc1_pc2_variance_fraction": float(explained[:2].sum()),
              "embedding_final_pc1_pc2_dominant_modular_frequency": frequency,
              "embedding_frequency_pairs": pairs,
              "embedding_frequency_power_fraction_by_step":
                  [{"step": int(step), "fraction": float(value)}
                   for step, value in zip(steps, fractions)],
              "weight_vector_parameter_count": n_parameters,
              "weight_trajectory_pc1_pc2_variance_fraction": float(trajectory_variance[:2].sum()),
              "weight_trajectory_explained_variance_first_10": trajectory_variance[:10].tolist(),
              "weight_trajectory_coordinates":
                  [{"step": int(step), "pc1": float(x), "pc2": float(y),
                    "test_accuracy": float(np.interp(step, metric_step, test_accuracy))}
                   for step, (x, y) in zip(steps, trajectory[:, :2])]}
    (HERE / "pca_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("embedding_frequency_power_fraction_by_step",
                                     "weight_trajectory_coordinates")}, indent=2))


if __name__ == "__main__":
    main()
