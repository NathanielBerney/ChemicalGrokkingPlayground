#!/usr/bin/env python3
"""Animate graph Fourier diagnostics for the checkpointed QM9 runs.

Two fixed graph constructions are used. Molecular bond graphs provide a
Laplacian basis for atom activations, while a Morgan-fingerprint similarity
graph over test molecules provides a basis for scalar prediction residuals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch.nn import functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from animate_qm9 import BLUE, ORANGE, OUT, PURPLE, save_gif
from reproduce_qm9 import DATA_ROOT, ROOT, ExampleNet


NODE_BOUNDS = np.array([0, .5, 1, 1.5, 2.000001])
NODE_LABELS = ("0–0.5", "0.5–1", "1–1.5", "1.5–2")


def normalized_laplacian(weights):
    degree = weights.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("Graph has isolated vertices")
    inverse = 1 / np.sqrt(degree)
    matrix = np.eye(len(weights)) - inverse[:, None] * weights * inverse[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def connected_components(weights):
    labels = np.full(len(weights), -1, dtype=int)
    count = 0
    for start in range(len(weights)):
        if labels[start] >= 0:
            continue
        labels[start] = count
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in np.flatnonzero(weights[vertex] > 0):
                if labels[neighbor] < 0:
                    labels[neighbor] = count
                    stack.append(int(neighbor))
        count += 1
    return count, labels


def bond_bases(dataset):
    bases = []
    for graph in dataset:
        count = graph.num_nodes
        adjacency = np.zeros((count, count), dtype=np.float64)
        edges = graph.edge_index.cpu().numpy()
        adjacency[edges[0], edges[1]] = 1
        adjacency = np.maximum(adjacency, adjacency.T)
        eigenvalues, vectors = normalized_laplacian(adjacency)
        bands = np.digitize(eigenvalues, NODE_BOUNDS[1:], right=False)
        bands[eigenvalues < 1e-8] = -1  # Exclude the degree-weighted constant mode.
        bases.append((vectors.T, bands))
    return bases


def similarity_basis(dataset, test_indices, neighbors=8):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    unsanitized = 0
    for index in test_indices:
        smiles = dataset[int(index)].smiles
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            # A few stored QM9 SMILES fail modern RDKit valence checks. They
            # still have parseable connectivity, which suffices for this
            # explicitly exploratory fingerprint graph.
            molecule = Chem.MolFromSmiles(smiles, sanitize=False)
            if molecule is None:
                raise ValueError(f"Could not parse SMILES at QM9 index {index}")
            molecule.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(molecule)
            unsanitized += 1
        fingerprints.append(generator.GetFingerprint(molecule))
    count = len(fingerprints)
    similarity = np.eye(count)
    for i, fingerprint in enumerate(fingerprints):
        similarity[i] = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, -np.inf)
    weights = np.zeros_like(similarity)
    for i in range(count):
        closest = np.argsort(similarity[i])[-min(neighbors, count - 1):]
        weights[i, closest] = np.maximum(similarity[i, closest], 0) + 1e-3
    weights = np.maximum(weights, weights.T)
    bridges = 0
    components, labels = connected_components(weights)
    while components > 1:
        cross = labels[:, None] != labels[None, :]
        i, j = np.unravel_index(np.argmax(np.where(cross, similarity, -np.inf)),
                               similarity.shape)
        weights[i, j] = weights[j, i] = max(similarity[i, j], 0) + 1e-3
        bridges += 1
        components, labels = connected_components(weights)
    eigenvalues, vectors = normalized_laplacian(weights)
    return eigenvalues, vectors, bridges, unsanitized


def node_power(model, loader, bases, device):
    fractions = np.zeros((2, 4), dtype=np.float64)
    ac_ratio = np.zeros(2, dtype=np.float64)
    ac_rms = np.zeros(2, dtype=np.float64)
    valid = np.zeros(2, dtype=int)
    offset = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            first = F.relu(model.conv1(batch.x, batch.edge_index, batch.edge_attr))
            second = F.relu(model.conv2(first, batch.edge_index, batch.edge_attr))
            arrays = (first.cpu().numpy(), second.cpu().numpy())
            ptr = batch.ptr.cpu().numpy()
            for local in range(batch.num_graphs):
                vectors_t, bands = bases[offset + local]
                for layer, values in enumerate(arrays):
                    features = values[ptr[local]:ptr[local + 1]]
                    energy = np.sum((vectors_t @ features) ** 2, axis=1)
                    nonconstant = float(energy[bands >= 0].sum())
                    total = float(energy.sum())
                    if nonconstant > 0:
                        valid[layer] += 1
                        for band in range(4):
                            fractions[layer, band] += energy[bands == band].sum() / nonconstant
                    ac_ratio[layer] += nonconstant / max(total, 1e-30)
                    ac_rms[layer] += np.sqrt(nonconstant / features.size)
            offset += batch.num_graphs
    if offset != len(bases):
        raise ValueError("Node basis and dataset order differ")
    if np.any(valid == 0):
        raise ValueError("No nonconstant atom activations in a convolution layer")
    return fractions / valid[:, None], ac_ratio / offset, ac_rms / offset


def checkpoint_node_spectra(total_size, alpha, loader, bases, steps, device):
    directory = ROOT / "runs" / f"grokking_total{total_size}_alpha{alpha}_animation"
    model = ExampleNet().to(device)
    fractions, ac_ratio, ac_rms = [], [], []
    for step in steps:
        path = directory / "snapshots" / f"step_{int(step):06d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        shape, ratio, rms = node_power(model, loader, bases, device)
        fractions.append(shape)
        ac_ratio.append(ratio)
        ac_rms.append(rms)
    return np.array(fractions), np.array(ac_ratio), np.array(ac_rms)


def residual_spectrum(predictions, truth, vectors):
    residual = predictions - truth[None, :]
    coefficients = residual @ vectors
    groups = np.array_split(np.arange(len(truth)), 10)
    energy = np.array([np.sum(coefficients[:, group] ** 2, axis=1)
                       for group in groups]).T / len(truth)
    if not np.allclose(energy.sum(axis=1), np.mean(residual ** 2, axis=1), rtol=1e-7,
                       atol=1e-7):
        raise ValueError("Graph Fourier energy does not reconstruct test MSE")
    return energy


def loss_at_step(directory, steps):
    with (directory / "full_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metric_steps = np.array([float(row["step"]) for row in rows])
    mse = np.array([float(row["test_mse"]) for row in rows])
    return np.interp(steps, metric_steps, mse)


def plot_residual(index, total_size, steps, spectra, mse_limits, xscale="log"):
    step = int(steps[index])
    current = np.concatenate([spectra[alpha][index] for alpha in (1, 3)])
    current = current[current > 0]
    bar_limits = (max(1e-14, float(current.min()) * .5), float(current.max()) * 2)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=100,
                            gridspec_kw={"height_ratios": [1.1, 1]}, layout="constrained")
    for column, alpha in enumerate((1, 3)):
        values = spectra[alpha]
        ax = axes[0, column]
        ax.bar(np.arange(1, 11), np.maximum(values[index], 1e-14), color=BLUE if alpha == 1 else PURPLE)
        ax.set(xlim=(.4, 10.6), ylim=bar_limits, yscale="log",
               xticks=np.arange(1, 11), xlabel="Laplacian eigenmode decile (low → high)",
               ylabel="Contribution to test MSE",
               title=f"α={alpha} · test MSE {values[index].sum():.3g}")
        ax.grid(alpha=.2, axis="y", which="both")
        ax = axes[1, column]
        x = steps[:index + 1] + (1 if xscale == "log" else 0)
        series = ((values.sum(axis=1), ORANGE, "all modes"),
                  (values[:, :3].sum(axis=1), BLUE, "low 3 deciles"),
                  (values[:, -3:].sum(axis=1), PURPLE, "high 3 deciles"))
        for series_values, color, label in series:
            y = np.maximum(series_values[:index + 1], 1e-14)
            ax.plot(x, y, color=color, lw=1.5, label=label)
            ax.scatter(x[-1], y[-1], color=color, s=20, zorder=3)
        ax.set(xscale=xscale, yscale="log",
               xlim=(1 if xscale == "log" else 0, steps[-1] * 1.07),
               ylim=mse_limits,
               xlabel="Optimizer step + 1" if xscale == "log" else "Optimizer step",
               ylabel="Test MSE contribution")
        ax.grid(alpha=.2, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"QM9 test-error graph Fourier spectrum · N={total_size} · step {step:,}\n"
                 "Morgan/Tanimoto 8-neighbor graph · top axes adjust per frame; bottom axes stay fixed",
                 fontsize=12)
    return fig


def plot_nodes(index, total_size, steps, fractions, ratios, rms, test_mse):
    step = int(steps[index])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.1), dpi=100, layout="constrained")
    for column, alpha in enumerate((1, 3)):
        for layer in range(2):
            ax = axes[layer, column]
            values = fractions[alpha][index, layer]
            ax.bar(np.arange(4), values, color=BLUE if alpha == 1 else PURPLE)
            ax.set(xlim=(-.5, 3.5), ylim=(0, 1), xticks=np.arange(4), xticklabels=NODE_LABELS,
                   xlabel="Bond-graph normalized Laplacian eigenvalue λ",
                   ylabel="Share of nonconstant activation power",
                   title=f"α={alpha} · conv{layer + 1} · test MSE {test_mse[alpha][index]:.3g}\n"
                         f"nonconstant / total {ratios[alpha][index, layer]:.1%} · AC RMS {rms[alpha][index, layer]:.3g}")
            ax.grid(alpha=.2, axis="y")
    fig.suptitle(f"QM9 atom-activation graph Fourier spectrum · N={total_size} · step {step:,}\n"
                 "Each molecule uses its own bond graph; mean of per-molecule power shares after removing λ=0",
                 fontsize=12)
    return fig


def make_one(total_size, fps, device):
    path = OUT / f"analysis_total{total_size}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run animate_qm9.py before animate_spectra.py: {path}")
    analysis = np.load(path)
    steps = analysis["steps"]
    dataset = QM9(root=str(DATA_ROOT))[:total_size]
    mask = analysis["train_mask"]
    test_indices = np.flatnonzero(~mask)
    eigenvalues, vectors, bridges, unsanitized = similarity_basis(dataset, test_indices)
    truth = analysis["truth"][test_indices]
    spectra = {alpha: residual_spectrum(analysis[f"alpha{alpha}_prediction"][:, test_indices],
                                        truth, vectors) for alpha in (1, 3)}
    bases = bond_bases(dataset)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    fractions, ratios, rms, test_mse = {}, {}, {}, {}
    for alpha in (1, 3):
        fractions[alpha], ratios[alpha], rms[alpha] = checkpoint_node_spectra(
            total_size, alpha, loader, bases, steps, device)
        directory = ROOT / "runs" / f"grokking_total{total_size}_alpha{alpha}_animation"
        test_mse[alpha] = loss_at_step(directory, steps)
        if not np.allclose(test_mse[alpha][[0, -1]], spectra[alpha][[0, -1]].sum(axis=1),
                           rtol=1e-4, atol=1e-3):
            raise ValueError(f"Spectral test MSE disagrees with training metrics for α={alpha}")
        if not np.allclose(fractions[alpha].sum(axis=2), 1, atol=1e-5):
            raise ValueError(f"Node spectral bands do not sum to one for α={alpha}")
    all_energy = np.concatenate([spectra[a].reshape(-1) for a in (1, 3)])
    positive = all_energy[all_energy > 0]
    mse_limits = (max(1e-12, float(np.percentile(positive, .5)) * .2),
                  float(positive.max()) * 2)
    np.savez_compressed(OUT / f"graph_spectra_total{total_size}.npz", steps=steps,
                        similarity_eigenvalues=eigenvalues, test_indices=test_indices,
                        node_band_bounds=NODE_BOUNDS, alpha1_test_error_energy=spectra[1],
                        alpha3_test_error_energy=spectra[3],
                        alpha1_node_power_fraction=fractions[1],
                        alpha3_node_power_fraction=fractions[3],
                        alpha1_node_ac_fraction=ratios[1], alpha3_node_ac_fraction=ratios[3],
                        alpha1_node_ac_rms=rms[1], alpha3_node_ac_rms=rms[3])
    save_gif(OUT / f"residual_spectrum_total{total_size}.gif", len(steps), fps,
             lambda i: plot_residual(i, total_size, steps, spectra, mse_limits))
    save_gif(OUT / f"node_spectrum_total{total_size}.gif", len(steps), fps,
             lambda i: plot_nodes(i, total_size, steps, fractions, ratios, rms, test_mse))
    return {"total_size": total_size, "test_similarity_graph_vertices": len(test_indices),
            "similarity_graph_bridges": bridges, "snapshot_count": len(steps),
            "unsanitized_smiles_fallbacks": unsanitized,
            "node_layers": ["conv1", "conv2"], "node_band_bounds": NODE_BOUNDS.tolist()}


def render_linear_residual(total_size, fps):
    with np.load(OUT / f"graph_spectra_total{total_size}.npz") as cached:
        steps = cached["steps"].copy()
        spectra = {alpha: cached[f"alpha{alpha}_test_error_energy"].copy()
                   for alpha in (1, 3)}
    all_energy = np.concatenate([spectra[a].reshape(-1) for a in (1, 3)])
    positive = all_energy[all_energy > 0]
    mse_limits = (max(1e-12, float(np.percentile(positive, .5)) * .2),
                  float(positive.max()) * 2)
    save_gif(OUT / f"residual_spectrum_total{total_size}_linear_x.gif", len(steps), fps,
             lambda i: plot_residual(i, total_size, steps, spectra, mse_limits,
                                     xscale="linear"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 400])
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--linear-x-only", action="store_true",
                        help="Render linear-step residual GIFs from cached spectra")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    if args.linear_x_only:
        for size in args.sizes:
            render_linear_residual(size, args.fps)
        return
    summary_path = OUT / "spectral_summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    summary = {int(row["total_size"]): row for row in existing}
    for size in args.sizes:
        summary[size] = make_one(size, args.fps, torch.device(args.device))
    summary = [summary[size] for size in sorted(summary)]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
