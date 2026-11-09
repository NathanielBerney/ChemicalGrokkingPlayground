#!/usr/bin/env python3
"""Quantify pooled-embedding geometry and inspect actual Omnigrok matrices.

PCA is fit on the archived training 200 molecules at each checkpoint. The
unchanged readout is evaluated after retaining or removing PCA directions.
The next 1,000 QM9 molecules are used only for external diagnostics.
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
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from ablate_qm9 import encode, head, head_parameters, ridge_predict, train_only_probe
from probe_chemistry_qm9 import (
    auc, chemical_labels, composition_matched_pairs, matched_auc,
)
from reproduce_qm9 import DATA_ROOT, ROOT, TARGET, ExampleNet


OUT = ROOT / "animations" / "structure_map" / "total400"
STEPS = (0, 10000, 20000, 40000, 87500)
COLORS = {1: "#2563eb", 3: "#7c3aed"}
BONDTYPES = ("single", "double", "triple")
QM9_PROPERTIES = (
    "Dipole moment", "Polarizability", "HOMO energy", "LUMO energy",
    "HOMO-LUMO gap", "Electronic spatial extent", "Zero-point energy",
    "Internal energy (0 K)", "Internal energy (298 K)", "Enthalpy",
    "Free energy", "Heat capacity", "Atomization energy (0 K)",
    "Atomization energy (298 K)", "Atomization enthalpy",
    "Atomization free energy", "Rotational constant A",
    "Rotational constant B", "Rotational constant C",
)


def load_model(alpha, step, device, total_size=400):
    path = (ROOT / "runs" / f"grokking_total{total_size}_alpha{alpha}_animation"
            / "snapshots" / f"step_{step:06d}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint["step"]) != step:
        raise ValueError(f"Checkpoint mismatch: {path}")
    model = ExampleNet().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model


def pca_train(pooled, train):
    mean = pooled[train].mean(axis=0)
    centered = pooled - mean
    _, singular, directions = np.linalg.svd(centered[train], full_matrices=False)
    variance = singular ** 2
    fractions = variance / variance.sum()
    scores = centered @ directions.T
    return mean, directions, scores, fractions


def squared_error(prediction, truth, indices):
    return float(np.mean((prediction[indices] - truth[indices]) ** 2))


def linear_cka(first, second):
    x = first - first.mean(axis=0)
    y = second - second.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / denominator)


def residualize_on_training(features, covariates, train):
    center = covariates[train].mean(axis=0)
    scale = np.maximum(covariates[train].std(axis=0), 1e-6)
    design = np.column_stack((np.ones(len(features)), (covariates - center) / scale))
    coefficients = np.linalg.lstsq(design[train], features[train], rcond=None)[0]
    return features - design @ coefficients


def bootstrap_delta(original, modified, truth, indices, seed=2026):
    per_molecule = (modified[indices] - truth[indices]) ** 2 - (
        original[indices] - truth[indices]) ** 2
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(indices), size=(2000, len(indices)))
    means = per_molecule[sampled].mean(axis=1)
    return {"mean_mse_increase": float(per_molecule.mean()),
            "bootstrap_95_percent_interval": np.percentile(means, [2.5, 97.5]).tolist(),
            "molecules": int(len(indices))}


def analyze_state(alpha, step, loader, device, truth, elements, train, test,
                  cached_predictions):
    model = load_model(alpha, step, device)
    pooled = encode(model, loader, device)
    parameters = head_parameters(model)
    hidden, prediction = head(pooled, parameters)
    expected = cached_predictions[alpha][np.flatnonzero(cached_predictions["steps"] == step)[0]]
    max_difference = float(np.max(np.abs(prediction - expected)))
    if max_difference > max(1e-3, 1e-6 * float(np.max(np.abs(expected)))):
        raise ValueError(f"Prediction mismatch for alpha={alpha}, step={step}")
    mean, directions, scores, fractions = pca_train(pooled, train)
    baseline = squared_error(prediction, truth, test)
    retained = []
    for count in (1, 2, 3, 4, 8, 16):
        reconstructed = mean + scores[:, :count] @ directions[:count]
        reconstructed_prediction = head(reconstructed, parameters)[1]
        clamped_prediction = head(np.maximum(reconstructed, 0), parameters)[1]
        retained.append({
            "components": count,
            "variance_fraction": float(fractions[:count].sum()),
            "test_mse": squared_error(reconstructed_prediction, truth, test),
            "test_mse_nonnegative_clamped": squared_error(clamped_prediction, truth, test),
            "fraction_negative_reconstructed_test_entries": float(
                np.mean(reconstructed[test] < -1e-8)),
            "rmse_from_original_prediction": float(np.sqrt(np.mean(
                (reconstructed_prediction[test] - prediction[test]) ** 2))),
        })
    removed = []
    for component in range(8):
        without = pooled - scores[:, component, None] * directions[component]
        changed = head(without, parameters)[1]
        removed.append({
            "component": component + 1,
            "variance_fraction": float(fractions[component]),
            "test_mse": squared_error(changed, truth, test),
            "test_mse_ratio": squared_error(changed, truth, test) / baseline,
            **({"paired_mse_increase": bootstrap_delta(prediction, changed, truth, test)}
               if component == 2 else {}),
        })
    target_residual = truth - ridge_predict(elements, truth, train, 1.0)
    atom_count = elements.sum(axis=1)
    pc_correlations = []
    for component in range(3):
        pc = scores[:, component]
        pc_residual = pc - ridge_predict(elements, pc, train, 1.0)
        pc_correlations.append({
            "component": component + 1,
            "pearson_with_test_target": float(np.corrcoef(pc[test], truth[test])[0, 1]),
            "pearson_with_test_atom_count": float(np.corrcoef(pc[test], atom_count[test])[0, 1]),
            "pearson_with_test_target_after_element_count_control": float(
                np.corrcoef(pc_residual[test], target_residual[test])[0, 1]),
        })
    preactivation = pooled @ parameters[0].T + parameters[1]
    local_sensitivity = (np.asarray(preactivation > 0, dtype=np.float64)
                         * parameters[2].ravel()) @ parameters[0]
    target_variance = float(np.var(truth[test]))
    linear_probes = {}
    for count in (1, 2, 3, 16):
        result = train_only_probe(scores[:, :count], truth, train, test)
        linear_probes[str(count)] = {"test_mse": result["test_mse"],
                                     "test_r2": 1 - result["test_mse"] / target_variance,
                                     "selected_penalty": result["selected_penalty"]}
    report = {
        "alpha": alpha, "step": step, "baseline_test_mse": baseline,
        "prediction_max_difference_from_animation": max_difference,
        "pca_variance_fraction": fractions.tolist(),
        "pc1_plus_pc2_variance_fraction": float(fractions[:2].sum()),
        "participation_ratio_effective_rank": float(1 / np.sum(fractions ** 2)),
        "active_pooled_channels": int(np.any(pooled > 1e-8, axis=0).sum()),
        "retained_pc_tests": retained, "removed_pc_tests": removed,
        "pc_correlations": pc_correlations, "train_only_linear_target_probes": linear_probes,
    }
    arrays = {"pooled": pooled, "hidden": hidden, "prediction": prediction,
              "scores": scores, "directions": directions,
              "sensitivity": local_sensitivity, "model": model,
              "parameters": parameters}
    return report, arrays


def weight_trajectory(alpha):
    directory = ROOT / "runs" / f"grokking_total400_alpha{alpha}_animation" / "snapshots"
    files = sorted(directory.glob("step_*.pt"))
    vectors, steps = [], []
    blocks = None
    for path in files:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        if blocks is None:
            offset = 0
            blocks = {}
            for name, value in state.items():
                blocks[name] = slice(offset, offset + value.numel())
                offset += value.numel()
        vectors.append(np.concatenate([value.numpy().ravel() for value in state.values()]))
        steps.append(int(checkpoint["step"]))
    vectors = np.array(vectors, dtype=np.float64)
    steps = np.array(steps)
    centered = vectors - vectors.mean(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered @ centered.T)[::-1], 0)
    fractions = eigenvalues / eigenvalues.sum()
    block_shares = {}
    for prefix in ("conv1", "conv2", "fc_1", "out"):
        indices = np.concatenate([np.arange(s.start, s.stop) for name, s in blocks.items()
                                  if name.startswith(prefix)])
        block_shares[prefix] = float(np.sum(centered[:, indices] ** 2)
                                     / np.sum(centered ** 2))
    later = centered[steps >= 10000]
    later = later - later.mean(axis=0)
    later_eigenvalues = np.maximum(np.linalg.eigvalsh(later @ later.T)[::-1], 0)
    return {"steps": steps.tolist(), "pca_variance_fraction": fractions.tolist(),
            "pc1_plus_pc2": float(fractions[:2].sum()),
            "pc1_plus_pc2_after_step10000": float(
                later_eigenvalues[:2].sum() / later_eigenvalues.sum()),
            "block_share_of_trajectory_variance": block_shares,
            "parameter_count": int(vectors.shape[1])}


def draw_pca_fidelity(states, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), layout="constrained")
    for alpha in (1, 3):
        chosen = states[(alpha, 40000)]
        fractions = np.array(chosen["pca_variance_fraction"])
        axes[0, 0].plot(range(1, 9), np.cumsum(fractions)[:8], "o-",
                        color=COLORS[alpha], label=f"α={alpha}")
        retained = chosen["retained_pc_tests"]
        axes[0, 1].plot([row["components"] for row in retained if row["components"] <= 8],
                        [row["test_mse"] for row in retained if row["components"] <= 8],
                        "o-", color=COLORS[alpha], label=f"α={alpha}")
        axes[0, 1].axhline(chosen["baseline_test_mse"], color=COLORS[alpha],
                           ls="--", alpha=.6)
        steps = [step for step in STEPS]
        axes[1, 0].plot(steps, [states[(alpha, step)]["pc1_plus_pc2_variance_fraction"]
                                  for step in steps], "o-", color=COLORS[alpha],
                        label=f"α={alpha}")
        axes[1, 1].plot(steps, [states[(alpha, step)]["active_pooled_channels"]
                                  for step in steps], "o-", color=COLORS[alpha],
                        label=f"α={alpha}")
    axes[0, 0].set(xlabel="Number of PCs", ylabel="Cumulative training-set variance",
                   title="How much of the embedding varies?")
    axes[0, 1].set(xlabel="PCs retained in unchanged readout",
                   ylabel="Held-out polarizability MSE",
                   title="Does the actual model use those PCs?")
    axes[1, 0].set(xlabel="Optimizer step", ylabel="Variance in PC1+PC2",
                   title="Two-dimensionality during training")
    axes[1, 1].set(xlabel="Optimizer step", ylabel="Active pooled channels / 16",
                   title="ReLU channel sparsity during training")
    for ax in axes.flat:
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    axes[0, 0].set_xticks(range(1, 9))
    axes[0, 1].set_xticks((1, 2, 3, 4, 8))
    axes[1, 1].set_ylim(0, 16)
    fig.suptitle("N=400: PCA variance and frozen-head fidelity (linear axes)", fontsize=15)
    fig.savefig(outdir / "pca_variance_vs_function.png", dpi=170)
    plt.close(fig)


def draw_geometry(arrays, truth, elements, concepts, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8), layout="constrained")
    for row, alpha in enumerate((1, 3)):
        scores = arrays[(alpha, 40000)]["scores"]
        values = (truth, elements.sum(axis=1), concepts["nitrile"])
        labels = ("True polarizability", "Total atom count", "Nitrile present")
        for column, (value, label) in enumerate(zip(values, labels)):
            ax = axes[row, column]
            scatter = ax.scatter(scores[:, 0], scores[:, 1] if column < 2 else scores[:, 2],
                                 c=value, cmap="viridis", s=19, alpha=.78,
                                 edgecolors="none")
            ax.set(xlabel="PC1", ylabel="PC2" if column < 2 else "PC3",
                   title=f"α={alpha} · {label}")
            ax.grid(alpha=.2)
            fig.colorbar(scatter, ax=ax, shrink=.73)
    fig.suptitle("Step 40,000: train-fitted axes, all 400 molecules", fontsize=15)
    fig.savefig(outdir / "embedding_chemistry_axes.png", dpi=170)
    plt.close(fig)


def draw_weight_matrices(arrays, outdir):
    keys = ("conv1.lin.weight", "conv2.lin.weight", "fc_1.weight", "out.weight")
    titles = ("Conv1 root 32×11", "Conv2 root 16×32",
              "Pooled → hidden 32×16", "Hidden → scalar 1×32")
    states = {alpha: arrays[(alpha, 40000)]["model"].state_dict() for alpha in (1, 3)}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), layout="constrained")
    for column, (key, title) in enumerate(zip(keys, titles)):
        bound = max(float(states[alpha][key].abs().max()) for alpha in (1, 3))
        for row, alpha in enumerate((1, 3)):
            matrix = states[alpha][key].detach().cpu().numpy()
            ax = axes[row, column]
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-bound, vmax=bound,
                           aspect="auto", interpolation="nearest")
            ax.set(xlabel="Input channel", ylabel="Output channel",
                   title=f"α={alpha} · {title}")
            fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle("Actual learned parameter matrices at step 40,000 (signed linear color)", fontsize=15)
    fig.savefig(outdir / "learned_weight_matrices_step40000.png", dpi=170)
    plt.close(fig)


def draw_edge_filters(arrays, alpha, outdir):
    model = arrays[(alpha, 40000)]["model"]
    bond_vectors = torch.eye(4, device=next(model.parameters()).device)[:3]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    with torch.inference_mode():
        filters = (
            model.conv1.nn(bond_vectors).reshape(3, 11, 32).permute(0, 2, 1).cpu().numpy(),
            model.conv2.nn(bond_vectors).reshape(3, 32, 16).permute(0, 2, 1).cpu().numpy(),
        )
    for row, matrices in enumerate(filters):
        bound = float(np.max(np.abs(matrices)))
        for column, bond in enumerate(BONDTYPES):
            ax = axes[row, column]
            im = ax.imshow(matrices[column], cmap="coolwarm", vmin=-bound, vmax=bound,
                           aspect="auto", interpolation="nearest")
            ax.set(xlabel="Source atom channel", ylabel="Destination channel",
                   title=f"Conv{row+1}: {bond} bond")
            fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle(f"α={alpha}, step 40,000: edge-conditioned convolution filters", fontsize=15)
    fig.savefig(outdir / f"edge_filters_alpha{alpha}_step40000.png", dpi=170)
    plt.close(fig)


def draw_activations(arrays, truth, outdir):
    order = np.argsort(truth)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), layout="constrained")
    for row, alpha in enumerate((1, 3)):
        values = arrays[(alpha, 40000)]
        for column, (key, title, cmap) in enumerate((
            ("pooled", "Pooled graph activations (400×16)", "viridis"),
            ("hidden", "Readout activations (400×32)", "viridis"),
            ("sensitivity", "Local d(output)/d(pooled) (400×16)", "coolwarm"),
        )):
            matrix = values[key][order]
            ax = axes[row, column]
            bound = float(np.percentile(np.abs(matrix), 99))
            im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest",
                           vmin=-bound if key == "sensitivity" else 0, vmax=bound)
            ax.set(xlabel="Channel index", ylabel="Molecules sorted by true target",
                   title=f"α={alpha} · {title}")
            fig.colorbar(im, ax=ax, shrink=.67)
        ax = axes[row, 3]
        ax.plot(truth[order], np.arange(len(order)), color="#111827", lw=1.2, label="True")
        ax.plot(values["prediction"][order], np.arange(len(order)), color=COLORS[alpha],
                lw=1.1, label="Predicted")
        ax.set(xlabel="Polarizability", ylabel="Molecules sorted by true target",
               title=f"α={alpha} · scalar outputs")
        ax.invert_yaxis()
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Step 40,000: activations and scalar regression output; heatmap color clipped at 99th percentile",
                 fontsize=14)
    fig.savefig(outdir / "activation_and_scalar_output_matrices_step40000.png", dpi=160)
    plt.close(fig)


def draw_scalar_output_matrix(arrays, truth, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                            gridspec_kw={"height_ratios": [.35, 2.2]},
                            layout="constrained")
    weight_bound = max(float(np.max(np.abs(arrays[(alpha, 40000)]["parameters"][2])))
                       for alpha in (1, 3))
    for column, alpha in enumerate((1, 3)):
        ax = axes[0, column]
        weight = arrays[(alpha, 40000)]["parameters"][2]
        bias = float(arrays[(alpha, 40000)]["parameters"][3][0])
        im = ax.imshow(weight, cmap="coolwarm", vmin=-weight_bound, vmax=weight_bound,
                       aspect="auto", interpolation="nearest")
        ax.set(yticks=[], xlabel="Hidden readout channel (0–31)",
               title=f"α={alpha}: final weight row, shape 1×32; bias {bias:.3g}")
        fig.colorbar(im, ax=ax, shrink=.8)
    order = np.argsort(truth)
    predictions = [arrays[(alpha, 40000)]["prediction"] for alpha in (1, 3)]
    absolute = np.column_stack((truth, *predictions))[order]
    ax = axes[1, 0]
    im = ax.imshow(absolute, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set(xticks=range(3), xticklabels=("True", "α=1", "α=3"),
           ylabel="Molecules sorted by true polarizability",
           title="Scalar values: 400 molecules × 3 columns")
    fig.colorbar(im, ax=ax, label="Polarizability", shrink=.75)
    residual = np.column_stack((predictions[0] - truth, predictions[1] - truth))[order]
    bound = max(.1, float(np.percentile(np.abs(residual), 99)))
    ax = axes[1, 1]
    im = ax.imshow(residual, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound,
                   interpolation="nearest")
    ax.set(xticks=range(2), xticklabels=("α=1", "α=3"),
           ylabel="Same sorted molecules", title="Prediction − truth (color clipped at 99th percentile)")
    fig.colorbar(im, ax=ax, label="Residual polarizability", shrink=.75)
    fig.suptitle("Step 40,000: scalar regression readout; this model has no classification logits",
                 fontsize=14)
    fig.savefig(outdir / "scalar_output_not_logits_step40000.png", dpi=170)
    plt.close(fig)


def draw_weight_trajectory(trajectory, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for alpha in (1, 3):
        fractions = trajectory[str(alpha)]["pca_variance_fraction"]
        axes[0].plot(range(1, 11), np.cumsum(fractions)[:10], "o-",
                     color=COLORS[alpha], label=f"α={alpha}")
    axes[0].set(xlabel="Weight-trajectory PCs", ylabel="Cumulative trajectory variance",
                title="PCA across 89 checkpoint vectors")
    axes[0].legend()
    labels = ("conv1", "conv2", "fc_1", "out")
    bottoms = np.zeros(2)
    for label, color in zip(labels, ("#2563eb", "#7c3aed", "#f59e0b", "#ef4444")):
        values = [trajectory[str(alpha)]["block_share_of_trajectory_variance"][label]
                  for alpha in (1, 3)]
        axes[1].bar(("α=1", "α=3"), values, bottom=bottoms, color=color, label=label)
        bottoms += values
    axes[1].set(ylabel="Share of total weight-trajectory variance",
                title="Which parameter blocks move?")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2, axis="y")
        ax.set_axisbelow(True)
    fig.savefig(outdir / "weight_trajectory_structure.png", dpi=170)
    plt.close(fig)


def draw_external_pc_concepts(matrix, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, alpha in zip(axes, (1, 3)):
        data = np.array([[matrix[str(alpha)][str(component)][concept]["matched_auc"]
                          for concept in matrix[str(alpha)]["concepts"]]
                         for component in range(1, 9)])
        im = ax.imshow(data, cmap="viridis", vmin=.3, vmax=1, aspect="auto")
        ax.set(xticks=np.arange(len(matrix[str(alpha)]["concepts"])),
               xticklabels=matrix[str(alpha)]["concepts"],
               yticks=np.arange(8), yticklabels=[f"PC{k}" for k in range(1, 9)],
               title=f"α={alpha}: external matched-pair AUC")
        for row in range(8):
            for column in range(data.shape[1]):
                ax.text(column, row, f"{data[row, column]:.2f}", ha="center",
                        va="center", fontsize=8,
                        color="white" if data[row, column] < .62 else "black")
    fig.colorbar(im, ax=axes, label="AUC after matching H/C/N/O/F counts", shrink=.8)
    fig.suptitle("Single PCA axes vs chemical concepts on 1,000 extra QM9 molecules",
                 fontsize=14)
    fig.savefig(outdir / "external_pc_chemical_concepts.png", dpi=170)
    plt.close(fig)


def draw_pc3_validation(external, outdir):
    data = external["arrays"]
    nitrile = data["nitrile"]
    oriented = data["pc3_oriented"]
    truth = data["truth"]
    original = data["original"]
    without = data["without_pc3"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    axes[0].boxplot([oriented[nitrile == 0], oriented[nitrile == 1]],
                    tick_labels=["Other", "Nitrile"], showfliers=False)
    axes[0].set(ylabel="PC3 score, sign chosen on training half",
                title="Frozen PC3 separates nitrile structures")
    subset = nitrile == 1
    low, high = float(truth[subset].min()), float(truth[subset].max())
    axes[1].plot([low, high], [low, high], ls="--", color="#111827")
    axes[1].scatter(truth[subset], original[subset], s=18, alpha=.6,
                    color="#2563eb", label="Original")
    axes[1].scatter(truth[subset], without[subset], s=18, alpha=.6,
                    color="#ef4444", label="Remove PC3")
    axes[1].set(xlabel="True polarizability", ylabel="Predicted polarizability",
                title="External nitrile molecules (n=111)")
    axes[1].legend(fontsize=8)
    groups = ((nitrile == 1, "Nitrile"), (nitrile == 0, "Other"))
    original_mse = [np.mean((original[mask] - truth[mask]) ** 2) for mask, _ in groups]
    removed_mse = [np.mean((without[mask] - truth[mask]) ** 2) for mask, _ in groups]
    x = np.arange(2)
    axes[2].bar(x - .18, original_mse, .36, color="#2563eb", label="Original")
    axes[2].bar(x + .18, removed_mse, .36, color="#ef4444", label="Remove PC3")
    axes[2].set(xticks=x, xticklabels=[label for _, label in groups],
                ylabel="External polarizability MSE",
                title="PC3 matters beyond nitrile too")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("α=3, step 40,000: PC3 tested beyond the archived 400 molecules",
                 fontsize=14)
    fig.savefig(outdir / "pc3_nitrile_external_validation.png", dpi=170)
    plt.close(fig)


def draw_nitrile_stages(stages, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = ("#2563eb", "#7c3aed", "#ea580c")
    for ax, alpha in zip(axes, (1, 3)):
        for component, color in zip((1, 2, 3), colors):
            ax.plot(STEPS, [stages[str(alpha)][str(step)][str(component)]["matched_auc"]
                            for step in STEPS], "o-", color=color, label=f"PC{component}")
        ax.set(xlabel="Optimizer step", ylabel="External composition-matched nitrile AUC",
               ylim=(0, 1.05), title=f"α={alpha}")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("First three PCs are refitted independently at each checkpoint", fontsize=13)
    fig.savefig(outdir / "nitrile_pc_axes_over_training.png", dpi=170)
    plt.close(fig)


def draw_matched_molecule_gallery(dataset, elements, external, outdir):
    from rdkit import Chem
    from rdkit.Chem import Draw

    values = external["arrays"]
    pairs = composition_matched_pairs(
        values["nitrile"].astype(float), elements, np.arange(len(dataset)))
    differences = np.abs(values["truth"][pairs[:, 0]] - values["truth"][pairs[:, 1]])
    selected = []
    used_formulas = set()
    for pair in pairs[np.argsort(differences)]:
        formula = tuple(elements[pair[0]].astype(int))
        if formula in used_formulas:
            continue
        molecules = [Chem.MolFromSmiles(dataset[int(index)].smiles) for index in pair]
        if any(molecule is None for molecule in molecules):
            continue
        selected.append((pair, molecules, formula))
        used_formulas.add(formula)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        raise ValueError("Could not find three drawable composition-matched pairs")
    query = Chem.MolFromSmarts("[CX2]#N")
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    for row, (pair, molecules, formula) in enumerate(selected):
        formula_text = "".join(f"{symbol}{count}" for symbol, count in
                               zip(("H", "C", "N", "O", "F"), formula) if count)
        for column, (index, molecule) in enumerate(zip(pair, molecules)):
            index = int(index)
            image = Draw.MolToImage(molecule, size=(440, 260),
                                    highlightAtoms=list(molecule.GetSubstructMatch(query)))
            ax = axes[row, column]
            ax.imshow(image)
            ax.set(xticks=[], yticks=[],
                   title=f"QM9 #{index + 400} · {formula_text} · "
                         f"{'nitrile' if values['nitrile'][index] else 'other structure'}",
                   xlabel=f"True {values['truth'][index]:.2f}  |  "
                          f"Model {values['original'][index]:.2f}  |  "
                          f"Remove PC3 {values['without_pc3'][index]:.2f}\n"
                          f"PC3 score {values['pc3_oriented'][index]:.1f}")
    fig.suptitle("External molecules matched by elemental formula and nearly equal true polarizability\n"
                 "Nitrile atoms highlighted; examples are illustrative, not a statistical test",
                 fontsize=14)
    fig.savefig(outdir / "composition_matched_nitrile_examples.png", dpi=170)
    plt.close(fig)


def draw_nitrile_matching_controls(records, outdir):
    fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
    x = np.arange(len(records))
    ax.plot(x, [row["pair_auc"] for row in records], "o-", color="#7c3aed",
            label="Pair-weighted AUC")
    ax.plot(x, [row["group_mean_auc"] for row in records], "s--", color="#2563eb",
            label="Mean formula-group AUC")
    for index, row in enumerate(records):
        ax.text(index, row["pair_auc"] + .015, f"{row['pairs']} pairs",
                ha="center", fontsize=8)
    ax.axhline(.5, color="#64748b", ls=":")
    ax.set(xticks=x,
           xticklabels=("Elements", "+ ring", "+ carbonyl", "+ hydroxyl", "+ amine"),
           ylim=(.42, 1.08), ylabel="External PC3 nitrile AUC",
           xlabel="Matched attributes (cumulative)",
           title="α=3 PC3: nitrile separation after stricter structural matching")
    ax.grid(alpha=.2, axis="y")
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(outdir / "pc3_nitrile_matching_controls.png", dpi=170)
    plt.close(fig)


def analyze_n100(device):
    dataset = QM9(root=str(DATA_ROOT))[:100]
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with np.load(ROOT / "animations" / "analysis_total100.npz") as cached:
        train = np.flatnonzero(cached["train_mask"])
        test = np.flatnonzero(~cached["train_mask"])
        truth = cached["truth"].copy()
    results = []
    for alpha in (1, 3):
        for step in (40000, 100000):
            model = load_model(alpha, step, device, total_size=100)
            pooled = encode(model, loader, device)
            parameters = head_parameters(model)
            mean, directions, scores, fractions = pca_train(pooled, train)
            baseline = squared_error(head(pooled, parameters)[1], truth, test)
            retained = []
            for count in (1, 2, 3, 4, 8, 16):
                reconstructed = mean + scores[:, :count] @ directions[:count]
                retained.append({"components": count,
                                 "test_mse": squared_error(
                                     head(reconstructed, parameters)[1], truth, test)})
            results.append({"alpha": alpha, "step": step,
                            "pc1_variance_fraction": float(fractions[0]),
                            "pc1_plus_pc2_variance_fraction": float(fractions[:2].sum()),
                            "active_pooled_channels": int(np.any(pooled > 1e-8, axis=0).sum()),
                            "baseline_test_mse": baseline,
                            "retained_pc_tests": retained})
    return results


def draw_n100_comparison(results, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), layout="constrained")
    for alpha in (1, 3):
        state = next(row for row in results if row["alpha"] == alpha and row["step"] == 100000)
        x = [record["components"] for record in state["retained_pc_tests"] if record["components"] <= 8]
        y = [record["test_mse"] for record in state["retained_pc_tests"] if record["components"] <= 8]
        axes[0].plot(x, y, "o-", color=COLORS[alpha], label=f"α={alpha}")
        axes[0].axhline(state["baseline_test_mse"], color=COLORS[alpha], ls="--", alpha=.6)
        axes[1].bar(alpha + (-.18 if alpha == 1 else .18),
                    state["pc1_plus_pc2_variance_fraction"], width=.36,
                    color=COLORS[alpha], label=f"α={alpha}")
    axes[0].set(xlabel="PCs retained in unchanged head", ylabel="Held-out MSE",
                title="N=100 final-checkpoint fidelity", xticks=(1, 2, 3, 4, 8))
    axes[0].legend(fontsize=8)
    axes[1].set(xticks=(.82, 3.18), xticklabels=("α=1", "α=3"), ylim=(.95, 1.005),
                ylabel="Variance in PC1+PC2", title="N=100 final embedding")
    for ax in axes:
        ax.grid(alpha=.2, axis="y")
    fig.suptitle("N=100 uses fewer directions than N=400; linear axes", fontsize=14)
    fig.savefig(outdir / "n100_pca_fidelity_comparison.png", dpi=170)
    plt.close(fig)


def analyze_property_transfer(dataset, elements, external_elements, arrays,
                              external_loader, device, train):
    """Fit each probe on the archived train split; score the outside 1,000."""
    targets = np.stack([graph.y.numpy().ravel() for graph in dataset])
    covariates = np.vstack((elements, external_elements))
    outside = np.arange(400, len(dataset))
    features = {"element_counts": covariates}
    for alpha in (1, 3):
        for step in (0, 40000):
            state = arrays[(alpha, step)]
            external_pooled = encode(state["model"], external_loader, device)
            pooled = np.vstack((state["pooled"], external_pooled))
            key = f"alpha{alpha}_step{step}"
            features[f"{key}_pooled"] = pooled
            features[f"{key}_elements_plus_pooled"] = np.column_stack(
                (covariates, pooled))
            if alpha == 3 and step == 40000:
                scores = np.vstack((
                    state["scores"],
                    (external_pooled - state["pooled"][train].mean(axis=0))
                    @ state["directions"].T))
                for count in (2, 3):
                    features[f"{key}_pc{count}"] = scores[:, :count]
                    features[f"{key}_elements_plus_pc{count}"] = np.column_stack(
                        (covariates, scores[:, :count]))
    report = {}
    for index, name in enumerate(QM9_PROPERTIES):
        target = targets[:, index]
        reference = float(np.mean((target[outside] - target[train].mean()) ** 2))
        probes = {}
        contrast_predictions = {}
        for feature_name, matrix in features.items():
            result = train_only_probe(matrix, target, train, outside)
            probes[feature_name] = {
                "external_mse": result["test_mse"],
                "score_relative_to_training_mean": 1 - result["test_mse"] / reference,
                "train_cv_mse": result["train_cv_mse"],
                "selected_penalty": result["selected_penalty"],
            }
            if feature_name in ("alpha3_step0_elements_plus_pooled",
                                "alpha3_step40000_elements_plus_pooled"):
                contrast_predictions[feature_name] = ridge_predict(
                    matrix, target, train, result["selected_penalty"])[outside]
        initial = contrast_predictions["alpha3_step0_elements_plus_pooled"]
        trained = contrast_predictions["alpha3_step40000_elements_plus_pooled"]
        score_gain_per_molecule = (
            (initial - target[outside]) ** 2 -
            (trained - target[outside]) ** 2) / reference
        rng = np.random.default_rng(2026 + index)
        sampled = rng.integers(
            0, len(outside), size=(2000, len(outside)))
        score_gain_interval = np.percentile(
            score_gain_per_molecule[sampled].mean(axis=1), [2.5, 97.5])
        report[str(index)] = {
            "property": name, "external_training_mean_mse": reference,
            "probes": probes,
            "alpha3_trained_minus_initial_score": float(
                score_gain_per_molecule.mean()),
            "alpha3_trained_minus_initial_score_bootstrap_95_percent_interval":
                score_gain_interval.tolist(),
        }
    return report


def draw_property_transfer(report, outdir):
    selected = (0, 1, 2, 3, 4, 5, 11)
    columns = (
        ("element_counts", "Element\ncounts"),
        ("alpha1_step0_elements_plus_pooled", "α=1\nstep 0"),
        ("alpha1_step40000_elements_plus_pooled", "α=1\nstep 40k"),
        ("alpha3_step0_elements_plus_pooled", "α=3\nstep 0"),
        ("alpha3_step40000_elements_plus_pooled", "α=3\nstep 40k"),
    )
    values = np.array([
        [report[str(index)]["probes"][key]["score_relative_to_training_mean"]
         for key, _ in columns] for index in selected
    ])
    fig, ax = plt.subplots(figsize=(10.7, 6.1), layout="constrained")
    image = ax.imshow(values, vmin=-.1, vmax=1, cmap="viridis", aspect="auto")
    for row in range(len(selected)):
        for column in range(len(columns)):
            value = values[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=10, color="black" if value > .65 else "white")
    ax.set(xticks=np.arange(len(columns)),
           xticklabels=[label for _, label in columns],
           yticks=np.arange(len(selected)),
           yticklabels=[QM9_PROPERTIES[index] for index in selected],
           title="External QM9 property transfer: train 200 → extra 1,000")
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, label="1 − MSE / training-mean baseline MSE (linear)")
    fig.savefig(outdir / "external_qm9_property_transfer.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    full_dataset = QM9(root=str(DATA_ROOT))[:1400]
    archived = full_dataset[:400]
    additional = full_dataset[400:1400]
    loader = DataLoader(archived, batch_size=64, shuffle=False)
    external_loader = DataLoader(additional, batch_size=64, shuffle=False)
    with np.load(ROOT / "animations" / "analysis_total400.npz") as cache:
        train = np.flatnonzero(cache["train_mask"])
        test = np.flatnonzero(~cache["train_mask"])
        truth = cache["truth"].copy()
        cached_predictions = {alpha: cache[f"alpha{alpha}_prediction"].copy()
                              for alpha in (1, 3)}
        cached_predictions["steps"] = cache["steps"].copy()
    concepts, elements, fallback_archived = chemical_labels(archived)
    external_concepts, external_elements, fallback_external = chemical_labels(additional)
    external_truth = np.array([float(graph.y[0, TARGET]) for graph in additional])
    states, arrays = {}, {}
    for alpha in (1, 3):
        for step in STEPS:
            state, data = analyze_state(alpha, step, loader, device, truth, elements,
                                        train, test, cached_predictions)
            states[(alpha, step)] = state
            arrays[(alpha, step)] = data
    trajectory = {str(alpha): weight_trajectory(alpha) for alpha in (1, 3)}
    external_matrix = {}
    external_data = None
    for alpha in (1, 3):
        data = arrays[(alpha, 40000)]
        pooled = encode(data["model"], external_loader, device)
        scores = (pooled - data["pooled"][train].mean(axis=0)) @ data["directions"].T
        external_matrix[str(alpha)] = {"concepts": list(concepts)}
        for component in range(1, 9):
            external_matrix[str(alpha)][str(component)] = {}
            for concept, training_label in concepts.items():
                score_train = data["scores"][train, component - 1]
                covariance = np.corrcoef(score_train, training_label[train])[0, 1]
                sign = 1 if covariance >= 0 else -1
                score = sign * scores[:, component - 1]
                label = external_concepts[concept]
                pairs = composition_matched_pairs(
                    label, external_elements, np.arange(len(additional)))
                pair_auc, group_auc, groups = matched_auc(score, pairs, external_elements)
                external_matrix[str(alpha)][str(component)][concept] = {
                    "train_correlation": float(covariance),
                    "external_auc": auc(score, label),
                    "matched_auc": pair_auc, "matched_group_mean_auc": group_auc,
                    "matched_groups": groups, "matched_pairs": int(len(pairs))}
        if alpha == 3:
            original = head(pooled, data["parameters"])[1]
            without = head(pooled - scores[:, 2, None] * data["directions"][2],
                           data["parameters"])[1]
            nitrile = external_concepts["nitrile"].astype(bool)
            original_mse = squared_error(original, external_truth, np.arange(len(additional)))
            external_data = {
                "overall_test_mse": original_mse,
                "without_pc3_overall_test_mse": squared_error(
                    without, external_truth, np.arange(len(additional))),
                "nitrile": {
                    "original_test_mse": squared_error(
                        original, external_truth, np.flatnonzero(nitrile)),
                    "without_pc3_test_mse": squared_error(
                        without, external_truth, np.flatnonzero(nitrile)),
                    "paired_delta": bootstrap_delta(
                        original, without, external_truth, np.flatnonzero(nitrile))},
                "other": {
                    "original_test_mse": squared_error(
                        original, external_truth, np.flatnonzero(~nitrile)),
                    "without_pc3_test_mse": squared_error(
                        without, external_truth, np.flatnonzero(~nitrile)),
                    "paired_delta": bootstrap_delta(
                        original, without, external_truth, np.flatnonzero(~nitrile))},
            }
            sign = 1 if np.corrcoef(data["scores"][train, 2],
                                    concepts["nitrile"][train])[0, 1] >= 0 else -1
            external_plot = {"arrays": {"nitrile": nitrile, "pc3_oriented": sign * scores[:, 2],
                                        "truth": external_truth, "original": original,
                                        "without_pc3": without}}
    nitrile_stages = {}
    nitrile_pairs = composition_matched_pairs(
        external_concepts["nitrile"], external_elements, np.arange(len(additional)))
    for alpha in (1, 3):
        nitrile_stages[str(alpha)] = {}
        for step in STEPS:
            data = arrays[(alpha, step)]
            pooled = encode(data["model"], external_loader, device)
            scores = (pooled - data["pooled"][train].mean(axis=0)) @ data["directions"].T
            records = {}
            for component in (1, 2, 3):
                covariance = np.corrcoef(data["scores"][train, component - 1],
                                         concepts["nitrile"][train])[0, 1]
                sign = 1 if covariance >= 0 else -1
                oriented = sign * scores[:, component - 1]
                pair_auc, group_auc, groups = matched_auc(
                    oriented, nitrile_pairs, external_elements)
                records[str(component)] = {
                    "external_auc": auc(oriented, external_concepts["nitrile"]),
                    "matched_auc": pair_auc, "matched_group_mean_auc": group_auc,
                    "matched_groups": groups}
            nitrile_stages[str(alpha)][str(step)] = records
    matched_controls = []
    for additional_control in ((), ("ring",), ("carbonyl",), ("hydroxyl",), ("amine",)):
        previous = matched_controls[-1]["matched_concepts"] if matched_controls else []
        controls = previous + list(additional_control)
        matching = np.column_stack((external_elements, *[
            external_concepts[name] for name in controls])) if controls else external_elements
        pairs = composition_matched_pairs(
            external_concepts["nitrile"], matching, np.arange(len(additional)))
        pair_auc, group_auc, groups = matched_auc(
            external_plot["arrays"]["pc3_oriented"], pairs, matching)
        matched_controls.append({"matched_concepts": controls, "pairs": int(len(pairs)),
                                 "groups": groups, "pair_auc": pair_auc,
                                 "group_mean_auc": group_auc})
    similarities = {}
    for step in STEPS:
        first = arrays[(1, step)]["pooled"]
        second = arrays[(3, step)]["pooled"]
        similarities[str(step)] = {
            "raw": linear_cka(first[test], second[test]),
            "after_element_counts": linear_cka(
                residualize_on_training(first, elements, train)[test],
                residualize_on_training(second, elements, train)[test]),
            "after_element_counts_and_target": linear_cka(
                residualize_on_training(first, np.column_stack((elements, truth)), train)[test],
                residualize_on_training(second, np.column_stack((elements, truth)), train)[test]),
        }
    report = {
        "dataset": "QM9 first 400; archived 200 train/200 test split",
        "external_validation": "QM9 entries 400 through 1399, none in archived split",
        "pca_fit": "Archived training 200 only, independently for each state",
        "model_output": "One scalar polarizability regression value; no logit matrix",
        "states": [states[(alpha, step)] for alpha in (1, 3) for step in STEPS],
        "weight_trajectories": trajectory,
        "external_single_pc_concept_tests": external_matrix,
        "external_nitrile_pc_axes_over_training": nitrile_stages,
        "external_alpha3_pc3_nitrile_matching_controls": matched_controls,
        "external_alpha3_pc3_intervention": external_data,
        "heldout_alpha1_alpha3_pooled_linear_cka": similarities,
        "n100_comparison": analyze_n100(device),
        "external_qm9_property_transfer": analyze_property_transfer(
            full_dataset, elements, external_elements, arrays, external_loader,
            device, train),
        "unsanitized_smiles_fallbacks": {"archived": fallback_archived,
                                         "external": fallback_external},
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    draw_pca_fidelity(states, args.outdir)
    draw_geometry(arrays, truth, elements, concepts, args.outdir)
    draw_weight_matrices(arrays, args.outdir)
    for alpha in (1, 3):
        draw_edge_filters(arrays, alpha, args.outdir)
    draw_activations(arrays, truth, args.outdir)
    draw_scalar_output_matrix(arrays, truth, args.outdir)
    draw_weight_trajectory(trajectory, args.outdir)
    draw_external_pc_concepts(external_matrix, args.outdir)
    draw_pc3_validation(external_plot, args.outdir)
    draw_nitrile_stages(nitrile_stages, args.outdir)
    draw_matched_molecule_gallery(additional, external_elements, external_plot, args.outdir)
    draw_nitrile_matching_controls(matched_controls, args.outdir)
    draw_n100_comparison(report["n100_comparison"], args.outdir)
    draw_property_transfer(report["external_qm9_property_transfer"], args.outdir)
    for alpha in (1, 3):
        state = states[(alpha, 40000)]
        print(f"alpha={alpha} 40k PC1+2={state['pc1_plus_pc2_variance_fraction']:.4f} "
              f"baseline={state['baseline_test_mse']:.4f} "
              f"retain2={state['retained_pc_tests'][1]['test_mse']:.4f} "
              f"remove3={state['removed_pc_tests'][2]['test_mse']:.4f}")
    print(f"external α=3 PC3 nitrile AUC="
          f"{external_matrix['3']['3']['nitrile']['external_auc']:.3f}, "
          f"matched={external_matrix['3']['3']['nitrile']['matched_auc']:.3f}")
    print(f"Saved {args.outdir}")


if __name__ == "__main__":
    main()
