#!/usr/bin/env python3
"""Non-frequency diagnostics for frozen Omnigrok QM9 checkpoints.

Fit probes only on the archived 200 training molecules, evaluate on the next
1,000 QM9 molecules, and check actual size extrapolation and node permutation.
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
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from ablate_qm9 import ridge_predict, train_only_probe
from map_omnigrok_structure import load_model, pca_train
from reproduce_qm9 import DATA_ROOT, ROOT


OUT = ROOT / "animations" / "nonfrequency_checks" / "total400"
STEPS = (0, 20000, 40000, 87500)
PROPERTY_INDICES = {
    "dipole": 0, "polarizability": 1, "HOMO": 2, "LUMO": 3, "gap": 4,
}
LAYERS = ("input_sum", "conv1_sum32", "conv1_PCA16", "conv2_sum16", "hidden32")
MODEL_LINES = (
    (3, 0, "#9c80cd", "α=3 init"),
    (3, 40000, "#6d28d9", "α=3 40k"),
    (1, 40000, "#2563eb", "α=1 40k"),
)


@torch.inference_mode()
def extract_layers(model, loader, device):
    rows = {name: [] for name in ("input_sum", "conv1_sum32", "conv2_sum16",
                                  "hidden32", "prediction")}
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        input_sum = global_add_pool(batch.x, batch.batch)
        nodes1 = F.relu(model.conv1(batch.x, batch.edge_index, batch.edge_attr))
        conv1_sum = global_add_pool(nodes1, batch.batch)
        nodes2 = F.relu(model.conv2(nodes1, batch.edge_index, batch.edge_attr))
        conv2_sum = global_add_pool(nodes2, batch.batch)
        hidden = F.relu(model.fc_1(conv2_sum))
        prediction = model.out(hidden)
        for name, value in (
            ("input_sum", input_sum), ("conv1_sum32", conv1_sum),
            ("conv2_sum16", conv2_sum), ("hidden32", hidden),
            ("prediction", prediction),
        ):
            rows[name].append(value.cpu().numpy())
    return {name: np.concatenate(items).astype(np.float64)
            for name, items in rows.items()}


def heavy_atom_moments(graph):
    atomic_numbers = graph.z.numpy()
    selected = np.flatnonzero(atomic_numbers != 1)
    lookup = {int(original): index for index, original in enumerate(selected)}
    adjacency = np.zeros((len(selected), len(selected)), dtype=np.float64)
    for old_a, old_b in graph.edge_index.numpy().T:
        if int(old_a) in lookup and int(old_b) in lookup:
            adjacency[lookup[int(old_a)], lookup[int(old_b)]] = 1
    squared = adjacency @ adjacency
    cubed = squared @ adjacency
    fourth = squared @ squared
    return [float(np.trace(x)) for x in (squared, cubed, fourth)]


def element_counts(graph):
    atomic_numbers = graph.z.numpy()
    return [int(np.sum(atomic_numbers == z)) for z in (1, 6, 7, 8, 9)]


def external_score(prediction, target, outside, baseline_mse):
    mse = float(np.mean((prediction[outside] - target[outside]) ** 2))
    return {"external_mse": mse,
            "external_score_vs_training_mean": float(1 - mse / baseline_mse)}


def probe(features, target, train, outside, baseline_mse):
    selected = train_only_probe(features, target, train, outside)
    prediction = ridge_predict(features, target, train, selected["selected_penalty"])
    result = {
        **external_score(prediction, target, outside, baseline_mse),
        "selected_penalty": float(selected["selected_penalty"]),
        "train_cv_mse": float(selected["train_cv_mse"]),
    }
    return result, prediction


def bootstrap_probe_advantage(better, worse, target, outside, baseline_mse,
                              seed=2026):
    """Positive scores mean the first fitted probe has lower external MSE."""
    per_molecule = ((worse[outside] - target[outside]) ** 2 -
                    (better[outside] - target[outside]) ** 2) / baseline_mse
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(outside), size=(2000, len(outside)))
    return {
        "score_advantage": float(per_molecule.mean()),
        "molecule_bootstrap_95_percent_interval": np.percentile(
            per_molecule[sampled].mean(axis=1), [2.5, 97.5]).tolist(),
    }


def size_groups(heavy, archived_test):
    outside = np.arange(400, len(heavy))
    return {
        "archived_test_up_to_6_heavy": archived_test,
        "outside_6_heavy": outside[heavy[outside] == 6],
        "outside_7_heavy": outside[heavy[outside] == 7],
        "outside_other_heavy": outside[(heavy[outside] != 6) & (heavy[outside] != 7)],
    }


def group_errors(prediction, target, groups):
    return {name: {
        "molecules": int(len(indices)),
        "mse": float(np.mean((prediction[indices] - target[indices]) ** 2))
        if len(indices) else None,
        "mae": float(np.mean(np.abs(prediction[indices] - target[indices])))
        if len(indices) else None,
    } for name, indices in groups.items()}


@torch.inference_mode()
def permutation_check(model, dataset, device, seed=2026, count=50):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.arange(400, len(dataset)), size=count, replace=False)
    output_abs, pooled_abs = [], []
    for index in chosen:
        graph = dataset[int(index)]
        order = torch.tensor(rng.permutation(graph.num_nodes), dtype=torch.long)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(graph.num_nodes)
        permuted = graph.clone()
        permuted.x = graph.x[order]
        permuted.edge_index = inverse[graph.edge_index]
        batch = Batch.from_data_list([graph, permuted]).to(device)
        nodes1 = F.relu(model.conv1(batch.x, batch.edge_index, batch.edge_attr))
        nodes2 = F.relu(model.conv2(nodes1, batch.edge_index, batch.edge_attr))
        pooled = global_add_pool(nodes2, batch.batch)
        output = model.out(F.relu(model.fc_1(pooled)))
        output_abs.append(float(torch.abs(output[0] - output[1]).item()))
        pooled_abs.append(float(torch.max(torch.abs(pooled[0] - pooled[1])).item()))
    return {"molecules": count, "max_abs_output_difference": max(output_abs),
            "mean_abs_output_difference": float(np.mean(output_abs)),
            "max_abs_pooled_channel_difference": max(pooled_abs)}


def plot_layer_probes(states, baselines, outdir):
    selected = ("polarizability", "HOMO", "LUMO", "gap")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), layout="constrained")
    x = np.arange(len(LAYERS))
    for ax, property_name in zip(axes.flat, selected):
        for alpha, step, color, label in MODEL_LINES:
            values = [states[f"alpha{alpha}_step{step}"]["probes"][property_name][layer][
                "external_score_vs_training_mean"] for layer in LAYERS]
            ax.plot(x, values, "o-", label=label, color=color, linewidth=2)
        ax.axhline(baselines[property_name]["external_score_vs_training_mean"],
                   color="#64748b", linestyle="--", linewidth=1.5,
                   label="Element counts")
        ax.set(xticks=x, xticklabels=("Input\nsum", "Conv1\n32", "Conv1\nPCA16",
                                     "Conv2\n16", "Readout\n32"),
               ylabel="External probe score", title=property_name,
               ylim=(-.1, 1.02))
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Where do other QM9 properties remain linearly accessible?",
                 fontsize=14)
    fig.savefig(outdir / "layerwise_property_probes.png", dpi=170)
    plt.close(fig)


def plot_moment_probes(states, baselines, outdir):
    names = ("trace_A2", "trace_A3", "trace_A4")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), layout="constrained")
    x = np.arange(len(LAYERS))
    for ax, name in zip(axes, names):
        for alpha, step, color, label in MODEL_LINES:
            values = [states[f"alpha{alpha}_step{step}"]["probes"][name][layer][
                "external_score_vs_training_mean"] for layer in LAYERS]
            ax.plot(x, values, "o-", label=label, color=color, linewidth=2)
        ax.axhline(baselines[name]["external_score_vs_training_mean"],
                   color="#64748b", linestyle="--", label="Element counts")
        ax.set(xticks=x, xticklabels=("Input", "Conv1", "Conv1\nPCA16",
                                     "Conv2", "Readout"),
               ylabel="External probe score",
               title=name.replace("trace_A", "tr(A^") + ")")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Closed-walk counts of the unweighted heavy-atom bond graph",
                 fontsize=14)
    fig.savefig(outdir / "graph_walk_moment_probes.png", dpi=170)
    plt.close(fig)


def plot_size_errors(states, composition_baseline, outdir):
    names = ("archived_test_up_to_6_heavy", "outside_6_heavy",
             "outside_7_heavy")
    labels = ("Archived test\n≤6 heavy", "Outside\n6 heavy",
              "Outside\n7 heavy")
    fig, ax = plt.subplots(figsize=(9.6, 5.2), layout="constrained")
    x = np.arange(len(names))
    for alpha, step, color, label in (
        (1, 40000, "#2563eb", "α=1 40k"),
        (3, 40000, "#7c3aed", "α=3 40k"),
        (1, 87500, "#60a5fa", "α=1 final"),
        (3, 87500, "#a78bfa", "α=3 final"),
    ):
        groups = states[f"alpha{alpha}_step{step}"]["polarizability_error_by_size"]
        ax.plot(x, [groups[name]["mse"] for name in names], "o-", color=color,
                label=label, linewidth=2)
    ax.plot(x, [composition_baseline[name]["mse"] for name in names], "s--",
            color="#475569", label="Element-count ridge", linewidth=1.5)
    counts = [states["alpha3_step40000"]["polarizability_error_by_size"][name][
        "molecules"] for name in names]
    ax.set(xticks=x, xticklabels=[
        f"{label}\n(n={count})" for label, count in zip(labels, counts)],
        ylabel="Polarizability MSE", title="Saved models on the original split and larger QM9 graphs")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(outdir / "size_generalization_mse.png", dpi=170)
    plt.close(fig)


def plot_probe_trajectory(states, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), layout="constrained")
    for layer, color, label in (
        ("conv1_PCA16", "#9c80cd", "Conv1, first 16 PCs"),
        ("conv2_sum16", "#6d28d9", "Conv2, 16 channels"),
    ):
        values = [states[f"alpha3_step{step}"]["probes"]["LUMO"][layer][
            "external_score_vs_training_mean"] for step in STEPS]
        axes[0].plot(STEPS, values, "o-", color=color, label=label, linewidth=2)
    axes[0].set(xlabel="Optimizer step", ylabel="External LUMO probe score",
                title="LUMO access over training", ylim=(0, 1))
    axes[0].legend(fontsize=9)
    later = STEPS[1:]
    for group, color, label in (
        ("archived_test_up_to_6_heavy", "#2563eb", "Original held-out"),
        ("outside_7_heavy", "#7c3aed", "Outside, 7 heavy atoms"),
    ):
        values = [states[f"alpha3_step{step}"]["polarizability_error_by_size"][
            group]["mse"] for step in later]
        axes[1].plot(later, values, "o-", color=color, label=label, linewidth=2)
    axes[1].set(xlabel="Optimizer step", ylabel="Polarizability MSE",
                title="Prediction error over training")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("α=3 checkpoint progression; both axes linear", fontsize=13)
    fig.savefig(outdir / "probe_trajectory_and_prediction_error.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    dataset = QM9(root=str(DATA_ROOT))[:1400]
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with np.load(ROOT / "animations" / "analysis_total400.npz") as cache:
        train = np.flatnonzero(cache["train_mask"])
        archived_test = np.flatnonzero(~cache["train_mask"])
        cached_steps = cache["steps"].copy()
        cached_predictions = {alpha: cache[f"alpha{alpha}_prediction"].copy()
                              for alpha in (1, 3)}
    outside = np.arange(400, len(dataset))
    elements = np.asarray([element_counts(graph) for graph in dataset],
                          dtype=np.float64)
    heavy = elements[:, 1:].sum(axis=1).astype(int)
    qm9_targets = np.stack([graph.y.numpy().ravel() for graph in dataset])
    moment_values = np.asarray([heavy_atom_moments(graph) for graph in dataset],
                               dtype=np.float64)
    targets = {name: qm9_targets[:, index].astype(np.float64)
               for name, index in PROPERTY_INDICES.items()}
    targets.update({f"trace_A{power}": moment_values[:, pos]
                    for pos, power in enumerate((2, 3, 4))})
    baselines = {}
    for name, target in targets.items():
        reference = float(np.mean((target[outside] - target[train].mean()) ** 2))
        result, _ = probe(elements, target, train, outside, reference)
        baselines[name] = {"reference_training_mean_mse": reference, **result}
    element_penalty = baselines["polarizability"]["selected_penalty"]
    element_prediction = ridge_predict(elements, targets["polarizability"],
                                       train, element_penalty)
    groups = size_groups(heavy, archived_test)
    states = {}
    permutation = {}
    for alpha in (1, 3):
        for step in STEPS:
            model = load_model(alpha, step, device)
            layers = extract_layers(model, loader, device)
            expected = cached_predictions[alpha][np.flatnonzero(
                cached_steps == step)[0]]
            max_difference = float(np.max(np.abs(layers["prediction"][:400, 0] -
                                                 expected)))
            tolerance = max(1e-3, 1e-6 * float(np.max(np.abs(expected))))
            if max_difference > tolerance:
                raise ValueError(
                    f"Saved prediction mismatch for alpha={alpha}, step={step}: "
                    f"{max_difference} > {tolerance}")
            _, _, conv1_pca_scores, _ = pca_train(layers["conv1_sum32"], train)
            layers["conv1_PCA16"] = conv1_pca_scores[:, :16]
            state = {
                "alpha": alpha, "step": step,
                "max_prediction_difference_from_animation": max_difference,
                "probes": {},
                "polarizability_error_by_size": group_errors(
                    layers["prediction"][:, 0], targets["polarizability"], groups),
            }
            for name, target in targets.items():
                reference = baselines[name]["reference_training_mean_mse"]
                state["probes"][name] = {}
                contrast = {}
                for layer in LAYERS:
                    features = np.column_stack((elements, layers[layer]))
                    result, predicted = probe(features, target, train, outside, reference)
                    state["probes"][name][layer] = result
                    if layer in ("conv1_sum32", "conv1_PCA16", "conv2_sum16"):
                        contrast[layer] = predicted
                if name in ("HOMO", "LUMO", "gap", "trace_A2",
                            "trace_A3", "trace_A4"):
                    state["probes"][name]["conv1_vs_conv2"] = bootstrap_probe_advantage(
                        contrast["conv1_sum32"], contrast["conv2_sum16"],
                        target, outside, reference)
                    state["probes"][name]["conv1_PCA16_vs_conv2"] = (
                        bootstrap_probe_advantage(
                            contrast["conv1_PCA16"], contrast["conv2_sum16"],
                            target, outside, reference))
            states[f"alpha{alpha}_step{step}"] = state
            if step == 40000:
                permutation[str(alpha)] = permutation_check(model, dataset, device)
    report = {
        "method": "Frozen N=400 checkpoints; fit on archived 200 train; evaluate "
                  "on next 1000 QM9 molecules, except original test size group",
        "layers": list(LAYERS),
        "graph_moments": "tr(A^2), tr(A^3), tr(A^4) of unweighted heavy-atom "
                         "bond adjacency; not Hückel Hamiltonian eigenvalues",
        "outside_group_counts": {name: int(len(indices)) for name, indices
                                 in groups.items()},
        "element_count_probe_baselines": baselines,
        "element_count_polarizability_error_by_size": group_errors(
            element_prediction, targets["polarizability"], groups),
        "states": states,
        "node_permutation_checks": permutation,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_layer_probes(states, baselines, args.outdir)
    plot_moment_probes(states, baselines, args.outdir)
    plot_size_errors(states, report["element_count_polarizability_error_by_size"],
                     args.outdir)
    plot_probe_trajectory(states, args.outdir)
    for alpha in (1, 3):
        state = states[f"alpha{alpha}_step40000"]
        lum = state["probes"]["LUMO"]
        print(f"alpha={alpha}, LUMO: conv1={lum['conv1_sum32']['external_score_vs_training_mean']:.3f}, "
              f"conv1-PCA16={lum['conv1_PCA16']['external_score_vs_training_mean']:.3f}, "
              f"conv2={lum['conv2_sum16']['external_score_vs_training_mean']:.3f}")
    print(f"Saved {args.outdir}")


if __name__ == "__main__":
    main()
