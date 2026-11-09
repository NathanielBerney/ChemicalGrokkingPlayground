#!/usr/bin/env python3
"""Probe chemical concepts in the frozen N=400 Omnigrok graph embeddings.

All probe fitting and ridge-penalty selection use the archived training half.
The test half is used once for descriptive AUCs. Element-count and scalar-
output probes are controls for composition and target-correlated shortcuts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, rdBase
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from ablate_qm9 import encode, ridge_predict, train_only_probe
from reproduce_qm9 import DATA_ROOT, ROOT, ExampleNet


OUT = ROOT / "animations" / "chemistry_probes" / "total400"
STATES = ((3, 0), (3, 10000), (3, 20000), (3, 40000), (3, 87500),
          (1, 0), (1, 40000))
CONCEPTS = {
    "ring": "[R]",
    "carbonyl": "[CX3]=[OX1]",
    "hydroxyl": "[OX2H]",
    "nitrile": "[CX2]#N",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
}
PENALTIES = (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)


def chemical_labels(dataset):
    rdBase.DisableLog("rdApp.error")
    queries = {name: Chem.MolFromSmarts(smarts) for name, smarts in CONCEPTS.items()}
    values = {name: [] for name in CONCEPTS}
    element_counts = []
    fallback_count = 0
    for graph in dataset:
        molecule = Chem.MolFromSmiles(graph.smiles)
        if molecule is None:
            molecule = Chem.MolFromSmiles(graph.smiles, sanitize=False)
            if molecule is None:
                raise ValueError(f"Unparseable QM9 SMILES: {graph.smiles}")
            molecule.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(molecule)
            fallback_count += 1
        for name, query in queries.items():
            values[name].append(int(molecule.HasSubstructMatch(query)))
        z = graph.z.numpy()
        element_counts.append([np.sum(z == atomic_number)
                               for atomic_number in (1, 6, 7, 8, 9)])
    return ({name: np.array(items, dtype=np.float64)
             for name, items in values.items()},
            np.array(element_counts, dtype=np.float64), fallback_count)


def auc(scores, labels):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires both concept classes")
    return float(np.mean(positive[:, None] > negative[None, :])
                 + .5 * np.mean(positive[:, None] == negative[None, :]))


def composition_matched_pairs(labels, elements, indices):
    groups = {}
    for index in indices:
        groups.setdefault(tuple(elements[index]), []).append(index)
    pairs = [(positive, negative) for group in groups.values()
             for positive in group if labels[positive] == 1
             for negative in group if labels[negative] == 0]
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def matched_auc(scores, pairs, elements):
    if not len(pairs):
        raise ValueError("No composition-matched concept pairs")
    difference = scores[pairs[:, 0]] - scores[pairs[:, 1]]
    correct = (difference > 1e-10) + .5 * (np.abs(difference) <= 1e-10)
    groups = {}
    for pair_index, pair in enumerate(pairs):
        groups.setdefault(tuple(elements[pair[0]]), []).append(pair_index)
    return float(np.mean(correct)), float(np.mean([np.mean(correct[indices])
                                                  for indices in groups.values()])), len(groups)


def probe(features, labels, train, test, matched_pairs, elements):
    folds = np.array_split(np.random.default_rng(2026).permutation(train), 5)
    choices = []
    for penalty in PENALTIES:
        errors = []
        for validation in folds:
            fit = np.setdiff1d(train, validation, assume_unique=True)
            predicted = ridge_predict(features, labels, fit, penalty)
            errors.append(float(np.mean((predicted[validation] - labels[validation]) ** 2)))
        choices.append((float(np.mean(errors)), penalty))
    cv_mse, penalty = min(choices)
    predicted = ridge_predict(features, labels, train, penalty)
    matched_pair_auc, matched_group_auc, _ = matched_auc(predicted, matched_pairs, elements)
    return {"test_auc": auc(predicted[test], labels[test]),
            "composition_matched_test_auc": matched_pair_auc,
            "composition_matched_group_mean_auc": matched_group_auc,
            "train_cv_mse": cv_mse, "selected_penalty": penalty}


def load_pooled(alpha, step, loader, device):
    path = (ROOT / "runs" / f"grokking_total400_alpha{alpha}_animation"
            / "snapshots" / f"step_{step:06d}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != step:
        raise ValueError(f"Checkpoint step mismatch: {path}")
    model = ExampleNet().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return encode(model, loader, device)


def draw_probe_map(report, outdir):
    names = list(CONCEPTS)
    selected = ("element_counts", "alpha3_step0_pooled16", "alpha3_step10000_pooled16",
                "alpha3_step40000_pooled16", "alpha1_step40000_pooled16",
                "alpha3_step40000_output")
    labels = ("Element counts", "α=3 initial pooled", "α=3 10k pooled",
              "α=3 40k pooled", "α=1 40k pooled", "α=3 40k output")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), layout="constrained")
    for ax, metric, title in ((axes[0], "test_auc", "All held-out molecules"),
                              (axes[1], "composition_matched_test_auc",
                               "Same H/C/N/O/F counts")):
        matrix = np.array([[report["probes"][name][source][metric]
                            for name in names] for source in selected])
        im = ax.imshow(matrix, cmap="viridis", vmin=.45, vmax=1, aspect="auto")
        ax.set(xticks=np.arange(len(names)), xticklabels=names,
               yticks=np.arange(len(labels)), yticklabels=labels,
               xlabel="RDKit concept", title=title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                        color="white" if value < .72 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, label="Held-out AUC (0.5 = chance)", shrink=.8)
    fig.suptitle("Can a frozen graph embedding distinguish chemical concepts?", fontsize=15)
    fig.savefig(outdir / "chemical_concept_probe_auc.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    dataset = QM9(root=str(DATA_ROOT))[:400]
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with np.load(ROOT / "animations" / "analysis_total400.npz") as cached:
        train = np.flatnonzero(cached["train_mask"])
        test = np.flatnonzero(~cached["train_mask"])
        truth = cached["truth"].copy()
        steps = cached["steps"].copy()
        predictions = {alpha: cached[f"alpha{alpha}_prediction"].copy()
                       for alpha in (1, 3)}
    concepts, elements, fallback_count = chemical_labels(dataset)
    concept_matrix = np.column_stack(list(concepts.values()))
    target_baselines = {
        "total_atom_count": train_only_probe(elements.sum(axis=1, keepdims=True), truth,
                                              train, test),
        "element_counts": train_only_probe(elements, truth, train, test),
        "element_counts_plus_five_concepts": train_only_probe(
            np.column_stack((elements, concept_matrix)), truth, train, test),
    }
    sources = {"element_counts": elements}
    for alpha, step in STATES:
        pooled = load_pooled(alpha, step, loader, device)
        sources[f"alpha{alpha}_step{step}_pooled16"] = pooled
        index = int(np.flatnonzero(steps == step)[0])
        sources[f"alpha{alpha}_step{step}_output"] = predictions[alpha][index, :, None]
        sources[f"alpha{alpha}_step{step}_pooled_plus_elements"] = np.column_stack((pooled, elements))
    results = {}
    matched_counts = {}
    for name, label in concepts.items():
        if min(label[train].sum(), len(train) - label[train].sum(),
               label[test].sum(), len(test) - label[test].sum()) < 10:
            raise ValueError(f"Too few examples of concept {name} in either split")
        matched_pairs = composition_matched_pairs(label, elements, test)
        matched_counts[name] = len(matched_pairs)
        results[name] = {source: probe(features, label, train, test, matched_pairs, elements)
                         for source, features in sources.items()}
    report = {
        "dataset": "first 400 QM9 molecules; same 200/200 split as archived animations",
        "concept_definitions_smarts": CONCEPTS,
        "element_features": ["H", "C", "N", "O", "F"],
        "unsanitized_smiles_fallbacks": fallback_count,
        "prevalence": {name: {"train_positive": int(label[train].sum()),
                              "test_positive": int(label[test].sum()),
                              "composition_matched_test_pairs": matched_counts[name],
                              "composition_matched_test_groups": len({
                                  tuple(elements[pair[0]]) for pair in
                                  composition_matched_pairs(label, elements, test)})}
                       for name, label in concepts.items()},
        "probes": results,
        "polarizability_target_baselines": target_baselines,
        "method": "Ridge score for each binary concept; 5-fold training-only CV selects penalty by MSE. Matched AUC compares opposite-label held-out molecules with identical H/C/N/O/F counts. Pairs share molecules, so counts are not independent observations. No GNN weights are changed.",
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    draw_probe_map(report, args.outdir)
    for concept, rows in results.items():
        print(concept, "elements", round(rows["element_counts"]["test_auc"], 3),
              "alpha3_initial", round(rows["alpha3_step0_pooled16"]["test_auc"], 3),
              "alpha3_40k", round(rows["alpha3_step40000_pooled16"]["test_auc"], 3),
              "alpha3_40k_plus_elements",
              round(rows["alpha3_step40000_pooled_plus_elements"]["test_auc"], 3),
              "matched_initial",
              round(rows["alpha3_step0_pooled16"]["composition_matched_test_auc"], 3),
              "matched_40k",
              round(rows["alpha3_step40000_pooled16"]["composition_matched_test_auc"], 3))
    print(f"Saved {args.outdir}")


if __name__ == "__main__":
    main()
