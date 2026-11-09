#!/usr/bin/env python3
"""Measure whether equivalent atom/site orderings change model predictions."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train import MatrixRows, PiGraph, MolecularGraph, HERE, load_data, model_call, score

RUNS = HERE / "runs_20k_seed0"


@torch.inference_mode()
def evaluate(kind, model, d, ids, center, scale, permutations=3):
    device = d["gap"].device
    ids_t = torch.as_tensor(ids, device=device)
    model.eval()
    original = np.exp(model_call(model, kind, d, ids_t).cpu().numpy() * scale + center)
    truth = d["gap"][ids_t].cpu().numpy()
    rng = np.random.default_rng(41)
    deltas, shuffled_scores = [], []
    for _ in range(permutations):
        if kind.startswith("matrix_rows") or kind == "pi_graph":
            H = d["H"][ids_t].clone()
            e = d["site_pi_electrons"][ids_t].clone()
            ns = d["site_counts"][ids_t].cpu().numpy()
            for j, n in enumerate(ns):
                perm = rng.permutation(int(n))
                H[j, :n, :n] = H[j, :n, :n][perm][:, perm]
                e[j, :n] = e[j, :n][perm]
            z = model(H, e, d["site_mask"][ids_t], d["aux"][ids_t])
        else:
            node = d["node"][ids_t].clone()
            bond = d["bond"][ids_t].clone()
            coupling = d["pi_coupling"][ids_t].clone()
            ns = d["atom_counts"][ids_t].cpu().numpy()
            for j, n in enumerate(ns):
                perm = rng.permutation(int(n))
                node[j, :n] = node[j, :n][perm]
                bond[j, :n, :n] = bond[j, :n, :n][perm][:, perm]
                coupling[j, :n, :n] = coupling[j, :n, :n][perm][:, perm]
            z = model(node, bond, coupling, d["atom_mask"][ids_t], d["aux"][ids_t])
        pred = np.exp(z.cpu().numpy() * scale + center)
        deltas.extend(np.abs(pred - original).tolist())
        shuffled_scores.append(score(pred, truth))
    return {
        "original": score(original, truth),
        "permuted_mean_1pct_accuracy": float(np.mean([x["accuracy_rel_1pct"] for x in shuffled_scores])),
        "permuted_mean_mae": float(np.mean([x["mae"] for x in shuffled_scores])),
        "prediction_change_median": float(np.median(deltas)),
        "prediction_change_p90": float(np.quantile(deltas, .9)),
        "prediction_change_max": float(np.max(deltas)),
    }, np.asarray(deltas)


def main():
    torch.set_num_threads(1)
    d, _, test = load_data(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    ids = np.random.default_rng(7).choice(test, size=1000, replace=False)
    result = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, cls in (("matrix_rows", MatrixRows), ("pi_graph", PiGraph),
                      ("molecular_graph", MolecularGraph)):
        ckpt = torch.load(RUNS / kind / "checkpoint_20000.pt", map_location=d["gap"].device, weights_only=False)
        model = cls().to(d["gap"].device)
        model.load_state_dict(ckpt["model"])
        result[kind], delta = evaluate(kind, model, d, ids, ckpt["target_center"], ckpt["target_scale"])
        ax.hist(np.maximum(delta, 1e-9), bins=np.geomspace(1e-9, 3, 70), histtype="step", lw=2,
                label=kind, density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute gap-prediction change after atom/site relabeling (|β|)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    plot = HERE / "plots_20k_seed0"
    plot.mkdir(exist_ok=True)
    fig.savefig(plot / "permutation_sensitivity.png", dpi=180)
    (HERE / "permutation_probe_20k_seed0.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
