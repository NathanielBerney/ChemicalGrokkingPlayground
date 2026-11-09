#!/usr/bin/env python3
"""Compare site-renumbering sensitivity after long training and augmentation."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train import HERE, MatrixRows, MatrixRowsTwoLayer, PiGraph, load_data
from permutation_probe import evaluate


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d, _, test = load_data(device)
    ids = np.random.default_rng(7).choice(test, size=1000, replace=False)
    root = HERE / "long_100k_seed0"
    result = {}
    fig, ax = plt.subplots(figsize=(9, 4))
    cases = (
        ("matrix_rows", MatrixRows, root),
        ("matrix_rows_2layer", MatrixRowsTwoLayer, root),
        ("matrix_rows_permaug", MatrixRows, root),
        ("matrix_rows_permaug_slowdecay", MatrixRows,
         HERE / "long_100k_slowdecay_seed0"),
        ("pi_graph", PiGraph, root),
    )
    for label, cls, run_root in cases:
        kind = "matrix_rows_permaug" if label == "matrix_rows_permaug_slowdecay" else label
        ckpt = torch.load(run_root / kind / "checkpoint_100000.pt", map_location=device, weights_only=False)
        model = cls().to(device)
        model.load_state_dict(ckpt["model"])
        result[label], delta = evaluate(kind, model, d, ids,
                                       ckpt["target_center"], ckpt["target_scale"])
        ordered = np.sort(np.maximum(delta, 1e-9))
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                lw=2, label=f"{label} (median {np.median(delta):.3g})")
    ax.set_xscale("log")
    ax.set_xlabel("Prediction change after equivalent site renumbering (|β|)")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    out = root / "plots"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "site_permutation_sensitivity.png", dpi=180, bbox_inches="tight")
    (root / "permutation_probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
