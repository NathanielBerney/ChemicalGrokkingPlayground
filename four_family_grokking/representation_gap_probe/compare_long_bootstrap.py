#!/usr/bin/env python3
"""Cluster bootstrap held-out changes from 20k to 100k steps."""

import json
from collections import defaultdict

import numpy as np

from train import HERE

KINDS = ("matrix_rows", "pi_graph", "molecular_graph", "smiles")


def main():
    with np.load(HERE / "features.npz") as d:
        test = d["test"].copy()
        eig = d["eigenvalues"]
        n = d["site_counts"]
        ne = d["pi_electrons"]
        true = d["gap"]
    groups = defaultdict(list)
    for j, i in enumerate(test):
        key = (int(n[i]), int(ne[i]), tuple(np.round(eig[i, :n[i]], 5)))
        groups[key].append(j)
    clusters = [np.asarray(x) for x in groups.values()]
    rng = np.random.default_rng(303)
    sample = rng.integers(0, len(clusters), size=(2000, len(clusters)))
    counts = np.asarray([len(x) for x in clusters])
    denom = counts[sample].sum(1)
    results = {}
    for kind in KINDS:
        old_path = HERE / "runs_20k_seed0" / kind / "final_probe.npz"
        new_path = HERE / "long_100k_seed0" / kind / "final_probe.npz"
        if not new_path.exists():
            continue
        with np.load(old_path) as d:
            old = d["pred"][test]
        with np.load(new_path) as d:
            new = d["pred"][test]
        err_old, err_new = np.abs(old - true[test]), np.abs(new - true[test])
        acc_delta = (err_new / true[test] < .01).astype(float) - (err_old / true[test] < .01).astype(float)
        mae_delta = err_new - err_old
        out = {}
        for name, values in (("test_1pct_accuracy", acc_delta), ("test_mae", mae_delta)):
            cluster_sums = np.asarray([values[x].sum() for x in clusters])
            dist = cluster_sums[sample].sum(1) / denom
            out[name] = {"change_100k_minus_20k": float(values.mean()),
                         "cluster_bootstrap_95pct_interval": np.quantile(dist, [.025, .975]).tolist()}
        results[kind] = out
    root = HERE / "long_100k_seed0"
    (root / "paired_changes.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
