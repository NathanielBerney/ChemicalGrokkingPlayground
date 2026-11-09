#!/usr/bin/env python3
"""Constant-prediction and exact Hückel-diagonalization controls."""

import json

import numpy as np

from train import HERE, score


def main():
    with np.load(HERE / "features.npz", allow_pickle=False) as d:
        train, test = d["train"], d["test"]
        gap = d["gap"]
        eig = d["eigenvalues"]
        ne = d["pi_electrons"]
    mean_log = float(np.log(gap[train]).mean())
    constant = np.full(len(test), np.exp(mean_log))
    occ = ne // 2 - 1
    exact = eig[np.arange(len(gap)), occ + 1] - eig[np.arange(len(gap)), occ]
    results = {"constant_loggap": score(constant, gap[test]),
               "exact_huckel": score(exact[test], gap[test]),
               "exact_max_abs_error": float(np.max(np.abs(exact - gap)))}
    (HERE / "baselines.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
