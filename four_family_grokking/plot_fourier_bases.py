#!/usr/bin/env python3
"""Plot exact Hückel eigenbases and pericyclic cyclic reference bases.

Periodic DFT modes diagonalize uniform circulant rings; anti-periodic DFT
modes diagonalize a uniform Möbius signed ring; sine modes diagonalize an
open uniform chain. Pericyclic matrices vary in edge magnitude and diagonal
terms, so their gauged periodic/anti-periodic modes are reference bases,
not exact eigenbases.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "fourier_bases"


def uniform_h(family: str, n: int) -> np.ndarray:
    H = np.zeros((n, n))
    i = np.arange(n - 1)
    H[i, i + 1] = H[i + 1, i] = -1
    if family != "linear":
        H[0, -1] = H[-1, 0] = 1 if family == "mobius" else -1
    return H


def modal_basis(family: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    j = np.arange(n)[:, None]
    if family == "linear":
        m = np.arange(1, n + 1)[None, :]
        Q = np.sqrt(2 / (n + 1)) * np.sin(np.pi * (j + 1) * m / (n + 1))
        energy = -2 * np.cos(np.pi * m.ravel() / (n + 1))
    else:
        k = np.arange(n)[None, :]
        shift = .5 if family == "mobius" else 0
        theta = 2 * np.pi * (k + shift) / n
        Q = np.exp(1j * j * theta) / np.sqrt(n)
        energy = -2 * np.cos(theta.ravel())
    return Q, energy


def gauged_pericyclic_basis(H: np.ndarray) -> tuple[np.ndarray, int]:
    n = len(H)
    edge_sign = np.sign(-H[np.arange(n), (np.arange(n) + 1) % n]).astype(int)
    phase = int(np.prod(edge_sign))
    gauge = np.ones(n)
    for i in range(n - 1):
        gauge[i + 1] = gauge[i] * edge_sign[i]
    Q, _ = modal_basis("circulant" if phase > 0 else "mobius", n)
    return gauge[:, None] * Q, phase


def plot_uniform(family: str, n: int) -> dict:
    H = uniform_h(family, n)
    Q, expected = modal_basis(family, n)
    transformed = Q.conj().T @ H @ Q
    orth_error = float(np.max(np.abs(Q.conj().T @ Q - np.eye(n))))
    offdiag_error = float(np.max(np.abs(transformed - np.diag(np.diag(transformed)))))
    eigen_error = float(np.max(np.abs(np.diag(transformed).real - expected)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    im = axes[0].imshow(H, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    axes[0].set(xlabel="Site", ylabel="Site", title="Signed Hückel matrix H")
    fig.colorbar(im, ax=axes[0], shrink=.7)
    mode = n // 4
    x = np.arange(n)
    axes[1].plot(x, Q[:, mode].real, label="sine" if family == "linear" else "real / cos")
    if family != "linear":
        axes[1].plot(x, Q[:, mode].imag, label="imag / sin")
    axes[1].set(xlabel="Site j", ylabel="Normalized mode amplitude",
                title=f"Mode {mode}: " + ("sine" if family == "linear" else
                                     "anti-periodic DFT" if family == "mobius" else "periodic DFT"))
    axes[1].legend(fontsize=8)
    axes[2].plot(np.sort(expected), ".", label="analytic modes")
    axes[2].plot(np.linalg.eigvalsh(H), "x", ms=3, alpha=.7, label="matrix eigvalsh")
    axes[2].set(xlabel="Sorted orbital index", ylabel="Energy / |β|",
                title="Exact spectral diagonalization")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{family.capitalize()} · n={n}")
    fig.savefig(OUT / f"{family}_fourier_basis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"n": n, "basis_orthogonality_max_error": orth_error,
            "offdiagonal_max_error": offdiag_error,
            "analytic_eigenvalue_max_error": eigen_error,
            "boundary_phase": -1 if family == "mobius" else 1 if family == "circulant" else None}


def plot_pericyclic() -> dict:
    with np.load(ROOT / "pericyclic_thermal_dataset" / "dataset.npz", allow_pickle=False) as data:
        H_all = data["H"]
        sizes = data["n_sites"]
        labels = data["y"]
    examples = {}
    for phase in (1, -1):
        for i, n in enumerate(sizes):
            H = H_all[i, :n, :n].astype(np.float64)
            Q, sign = gauged_pericyclic_basis(H)
            if sign == phase and n == 6:
                examples[phase] = (i, H, Q, int(labels[i]))
                break
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    report = {}
    for row, phase in enumerate((1, -1)):
        i, H, Q, label = examples[phase]
        transformed = Q.conj().T @ H @ Q
        offdiag = transformed - np.diag(np.diag(transformed))
        fraction = float(np.linalg.norm(offdiag) / np.linalg.norm(transformed))
        report[str(phase)] = {"sample_index": i, "n": len(H), "label": label,
                              "offdiagonal_frobenius_fraction": fraction,
                              "basis_orthogonality_max_error": float(np.max(np.abs(Q.conj().T @ Q - np.eye(len(H)))))}
        axes[row, 0].imshow(H, cmap="coolwarm", vmin=-1.2, vmax=1.2)
        axes[row, 0].set(xlabel="Active orbital", ylabel="Active orbital",
                         title=f"H matrix · {'Hückel' if phase > 0 else 'Möbius'} phase")
        axes[row, 1].imshow(np.abs(transformed), cmap="magma")
        axes[row, 1].set(xlabel="Reference mode", ylabel="Reference mode",
                         title=f"|Q†HQ| · off-diagonal fraction {fraction:.2f}")
    fig.suptitle("Pericyclic transition-state matrices · cyclic Fourier reference bases")
    fig.savefig(OUT / "pericyclic_fourier_reference.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return report


def main():
    OUT.mkdir(exist_ok=True)
    report = {f: plot_uniform(f, n) for f, n in
              (("circulant", 42), ("mobius", 40), ("linear", 42))}
    report["pericyclic"] = plot_pericyclic()
    (OUT / "basis_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
