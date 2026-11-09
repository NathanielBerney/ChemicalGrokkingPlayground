#!/usr/bin/env python3
"""Generate compact, graph-distinct all-carbon Hückel datasets.

Each family contains ``--count`` different molecular graphs. Distinctness is
guaranteed by a different even number of carbon atoms in every row. The input
graph is encoded by its family and n; use ``molecular_graph`` or
``hamiltonian_sparse`` to materialize it. Dense 40,000 x 40,000 matrices would
make an otherwise small dataset impractical, so only a small example matrix is
saved for each family.

The Möbius family is an anti-periodic *orbital* model. Its graph has a planar
embedding, but the orbital twist requires nonplanar geometry. No optimized 3D
geometries or experimental stability claims are made for any family.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


FAMILIES = ("circulant", "mobius", "linear")
DEFAULT_MIN_N = {"circulant": 6, "mobius": 16, "linear": 4}
STEP = {"circulant": 4, "mobius": 4, "linear": 2}
COLORS = {"circulant": "#2563eb", "mobius": "#c026d3", "linear": "#059669"}


def validate_size(family: str, n: int) -> None:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    if n < 4 or n % 2:
        raise ValueError("n must be even and at least 4")
    if family == "circulant" and n % 4 != 2:
        raise ValueError("Circulant neutral closed-shell rings require n = 4k+2")
    if family == "mobius" and n % 4 != 0:
        raise ValueError("Möbius neutral closed-shell rings require n = 4k")


def molecular_graph(family: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return bonds (i, j, order) and implicit hydrogen counts.

    The Kekulé assignment alternates C=C and C-C. Every carbon has total
    bond order four after adding its implicit hydrogens. Möbius sign is an
    orbital phase, not a negative chemical bond order.
    """
    validate_size(family, n)
    i = np.arange(n - 1, dtype=np.int32)
    bonds = np.column_stack((i, i + 1, np.where(i % 2 == 0, 2, 1)))
    if family != "linear":
        bonds = np.vstack((bonds, np.array([[n - 1, 0, 1]], dtype=np.int32)))
    bond_order_sum = np.zeros(n, dtype=np.int8)
    np.add.at(bond_order_sum, bonds[:, 0], bonds[:, 2])
    np.add.at(bond_order_sum, bonds[:, 1], bonds[:, 2])
    hydrogens = 4 - bond_order_sum
    if np.any(hydrogens < 0):
        raise AssertionError("Carbon valence exceeded")
    return bonds, hydrogens


def hamiltonian_sparse(
    family: str, n: int, alpha: float = 0.0, beta: float = -1.0
) -> sparse.csr_matrix:
    """Materialize the unpadded Hückel matrix in O(n) storage.

    H_ii = alpha, H_ij = beta on each graph edge. For Möbius, the closing
    edge gets -beta, giving one anti-periodic phase boundary.
    """
    bonds, _ = molecular_graph(family, n)
    if not math.isfinite(alpha) or not math.isfinite(beta) or beta >= 0:
        raise ValueError("alpha must be finite and beta must be finite and negative")
    row = bonds[:, 0]
    col = bonds[:, 1]
    values = np.full(len(bonds), beta, dtype=np.float64)
    if family == "mobius":
        values[-1] = -beta
    return sparse.coo_matrix(
        (np.r_[values, values, np.full(n, alpha)],
         (np.r_[row, col, np.arange(n)], np.r_[col, row, np.arange(n)])),
        shape=(n, n),
    ).tocsr()


def frontier(family: str, n: int, alpha: float = 0.0, beta: float = -1.0) -> tuple[float, float, float]:
    """Closed-form neutral, closed-shell HOMO, LUMO, and gap."""
    validate_size(family, n)
    if not math.isfinite(alpha) or not math.isfinite(beta) or beta >= 0:
        raise ValueError("alpha must be finite and beta must be finite and negative")
    half_gap = 2 * abs(beta) * math.sin(
        math.pi / (2 * (n + 1)) if family == "linear" else math.pi / n
    )
    return alpha - half_gap, alpha + half_gap, 2 * half_gap


def n_values(family: str, count: int, min_n: int | None = None) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    first = DEFAULT_MIN_N[family] if min_n is None else min_n
    validate_size(family, first)
    return first + STEP[family] * np.arange(count, dtype=np.int64)


def _row(family: str, n: int, alpha: float, beta: float) -> dict:
    homo, lumo, gap = frontier(family, n, alpha, beta)
    ring = family != "linear"
    return {
        "sample_id": f"{family}_{n:05d}",
        "family": family,
        "graph_key": f"{'cycle' if ring else 'path'}:{n}",
        "n_carbon": n,
        "n_hydrogen": n if ring else n + 2,
        "n_pi_electrons": n,
        "edge_count": n if ring else n - 1,
        "max_carbon_degree": 2,
        "max_carbon_bond_order_sum": 3,
        "carbon_valence_with_hydrogen": 4,
        "graph_planar": True,
        "planarity_constraint": "none" if family == "mobius" else "planar_graph",
        "geometry_verified": False,
        "orbital_twist": family == "mobius",
        "phase_flip_edge": f"{n-1}-0" if family == "mobius" else "",
        "alpha": alpha,
        "beta": beta,
        "homo_index_zero_based": n // 2 - 1,
        "lumo_index_zero_based": n // 2,
        "homo_energy": homo,
        "lumo_energy": lumo,
        "homo_lumo_gap": gap,
    }


def _draw_graph(ax, family: str, n: int) -> None:
    bonds, hydrogens = molecular_graph(family, n)
    if family == "linear":
        x = np.arange(n, dtype=float)
        y = 0.22 * np.where(np.arange(n) % 2 == 0, 1, -1)
    else:
        theta = 2 * np.pi * np.arange(n) / n + np.pi / 2
        x, y = np.cos(theta), np.sin(theta)
    for i, j, order in bonds:
        i, j = int(i), int(j)
        color = "#dc2626" if family == "mobius" and i == n - 1 else "#334155"
        style = "--" if family == "mobius" and i == n - 1 else "-"
        ax.plot([x[i], x[j]], [y[i], y[j]], color=color, lw=2, ls=style, zorder=1)
        if order == 2:
            vx, vy = x[j] - x[i], y[j] - y[i]
            length = math.hypot(vx, vy)
            dx, dy = -0.055 * vy / length, 0.055 * vx / length
            ax.plot([x[i] + dx, x[j] + dx], [y[i] + dy, y[j] + dy],
                    color=color, lw=1.5, zorder=1)
    ax.scatter(x, y, s=250, c="white", edgecolors=COLORS[family], lw=1.5, zorder=2)
    for i in range(n):
        ax.text(x[i], y[i], "C", ha="center", va="center", fontsize=8, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(float(np.min(x)) - 0.45, float(np.max(x)) + 0.45)
    ax.set_ylim(float(np.min(y)) - 0.35, float(np.max(y)) + 0.35)
    ax.axis("off")
    ax.set_title(f"Example: C{n}H{sum(hydrogens)}")
    if family == "mobius":
        ax.text(0.5, -0.07, "red seam = π phase flip; 2D sketch is not a 3D geometry",
                transform=ax.transAxes, ha="center", fontsize=8, color="#9f1239")


def plot_family(family: str, ns: np.ndarray, gap: np.ndarray, output: Path,
                alpha: float, beta: float, scale: str = "linear") -> None:
    if scale not in ("linear", "log10"):
        raise ValueError("scale must be 'linear' or 'log10'")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    example_n = {"circulant": 6, "mobius": 16, "linear": 8}[family]
    _draw_graph(axes[0], family, example_n)
    axes[1].plot(ns, gap, color=COLORS[family], lw=1.8)
    axes[1].set_xscale("log" if scale == "log10" else "linear")
    axes[1].set_yscale("log" if scale == "log10" else "linear")
    axes[1].set_xlabel("Carbon atoms, n")
    axes[1].set_ylabel("HOMO–LUMO gap (energy units)")
    axes[1].set_title("Gap versus graph size")
    axes[1].grid(alpha=0.25)
    axes[2].plot(ns, alpha - gap / 2, color="#1d4ed8", label="HOMO")
    axes[2].plot(ns, alpha + gap / 2, color="#ea580c", label="LUMO")
    axes[2].set_xscale("log" if scale == "log10" else "linear")
    axes[2].set_xlabel("Carbon atoms, n")
    axes[2].set_ylabel("Energy (same units as β)")
    axes[2].set_title("Frontier energies")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"{family.capitalize()} — {len(ns):,} distinct carbon graphs")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_comparison(data: dict, output: Path, scale: str = "linear") -> None:
    if scale not in ("linear", "log10"):
        raise ValueError("scale must be 'linear' or 'log10'")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    for family, (ns, gaps) in data.items():
        ax.plot(ns, gaps, label=family.capitalize(), color=COLORS[family], lw=1.8)
    ax.set_xscale("log" if scale == "log10" else "linear")
    ax.set_yscale("log" if scale == "log10" else "linear")
    ax.set_xlabel("Carbon atoms, n")
    ax.set_ylabel("HOMO–LUMO gap (energy units)")
    ax.set_title("Neutral all-carbon conjugated systems")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def generate_family(
    family: str, count: int, outdir: Path, min_n: int | None = None,
    alpha: float = 0.0, beta: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    ns = n_values(family, count, min_n)
    directory = outdir / family
    directory.mkdir(parents=True, exist_ok=True)
    rows = [_row(family, int(n), alpha, beta) for n in ns]
    with (directory / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    homo = np.array([r["homo_energy"] for r in rows], dtype=np.float64)
    lumo = np.array([r["lumo_energy"] for r in rows], dtype=np.float64)
    gaps = np.array([r["homo_lumo_gap"] for r in rows], dtype=np.float64)
    np.savez_compressed(directory / "dataset.npz", n=ns, homo_energy=homo,
                        lumo_energy=lumo, homo_lumo_gap=gaps,
                        alpha=np.array(alpha), beta=np.array(beta))
    example_n = {"circulant": 6, "mobius": 16, "linear": 8}[family]
    bonds, hydrogens = molecular_graph(family, example_n)
    np.savez_compressed(directory / "example.npz", n=np.array(example_n),
                        bonds=bonds, implicit_hydrogens=hydrogens,
                        hamiltonian=hamiltonian_sparse(family, example_n, alpha, beta).toarray())
    settings = {
        "family": family,
        "count": count,
        "n_min": int(ns[0]),
        "n_max": int(ns[-1]),
        "n_step": STEP[family],
        "alpha": alpha,
        "beta": beta,
        "graph_identity": "cycle:n" if family != "linear" else "path:n",
        "bond_order_rule": "edge (i,i+1) is double when i is even; ring closure (n-1,0) is single",
        "hydrogen_rule": "4 minus sum of C-C bond orders at each carbon",
        "phase_rule": "closing edge has -beta" if family == "mobius" else "all edges have beta",
        "graph_planar": True,
        "planarity_constraint": "none" if family == "mobius" else "planar_graph",
        "geometry_verified": False,
        "model_note": "Möbius orbital twist requires nonplanar geometry; 2D drawings are graph projections"
                      if family == "mobius" else "Graph planarity does not certify a low-energy 3D conformation",
        "stored_matrices": "small example only; use constrained_huckel.hamiltonian_sparse(family,n)",
    }
    (directory / "parameters.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    plot_family(family, ns, gaps, directory / "visualization.png", alpha, beta)
    plot_family(family, ns, gaps, directory / "visualization_log10.png", alpha, beta,
                scale="log10")
    print(f"{family}: {count:,} unique graphs, n={ns[0]:,}..{ns[-1]:,}; {directory}")
    return ns, gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10_000, help="Distinct molecular graphs per family")
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--outdir", type=Path, default=Path("constrained_huckel_datasets"))
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument("--circulant-min-n", type=int, default=DEFAULT_MIN_N["circulant"])
    parser.add_argument("--mobius-min-n", type=int, default=DEFAULT_MIN_N["mobius"])
    parser.add_argument("--linear-min-n", type=int, default=DEFAULT_MIN_N["linear"])
    args = parser.parse_args()
    minima = {"circulant": args.circulant_min_n,
              "mobius": args.mobius_min_n, "linear": args.linear_min_n}
    families = FAMILIES if args.family == "all" else (args.family,)
    data = {}
    for family in families:
        data[family] = generate_family(family, args.count, args.outdir,
                                       minima[family], args.alpha, args.beta)
    if len(data) > 1:
        plot_comparison(data, args.outdir / "comparison.png")
        plot_comparison(data, args.outdir / "comparison_log10.png", scale="log10")


if __name__ == "__main__":
    main()
