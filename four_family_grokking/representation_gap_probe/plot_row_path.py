#!/usr/bin/env python3
"""Show how the same Hückel Hamiltonian enters row attention and π-graph GNN."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent / "long_100k_seed0" / "plots"


def box(ax, x, y, text, color, width=2.15):
    rect = FancyBboxPatch((x, y), width, .76, boxstyle="round,pad=0.12",
                          linewidth=1.4, edgecolor=color, facecolor="white")
    ax.add_patch(rect)
    ax.text(x + width / 2, y + .38, text, ha="center", va="center", fontsize=10)


def arrow(ax, x1, x2, y, color):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.5, color=color))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-.25, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    top = ["Hückel matrix\nH (40×40)", "Rows H[i, :]\nfixed column indices",
           "Linear row map\n+ row position", "One attention layer\nCLS reads raw rows",
           "CLS readout\n→ gap"]
    lower = ["Same H and\nπ occupancy", "Sites + H[i,j]\nas weighted edges",
             "Six rounds of\nneighbor messages", "Permutation-invariant\ngraph pool",
             "Graph readout\n→ gap"]
    xs = [0, 2.6, 5.2, 7.8, 10.4]
    for y, labels, color in ((2.85, top, "#6f42a6"), (1.0, lower, "#cc7000")):
        for j, (x, label) in enumerate(zip(xs, labels)):
            box(ax, x, y, label, color)
            if j:
                arrow(ax, xs[j - 1] + 2.18, x - .06, y + .38, color)
    ax.text(0, 3.82, "Matrix row attention", fontsize=13, weight="bold", color="#6f42a6")
    ax.text(0, 1.97, "π-graph GNN", fontsize=13, weight="bold", color="#cc7000")
    ax.text(0, .35, "Equivalent site renumbering permutes both axes of H."
            " The row model changes prediction; graph pooling preserves it.", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "row_vs_graph_architecture.svg")
    fig.savefig(ROOT / "row_vs_graph_architecture.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
