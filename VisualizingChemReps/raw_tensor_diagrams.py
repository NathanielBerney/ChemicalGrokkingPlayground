#!/usr/bin/env python3
"""Directly align numbered atoms with every raw numeric input tensor."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from non_ml_graph_story import load_example, numbered_structure
from visualize import BG, ELEMENTS, SITE_CLASSES, fmt

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RAW = HERE / "raw_arrays"
NAVY = "#193149"
MUTED = "#435e73"
GOLD = "#d99a28"


def make_canvas(title, subtitle, size=(23, 14)):
    fig = plt.figure(figsize=size, facecolor=BG)
    fig.text(.035, .972, title, ha="left", va="top", fontsize=23,
             weight="bold", color=NAVY)
    fig.text(.035, .925, subtitle, ha="left", va="top", fontsize=12.5,
             color=MUTED)
    return fig


def cell_color(value, kind, col=None):
    value = float(value)
    if abs(value) < 1e-7:
        return "#f4f7f9"
    if kind == "features":
        if col is not None and col < 6:
            return "#b8d9ed"
        if col is not None and col < 14:
            return "#c5e9d5"
        return "#fbe2ae" if value > 0 else "#f6cbd0"
    if kind == "bond":
        return {1: "#d9e9f4", 2: "#a9d0e7", 3: "#77b2d9"}[int(value)]
    if kind == "coupling" or kind == "huckel":
        return "#b9cceb" if value <= -.75 else "#d7dff1"
    return "#fbe2ae" if value > 0 else "#c8d9f0"


def numeric_grid(ax, array, row_labels, col_labels, title, kind="features",
                 highlight_rows=(), highlight_cells=(), group_headers=(),
                 font=8.8, row_font=10.3):
    rows, cols = array.shape
    ax.set_xlim(-1.35, cols + .5)
    ax.set_ylim(rows + .6, -2.0 if group_headers else -1.35)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=17, weight="bold", color=NAVY, pad=10)
    for j, label in enumerate(col_labels):
        ax.text(j + .5, -.66, str(label), ha="center", va="center",
                fontsize=font, color=NAVY, rotation=58 if cols > 12 else 0)
    for start, end, label in group_headers:
        ax.text((start + end + 1) / 2, -1.58, label, ha="center", va="center",
                fontsize=11.3, weight="bold", color=MUTED)
        ax.plot([start, end + 1], [-1.3, -1.3], color="#9fb6c9", lw=1.5)
    for i, label in enumerate(row_labels):
        color = GOLD if i in highlight_rows else NAVY
        ax.text(-.22, i + .5, str(label), ha="right", va="center", fontsize=row_font,
                color=color, weight="bold" if i in highlight_rows else "normal")
        for j in range(cols):
            value = array[i, j]
            ax.add_patch(Rectangle((j, i), 1, 1,
                         facecolor=cell_color(value, kind, j),
                         edgecolor="white", lw=.9))
            ax.text(j + .5, i + .5, fmt(value), ha="center", va="center",
                    fontsize=font, color=NAVY,
                    weight="bold" if abs(float(value)) > 1e-7 else "normal")
            if (i, j) in highlight_cells:
                ax.add_patch(Rectangle((j + .03, i + .03), .94, .94,
                             facecolor="none", edgecolor=GOLD, lw=2.8))
    for i in highlight_rows:
        ax.add_patch(Rectangle((0, i), cols, 1, facecolor="none", ec=GOLD, lw=2.6))


def text_panel(ax, title, lines):
    ax.axis("off")
    ax.text(0, 1, title, ha="left", va="top", fontsize=16.5,
            weight="bold", color=NAVY, transform=ax.transAxes)
    for i, line in enumerate(lines):
        ax.text(0, .84 - .095 * i, line, ha="left", va="top", fontsize=11.0,
                color=MUTED, transform=ax.transAxes)


def graph_tensors_figure(ex):
    fig = make_canvas("Numbered molecule → exact atom-graph input arrays",
        "Atom number i is row i of X. A bond between i and j fills cells (i,j) and (j,i) of B and C.")
    gs = fig.add_gridspec(2, 3, left=.035, right=.985, top=.85, bottom=.055,
                          width_ratios=[.86, 1.15, 1.15], height_ratios=[1.08, 1.0],
                          wspace=.18, hspace=.28)
    ax = fig.add_subplot(gs[0, 0]); numbered_structure(ax, ex, highlight=True)
    ax.set_title("Stored atom IDs (0–9)", loc="left", fontsize=17,
                 weight="bold", color=NAVY, pad=10)
    x = ex["node"]
    cols = ["C", "N", "O", "S", "F", "Cl", "C", "Np", "Nr", "Nn",
            "Oc", "Oe", "Oa", "St", "q/2", "deg/4", "e/2", "Hii/2"]
    numeric_grid(fig.add_subplot(gs[0, 1:]), x, [str(i) for i in range(ex["a"])], cols,
                 "X — 18 numbers for each atom (10 active rows shown)",
                 "features", highlight_rows=(2,),
                 group_headers=((0, 5, "element: one 1 among six"),
                                (6, 13, "π-site class: one 1 among eight"),
                                (14, 17, "four scaled values")), font=8.8)
    text_panel(fig.add_subplot(gs[1, 0]), "Same numbers, three arrays",
         ["X[i,:] = properties of atom i",
          "B[i,j] = chemical bond order",
          "C[i,j] = Hückel π coupling",
          "Gold: atom 2 and its edges to 1, 3",
          "All 0 entries are shown literally.",
          "Actual shapes: X 41×18; B,C 41×41",
          "First 10 atoms active; 31 padded rows",
          "Atom mask: 10 ones, then 31 zeros",
          f"Global counts: [{ex['aux'][0]:.2f}, {ex['aux'][1]:.2f}]" ])
    hcells = {(2, 1), (1, 2), (2, 3), (3, 2)}
    numeric_grid(fig.add_subplot(gs[1, 1]), ex["bond"],
                 [str(i) for i in range(ex["a"])], [str(i) for i in range(ex["a"])],
                 "B — bond order (0, 1, 2, or 3)", "bond",
                 highlight_cells=hcells, font=9.5)
    numeric_grid(fig.add_subplot(gs[1, 2]), ex["coupling"],
                 [str(i) for i in range(ex["a"])], [str(i) for i in range(ex["a"])],
                 "C — π coupling (H off-diagonal)", "coupling",
                 highlight_cells=hcells, font=9.5)
    return fig


def huckel_tensors_figure(ex):
    fig = make_canvas("Numbered molecule → exact π-graph and Hückel-row inputs",
        "For this molecule, π-site number j equals atom number j. The active 10-site blocks are shown; zero padding is explicit below.")
    gs = fig.add_gridspec(2, 3, left=.035, right=.985, top=.85, bottom=.055,
                          width_ratios=[.88, .85, 1.22], height_ratios=[1, 1.08],
                          wspace=.20, hspace=.28)
    ax = fig.add_subplot(gs[0, 0]); numbered_structure(ax, ex, highlight=True)
    ax.set_title("Stored π-site IDs (0–9)", loc="left", fontsize=17,
                 weight="bold", color=NAVY, pad=10)
    numeric_grid(fig.add_subplot(gs[0, 1]), ex["pi_features"],
                 [str(i) for i in range(ex["n"])],
                 ["Hjj/2", "eπ/2", "deg/4"],
                 "Xπ — 3 numbers per site", "other",
                 highlight_rows=(2,), font=10.5, row_font=11.5)
    hcells = {(2, 1), (1, 2), (2, 3), (3, 2)}
    numeric_grid(fig.add_subplot(gs[0, 2]), ex["H"],
                 [str(i) for i in range(ex["n"])], [str(i) for i in range(ex["n"])],
                 "H — site values and π edges", "huckel",
                 highlight_cells=hcells, font=9.6)
    text_panel(fig.add_subplot(gs[1, 0]), "The two numeric views",
         ["π graph: Xπ + H edge weights",
          "  Xπ is 40×3; H is 40×40",
          "H-row model: one row token per site",
          "  token = H row + eπ/2",
          "  actual token array is 40×41",
          "Only rows 0–9 are active here.",
          "H columns 10–39 are zero.",
          "Site mask: 10 ones, then 30 zeros",
          f"Global counts: [{ex['aux'][0]:.2f}, {ex['aux'][1]:.2f}]" ])
    active_rows = np.concatenate([ex["H"], ex["site_e"][:, None] / 2], axis=1)
    numeric_grid(fig.add_subplot(gs[1, 1:]), active_rows,
                 [str(i) for i in range(ex["n"])],
                 [f"H{j}" for j in range(ex["n"])] + ["eπ/2"],
                 "Matrix-row tokens — 10 active rows and the visible 11 of 41 columns",
                 "other", highlight_rows=(2,), font=9.4, row_font=10.5)
    return fig


def export_csvs():
    RAW.mkdir(exist_ok=True)
    with np.load(HERE / "selected_inputs.npz", allow_pickle=False) as d:
        i = int(np.where(d["source_indices"] == 498)[0][0])
        arrays = {
            "atom_X_41x18.csv": d["node"][i],
            "bond_B_41x41.csv": d["bond"][i],
            "pi_coupling_C_41x41.csv": d["pi_coupling"][i],
            "atom_mask_41.csv": d["atom_mask"][i].astype(np.uint8),
            "pi_X_40x3.csv": d["pi_graph_node_features"][i],
            "H_40x40.csv": d["H"][i],
            "matrix_rows_40x41.csv": d["matrix_row_tokens"][i],
            "site_mask_40.csv": d["site_mask"][i].astype(np.uint8),
            "aux_2.csv": d["aux"][i],
        }
    for name, arr in arrays.items():
        np.savetxt(RAW / name, np.atleast_2d(arr), delimiter=",", fmt="%.9g")
    (RAW / "README.md").write_text(
        "# Literal model inputs for dataset row 498\n\n"
        "Every CSV contains only numeric values; row and column indices start at 0. "
        "These arrays are copied or derived from `selected_inputs.npz` and written "
        "with nine significant digits, enough to recover their float32 values. "
        "For one-dimensional masks and `aux`, the values occupy one CSV row.\n\n"
        "`atom_X_41x18.csv` columns: "
        + ", ".join([f"element_{x}" for x in ELEMENTS]
                    + [f"site_class_{x}" for x in SITE_CLASSES]
                    + ["formal_charge/2", "atom_degree/4", "site_pi_electrons/2", "H_diagonal/2"])
        + ".\n\n"
        "`pi_X_40x3.csv` columns: H_diagonal/2, site_pi_electrons/2, "
        "nonzero_H_neighbor_count/4. `matrix_rows_40x41.csv` columns 0–39 "
        "are H row entries; column 40 is site_pi_electrons/2. "
        "The first 10 atoms/sites are active for this molecule. "
        "Zero padding is excluded from the models by the masks.\n")


def main():
    FIGURES.mkdir(exist_ok=True)
    ex = load_example()
    for name, fig in [
        ("numbered_to_atom_graph_tensors", graph_tensors_figure(ex)),
        ("numbered_to_pi_and_matrix_tensors", huckel_tensors_figure(ex)),
    ]:
        fig.savefig(FIGURES / (name + ".png"), dpi=180, bbox_inches="tight", facecolor=BG)
        fig.savefig(FIGURES / (name + ".svg"), bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    export_csvs()
    print("Wrote exact numeric tensor figures and CSVs to", HERE)


if __name__ == "__main__":
    main()
