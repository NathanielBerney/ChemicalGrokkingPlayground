#!/usr/bin/env python3
"""Three presentation pages explaining graph encoding to a chemistry audience."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from visualize import (BG, PROBE, SOURCE, ELEMENT_COLOR, centered_coords,
                       draw_bond, make_example, read_sdf_selected)

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
EXAMPLE_INDEX = 498
NAVY = "#193149"
MUTED = "#486177"
GOLD = "#e6a23a"
TEAL = "#269db7"


def load_example():
    vocab = json.loads((PROBE / "vocab.json").read_text())
    sdf = read_sdf_selected(SOURCE / "molecules.sdf", {EXAMPLE_INDEX})[EXAMPLE_INDEX]
    with np.load(PROBE / "features.npz", allow_pickle=False) as feat, \
         np.load(SOURCE / "dataset.npz", allow_pickle=False) as source:
        ex = make_example(EXAMPLE_INDEX, "diazine_nitrile_aldehyde",
                          "Nitrile/aldehyde diazine", feat, source, vocab, sdf)
    assert ex["smiles"] == "N#Cc1ncncc1C=O"
    assert ex["a"] == 10 and ex["n"] == 10 and len(ex["sdf"]["bonds"]) == 10
    assert ex["sdf"]["elements"][2] == "N" and ex["site_classes"][2] == "N_pyridine"
    assert ex["bond"][2, 1] == 2 and ex["bond"][2, 3] == 1
    assert np.isclose(ex["coupling"][2, 1], -1) and np.isclose(ex["coupling"][2, 3], -1)
    return ex


def base_figure(title, subtitle, size=(17, 10.5)):
    fig = plt.figure(figsize=size, facecolor=BG)
    fig.subplots_adjust(left=.045, right=.975, top=.825, bottom=.055,
                        wspace=.10, hspace=.20)
    fig.text(.045, .965, title, ha="left", va="top", fontsize=21,
             fontweight="bold", color=NAVY)
    fig.text(.045, .905, subtitle, ha="left", va="top", fontsize=12, color=MUTED)
    return fig


def numbered_structure(ax, ex, highlight=False):
    """Use the user's numbered-skeleton convention, not atom-name labels."""
    pos = centered_coords(ex["sdf"]["coords"])
    for i, j, order in ex["sdf"]["bonds"]:
        draw_bond(ax, pos[i], pos[j], order=order, color="#253443", width=2.2)
    for i, element in enumerate(ex["sdf"]["elements"]):
        if highlight and i in (1, 2, 3):
            ring_color = GOLD if i == 2 else TEAL
            ax.add_patch(Circle(pos[i], .21, fill=False, ec=ring_color, lw=3.0, zorder=3))
        ax.text(*pos[i], str(i), ha="center", va="center", fontsize=20,
                fontweight="bold", color=ELEMENT_COLOR.get(element, NAVY), zorder=4,
                bbox=dict(boxstyle="round,pad=.055", fc=BG, ec="none"))
    ax.set_aspect("equal")
    ax.set_xlim(pos[:, 0].min() - .45, pos[:, 0].max() + .45)
    ax.set_ylim(pos[:, 1].min() - .45, pos[:, 1].max() + .45)
    ax.axis("off")


def card(ax, title, body, face="#ffffff", fontsize=11.5):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((.02, .04), .96, .9,
                 boxstyle="round,pad=.012,rounding_size=.025",
                 fc=face, ec="#c0d0dc", lw=1.2))
    ax.text(.07, .85, title, va="top", fontsize=13, fontweight="bold", color=NAVY)
    for i, line in enumerate(body):
        ax.text(.07, .68 - .145 * i, line, va="top", fontsize=fontsize, color=MUTED)


def simple_table(ax, title, col_labels, rows, highlights=()):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=NAVY, pad=10)
    t = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                 colLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11.5)
    t.scale(1, 1.65)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d5e0e8")
        if r == 0:
            cell.set_facecolor("#dcebf4")
            cell.set_text_props(weight="bold", color=NAVY)
        elif r - 1 in highlights:
            cell.set_facecolor("#fff0cf")
        else:
            cell.set_facecolor("white" if r % 2 else "#f1f6fa")
    return t


def takeaway(ax, text):
    ax.axis("off")
    ax.text(.02, .55, text, va="center", ha="left", fontsize=12.5,
            color=NAVY, fontweight="bold", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.55", fc="#e7f1f8", ec="#bed1de"))


def page_1(ex):
    fig = base_figure("1  A numbered structure becomes two lists",
        "A graph is a list of atoms plus a list of connections. The drawing coordinates only help us see it.")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, .16], width_ratios=[1.2, 1.05, 1.35])
    ax = fig.add_subplot(gs[0, 0]); numbered_structure(ax, ex)
    ax.set_title("Numbered chemical structure", loc="left", fontsize=15,
                 fontweight="bold", color=NAVY, pad=10)
    atoms = []
    for i, element in enumerate(ex["sdf"]["elements"]):
        atoms.append([str(i), element, str(int((ex["bond"][i] > 0).sum()))])
    simple_table(fig.add_subplot(gs[0, 1]), "Atom list", ["ID", "element", "neighbors"], atoms,
                 highlights=(1, 2, 3))
    bond_rows = []
    highlighted = []
    names = {1: "single", 2: "double", 3: "triple"}
    for k, (i, j, order) in enumerate(ex["sdf"]["bonds"]):
        bond_rows.append([f"{i}–{j}", names[order], f"{float(ex['coupling'][i, j]):.1f}"])
        if 2 in (i, j):
            highlighted.append(k)
    simple_table(fig.add_subplot(gs[0, 2]), "Connection list",
                 ["atoms", "chemical bond", "π coupling"], bond_rows, highlighted)
    takeaway(fig.add_subplot(gs[1, :]),
             "The model gets atom and connection data. It does not get the x–y positions in this picture.")
    return fig


def page_2(ex):
    fig = base_figure("2  Each atom and connection becomes numbers",
        "Example: the highlighted ring nitrogen (atom 2) and its two direct neighbors.")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, .20], width_ratios=[1, 1.3, 1.15])
    ax = fig.add_subplot(gs[0:2, 0]); numbered_structure(ax, ex, highlight=True)
    ax.set_title("Gold: atom 2   •   teal: neighbors 1 and 3", loc="left",
                 fontsize=13, fontweight="bold", color=NAVY, pad=10)
    flags_element = " ".join(str(int(x)) for x in ex["node"][2, :6])
    flags_class = " ".join(str(int(x)) for x in ex["node"][2, 6:14])
    numeric = ex["node"][2, 14:18]
    card(fig.add_subplot(gs[0, 1]), "Atom 2: category flags",
         ["Element is N:",
          "C  N  O  S  F  Cl",
          flags_element,
          "π class is N_pyridine:",
          flags_class], "#edf5fa", 11.5)
    card(fig.add_subplot(gs[0, 2]), "Atom 2: four numerical values",
         [f"formal charge / 2    =  {numeric[0]:.2g}",
          f"neighbor count / 4  =  {numeric[1]:.2g}",
          f"π electrons / 2      =  {numeric[2]:.2g}",
          f"Hückel site value / 2 = {numeric[3]:.2g}",
          "Together: 6 + 8 + 4 = 18 numbers"], "#eef5e9", 11.0)
    rows = [
        ["1–2", f"double = {int(ex['bond'][1, 2])}", f"{float(ex['coupling'][1, 2]):.1f}"],
        ["2–3", f"single = {int(ex['bond'][2, 3])}", f"{float(ex['coupling'][2, 3]):.1f}"],
    ]
    simple_table(fig.add_subplot(gs[1, 1]), "Atom 2's two connections",
                 ["atom pair", "bond order", "π coupling"], rows, highlights=(0, 1))
    card(fig.add_subplot(gs[1, 2]), "Repeat for the whole molecule",
         ["10 active atoms → 10 rows × 18 numbers",
          "10 bonds → two connection grids:",
          "bond orders and π couplings",
          "Arrays are padded to 41 atoms;",
          "a mask hides the unused rows."], "#f7f0e7", 11.1)
    takeaway(fig.add_subplot(gs[2, :]),
             "A category flag is a 1 for the matching category and 0 for the others; these are inputs, not predictions.")
    return fig


def node(ax, xy, label, fill, radius=.26):
    ax.add_patch(Circle(xy, radius, fc=fill, ec="white", lw=2, zorder=3))
    ax.text(*xy, label, ha="center", va="center", fontsize=14, weight="bold",
            color="white", zorder=4)


def page_3(ex):
    fig = base_figure("3  The model shares information along those connections",
        "One update around atom 2, then the same operation across every atom and four rounds.")
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.15, .84, .35],
                  width_ratios=[1, 1, 1, 1])
    ax = fig.add_subplot(gs[0, 0:2]); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("A local update at atom 2", loc="left", fontsize=15,
                 fontweight="bold", color=NAVY, pad=9)
    node(ax, (1.7, 3.6), "1", "#25364b", .38)
    node(ax, (1.7, 1.2), "3", "#25364b", .38)
    node(ax, (7.1, 2.4), "2", "#2563a8", .48)
    for y in (3.6, 1.2):
        ax.add_patch(FancyArrowPatch((2.25, y), (6.45, 2.4), arrowstyle="-|>",
                     mutation_scale=18, lw=2.4, color=TEAL))
    ax.text(4.0, 4.12, f"atom 1: double bond; π coupling {float(ex['coupling'][2, 1]):.1f}", fontsize=11,
            color=NAVY, ha="center")
    ax.text(4.0, .38, f"atom 3: single bond; π coupling {float(ex['coupling'][2, 3]):.1f}", fontsize=11,
            color=NAVY, ha="center")
    card(fig.add_subplot(gs[0, 2:4]), "What the model learns to do",
         ["Read atom 2's own 18-number description.",
          "Read the descriptions of its neighbors 1 and 3.",
          "Treat single and double bonds differently.",
          "Use the π-coupling values as edge weights.",
          "Write an updated description for atom 2."], "#eaf4ee", 11.8)
    stages = [
        ("1  Translate", "18 input numbers per atom\nbecome a 96-number\nworking description"),
        ("2  Exchange", "Every atom combines its\nown information with\nconnected neighbors"),
        ("3  Repeat ×4", "After four rounds, an atom\ncan reflect information\nfrom up to four bonds away"),
        ("4  Predict", "Summarize all atoms into\none molecule description;\npredict the HOMO–LUMO gap"),
    ]
    for col, (title, body) in enumerate(stages):
        card(fig.add_subplot(gs[1, col]), title, body.split("\n"),
             "#edf4f9" if col % 2 == 0 else "#f3f7ed", 11.0)
    takeaway(fig.add_subplot(gs[2, :]),
             "The π-only graph follows the same idea using Hückel couplings, three numbers per site, and six rounds."
             " Neither graph model uses transformer attention.")
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    ex = load_example()
    pages = [page_1(ex), page_2(ex), page_3(ex)]
    names = ("graph_story_1_numbered_to_lists",
             "graph_story_2_lists_to_numbers",
             "graph_story_3_neighbors_to_gap")
    with PdfPages(OUT / "graph_story_for_non_ml_audience.pdf") as pdf:
        for fig, name in zip(pages, names):
            pdf.savefig(fig, facecolor=BG)
            fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight", facecolor=BG)
            fig.savefig(OUT / (name + ".svg"), bbox_inches="tight", facecolor=BG)
            plt.close(fig)
    print("Wrote 3 PNGs, 3 SVGs, and one PDF to", OUT)


if __name__ == "__main__":
    main()
