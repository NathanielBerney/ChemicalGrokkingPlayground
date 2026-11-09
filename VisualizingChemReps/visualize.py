#!/usr/bin/env python3
"""Visualize exact molecular encodings used by representation_gap_probe.

Uses the stored SDF coordinates and prepared arrays; no RDKit installation is
needed. Run from anywhere with: python VisualizingChemReps/visualize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROBE = ROOT / "four_family_grokking" / "representation_gap_probe"
SOURCE = ROOT / "heteroatom_huckel_dataset"
sys.path.insert(0, str(PROBE))
from prepare import ELEMENTS, SITE_CLASSES, TOKEN_RE  # noqa: E402

EXAMPLES = (
    (112, "methylfuran", "2-methylfuran"),
    (122, "pyridine", "Pyridine"),
    (2, "aniline", "Aniline"),
    (498, "diazine_nitrile_aldehyde", "Nitrile/aldehyde diazine"),
)
ELEMENT_COLOR = {"C": "#25364b", "N": "#2563a8", "O": "#d14952", "S": "#bd8b1e", "F": "#24856d", "Cl": "#24856d"}
BG = "#f7fafc"


def read_sdf_selected(path: Path, wanted: set[int]) -> dict[int, dict]:
    found = {}
    with path.open() as handle:
        block = []
        index = 0
        for line in handle:
            if line.strip() != "$$$$":
                block.append(line)
                continue
            if index in wanted:
                na = int(block[3][:3])
                nb = int(block[3][3:6])
                atoms = block[4:4 + na]
                bond_lines = block[4 + na:4 + na + nb]
                found[index] = {
                    "coords": np.array([(float(x[:10]), float(x[10:20])) for x in atoms]),
                    "elements": [x[31:34].strip() for x in atoms],
                    "bonds": [(int(x[:3]) - 1, int(x[3:6]) - 1, int(x[6:9])) for x in bond_lines],
                }
            index += 1
            block = []
            if len(found) == len(wanted):
                break
    assert set(found) == wanted
    return found


def make_example(index: int, slug: str, title: str, feat, source, vocab: dict, sdf: dict) -> dict:
    n = int(feat["site_counts"][index])
    a = int(feat["atom_counts"][index])
    smiles = str(source["smiles"][index])
    tokens = ["CLS", *TOKEN_RE.findall(smiles)]
    ids = feat["smiles_ids"][index, :len(tokens)].astype(int)
    assert "".join(tokens[1:]) == smiles
    assert ids.tolist() == [vocab[t] for t in tokens]
    assert len(sdf["elements"]) == a
    assert np.array_equal(feat["site_mask"][index], np.arange(40) < n)
    assert np.allclose(feat["H"][index],
                       np.nan_to_num(source["matrices"][index], nan=0.0), atol=1e-7)
    assert np.all(feat["H"][index, n:, :] == 0) and np.all(feat["H"][index, :, n:] == 0)
    atom_ids = source["site_atom_indices"][index, :n].astype(int)
    classes = [str(x) for x in source["site_classes"][index, :n]]
    H = feat["H"][index, :n, :n].copy()
    site_e = feat["site_pi_electrons"][index, :n].copy()
    bond = feat["bond"][index, :a, :a].copy()
    coupling = feat["pi_coupling"][index, :a, :a].copy()
    node = feat["node"][index, :a].copy()
    assert np.array_equal(bond, bond.T)
    assert np.allclose(coupling, coupling.T)
    assert all(int(bond[i, j]) == order for i, j, order in sdf["bonds"])
    assert np.allclose(coupling[np.ix_(atom_ids, atom_ids)], H - np.diag(np.diag(H)))
    diagonal = np.diag(H)
    degree = (H - np.diag(diagonal) != 0).sum(axis=1) / 4
    pi_features = np.stack([diagonal / 2, site_e / 2, degree], axis=1)
    assert np.allclose(node[atom_ids, 16], site_e / 2)
    assert np.allclose(node[atom_ids, 17], diagonal / 2)
    assert np.allclose(node[:, 15], (bond > 0).sum(axis=1) / 4)
    aux = np.array([int(feat["pi_electrons"][index]) / 40, n / 40], dtype=np.float32)
    row_input = np.concatenate([feat["H"][index, :n], (site_e / 2)[:, None]], axis=1)
    return dict(index=index, slug=slug, title=title, smiles=smiles, tokens=tokens,
                ids=ids, n=n, a=a, pi_e=int(feat["pi_electrons"][index]),
                gap=float(feat["gap"][index]), site_atoms=atom_ids,
                site_classes=classes, site_e=site_e, H=H, bond=bond,
                coupling=coupling, node=node, pi_features=pi_features,
                row_input=row_input, aux=aux, sdf=sdf)


def setup_figure(width: float, height: float, title: str):
    fig = plt.figure(figsize=(width, height), facecolor=BG, layout="constrained")
    fig.suptitle(title, fontsize=19, fontweight="bold", color="#142438")
    return fig


def panel_title(ax, title: str, subtitle: str = ""):
    ax.set_title(title + ("\n" + subtitle if subtitle else ""), loc="left", fontsize=11,
                 fontweight="bold", color="#142438", pad=11)


def centered_coords(coords):
    p = coords.astype(float).copy()
    p -= p.mean(axis=0)
    span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), .1)
    p *= 2.4 / span
    return p


def draw_bond(ax, p, q, order=1, color="#75879b", width=2.0):
    v = q - p
    norm = np.linalg.norm(v)
    offset = np.array([-v[1], v[0]]) / max(norm, 1e-8) * .045
    if order == 1:
        shifts = [np.zeros(2)]
    elif order == 2:
        shifts = [-offset, offset]
    else:
        shifts = [-1.5 * offset, np.zeros(2), 1.5 * offset]
    for shift in shifts:
        ax.plot([p[0] + shift[0], q[0] + shift[0]],
                [p[1] + shift[1], q[1] + shift[1]], color=color,
                lw=width, solid_capstyle="round", zorder=1)


def draw_molecule(ax, ex, mode="chemical"):
    sdf = ex["sdf"]
    p = centered_coords(sdf["coords"])
    site_by_atom = {int(a): j for j, a in enumerate(ex["site_atoms"])}
    if mode == "pi":
        for j in range(ex["n"]):
            for k in range(j + 1, ex["n"]):
                if ex["H"][j, k] != 0:
                    draw_bond(ax, p[ex["site_atoms"][j]], p[ex["site_atoms"][k]],
                              1, "#7993a7", 1.4 + abs(float(ex["H"][j, k])) * 1.5)
        for j, a in enumerate(ex["site_atoms"]):
            element = sdf["elements"][a]
            ax.add_patch(Circle(p[a], .21, facecolor=ELEMENT_COLOR.get(element, "#25364b"),
                                edgecolor="white", linewidth=1.2, zorder=3))
            ax.text(*p[a], str(j), ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold", zorder=4)
    else:
        for i, j, order in sdf["bonds"]:
            draw_bond(ax, p[i], p[j], order,
                      "#708298", 1.9 if mode == "chemical" else 2.1)
        for i, element in enumerate(sdf["elements"]):
            color = ELEMENT_COLOR.get(element, "#25364b")
            if mode == "chemical":
                ax.text(*p[i], f"{element}{i}", ha="center", va="center", fontsize=10,
                        color=color, fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="round,pad=.14", fc="white", ec="none", alpha=.97))
            else:
                is_pi = i in site_by_atom
                ax.add_patch(Circle(p[i], .23, facecolor=color if is_pi else "#9aa9b8",
                                    edgecolor="white", linewidth=1.3, zorder=3))
                ax.text(*p[i], f"{element}{i}", ha="center", va="center", color="white",
                        fontsize=8.2, fontweight="bold", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(p[:, 0].min() - .5, p[:, 0].max() + .5)
    ax.set_ylim(p[:, 1].min() - .5, p[:, 1].max() + .5)
    ax.axis("off")


def matrix_panel(ax, array, labels, title, vmin=None, vmax=None, cmap="Blues", fmt=".2g"):
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            value = float(array[i, j])
            if abs(value) > 1e-7:
                ax.text(j, i, format(value, fmt), ha="center", va="center",
                        fontsize=7 if len(labels) > 8 else 8, color="#15263b", fontweight="bold")
    panel_title(ax, title)


def tokens_panel(ax, ex):
    ax.axis("off")
    panel_title(ax, "Canonical SMILES → exact token IDs", "The transformer sees IDs, then learned embeddings + positions")
    ax.text(.02, .82, ex["smiles"], fontfamily="monospace", fontsize=16,
            color="#15263b", transform=ax.transAxes)
    tokens = ex["tokens"]
    ids = ex["ids"]
    cols = max(8, min(10, len(tokens)))
    for j, (token, id_) in enumerate(zip(tokens, ids)):
        row, col = divmod(j, cols)
        x = .02 + col * (.96 / cols)
        y = .58 - row * .21
        ax.add_patch(FancyBboxPatch((x, y), .88 / cols, .15,
                                    boxstyle="round,pad=.01,rounding_size=.01",
                                    fc="#e8f1f7" if j else "#d9e8f7", ec="#b9cad8",
                                    transform=ax.transAxes))
        ax.text(x + .44 / cols, y + .10, token, ha="center", va="center",
                fontfamily="monospace", fontsize=8.6, transform=ax.transAxes)
        ax.text(x + .44 / cols, y + .035, str(id_), ha="center", va="center",
                fontsize=8, color="#49637c", transform=ax.transAxes)
    ax.text(.02, .05, "ID 0 = PAD; remaining positions to length 68 are masked.",
            fontsize=9, color="#4c6073", transform=ax.transAxes)


def plot_views(ex, outdir):
    fig = setup_figure(16, 11,
        f"{ex['title']}  |  row {ex['index']}  |  {ex['smiles']}\n"
        f"{ex['pi_e']} π electrons  •  {ex['n']} π sites  •  {ex['a']} atoms  •  gap {ex['gap']:.4f} |β|")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, 0]); draw_molecule(ax, ex, "chemical")
    panel_title(ax, "Chemical structure (SDF 2D)", "Atom labels give the stored atom order")
    ax = fig.add_subplot(gs[0, 1]); draw_molecule(ax, ex, "full")
    panel_title(ax, "Full atom graph", "Nodes = atoms; parallel lines show bond order")
    ax = fig.add_subplot(gs[0, 2]); draw_molecule(ax, ex, "pi")
    panel_title(ax, "π-site weighted graph", "Site IDs follow the Hückel matrix; edges = H[j,k]")
    labels = [str(j) for j in range(ex["n"])]
    matrix_panel(fig.add_subplot(gs[1, 0]), ex["H"], labels,
                 f"Hückel H: active {ex['n']}×{ex['n']} block", -2, 0, "Blues_r")
    matrix_panel(fig.add_subplot(gs[1, 1]), ex["bond"],
                 [str(j) for j in range(ex["a"])], "Full graph bond-order matrix", 0, 3, "Blues")
    tokens_panel(fig.add_subplot(gs[1, 2]), ex)
    path = outdir / f"{ex['slug']}_views.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def table_panel(ax, rows, columns, title, subtitle="", font=8.3, widths=None):
    ax.axis("off")
    panel_title(ax, title, subtitle)
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center",
                     colLoc="center", loc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(font)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d7e1e9")
        cell.set_facecolor("#dceaf4" if r == 0 else "white" if r % 2 else "#f2f7fa")
        if r == 0:
            cell.set_text_props(weight="bold", color="#17344d")


def fmt(x):
    if abs(float(x)) < 5e-7:
        return "0"
    return f"{float(x):.3g}"


def plot_encodings(ex, outdir):
    fig = setup_figure(17.5, 14, f"Exact model inputs  |  {ex['title']}  |  {ex['smiles']}")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.17, 1.25, .58],
                  width_ratios=[1.16, 1])
    n = ex["n"]
    arr = np.concatenate([ex["H"], ex["site_e"][:, None] / 2], axis=1)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(arr, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(n + 1), [f"H{k}" for k in range(n)] + ["e/2"], rotation=45, fontsize=8)
    ax.set_yticks(range(n), [f"site {j}" for j in range(n)], fontsize=8)
    ax.tick_params(length=0)
    for j in range(n):
        for k in range(n + 1):
            if abs(float(arr[j, k])) > 1e-7:
                ax.text(k, j, fmt(arr[j, k]), ha="center", va="center", fontsize=7.5, color="#172d42")
    panel_title(ax, "Matrix row tokens: active columns + occupancy",
                "Actual width 41 = H[j,0:40] || eπ,j/2; omitted H columns are zero")
    pi_rows = []
    for j, a in enumerate(ex["site_atoms"]):
        pi_rows.append([str(j), f"{ex['sdf']['elements'][a]}{a}", ex["site_classes"][j],
                        *[fmt(x) for x in ex["pi_features"][j]]])
    table_panel(fig.add_subplot(gs[0, 1]), pi_rows,
                ["π site", "atom", "class", "Hjj/2", "eπ/2", "degree/4"],
                "π-graph node features",
                "Edges carry H[j,k] off diagonal; six message-passing layers",
                font=8.0, widths=[.09, .09, .29, .14, .14, .17])
    node_rows = []
    for a in range(ex["a"]):
        f = ex["node"][a]
        element = ELEMENTS[int(np.argmax(f[:6]))]
        site_class = SITE_CLASSES[int(np.argmax(f[6:14]))] if np.any(f[6:14]) else "—"
        node_rows.append([str(a), element, site_class, *[fmt(x) for x in f[14:18]]])
    table_panel(fig.add_subplot(gs[1, 0]), node_rows,
                ["atom", "element", "π class", "charge/2", "degree/4", "eπ/2", "Hjj/2"],
                "Full atom-graph node features (18 numbers per atom)",
                "Element = 6 one-hot entries; π class = 8 one-hot entries; final four scalars shown",
                font=8.3, widths=[.07, .12, .25, .13, .13, .13, .13])
    matrix_panel(fig.add_subplot(gs[1, 1]), ex["coupling"],
                 [str(j) for j in range(ex["a"])],
                 "Full graph π-coupling edge matrix", -1.2, 1.2, "coolwarm")
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    panel_title(ax, "Sequence, padding, masks, and global inputs")
    token_pairs = "  ".join(f"{t}:{int(i)}" for t, i in zip(ex["tokens"], ex["ids"]))
    ax.text(.01, .76, "SMILES tokens → IDs:  " + token_pairs,
            fontfamily="monospace", fontsize=10, transform=ax.transAxes)
    ax.text(.01, .55, f"aux = [nπ/40, nsites/40] = [{ex['aux'][0]:.3f}, {ex['aux'][1]:.3f}]",
            fontfamily="monospace", fontsize=10, transform=ax.transAxes)
    ax.text(.01, .34,
            f"matrix H: 40×40; site mask: {n} true / {40 - n} false;  "
            f"atom graph: 41 nodes max, {ex['a']} active;  SMILES: 68 tokens max, {len(ex['ids'])} active",
            fontfamily="monospace", fontsize=9.7, transform=ax.transAxes)
    ax.text(.01, .13,
            "Learned embeddings and attention/message-passing activations depend on trained weights; "
            "this page shows the deterministic input encoding.", fontsize=9.4, color="#49637c",
            transform=ax.transAxes)
    path = outdir / f"{ex['slug']}_encodings.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def plot_encoder_paths(outdir):
    fig, ax = plt.subplots(figsize=(17, 8.5), facecolor=BG, layout="constrained")
    ax.set_xlim(0, 16.7)
    ax.set_ylim(-.35, 5.35)
    ax.axis("off")
    ax.set_title("How the four prepared views enter the gap models", loc="left",
                 fontsize=19, fontweight="bold", color="#142438", pad=16)
    columns = ("Deterministic input", "Learned input map", "Encoder", "Readout → gap")
    left = [0, 4.2, 8.4, 12.6]
    for x, heading in zip(left, columns):
        ax.text(x + 1.8, 4.88, heading, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#304a61")
    rows = [
        ("H rows", "H: 40×40 + eπ/2 per row\nsite mask; aux = [nπ,nsite]/40",
         "41 → 128 linear\nCLS + learned row positions",
         "1 attention layer, 4 heads\nresidual + feed-forward",
         "CLS || aux → MLP\nlog gap regression"),
        ("π graph", "3 site features: Hjj/2, eπ/2, degree/4\nweighted edges Hjk; aux",
         "3 → 96 linear\nshared across sites",
         "6 coupling-weighted\nmessage-passing layers",
         "mean + max || aux → MLP\nlog gap regression"),
        ("Atom graph", "18 atom features; bond orders 1–3\nπ coupling on atom edges; aux",
         "18 → 96 linear\nshared across atoms",
         "4 bond/coupling\nmessage-passing layers",
         "mean + max || aux → MLP\nlog gap regression"),
        ("SMILES", "CLS + lexical token IDs, PAD=0\nlength 68; aux",
         "ID → 96 embedding\nlearned token positions",
         "2 transformer layers\n4 attention heads",
         "CLS || aux → MLP\nlog gap regression"),
    ]
    colors = ["#e5eff9", "#e7f4ee", "#f7eee3", "#f0e9f8"]
    for row, (label, *cells) in enumerate(rows):
        y = 3.52 - row * 1.16
        ax.text(-.08, y + .37, label, ha="right", va="center", fontsize=11,
                fontweight="bold", color="#1e344b")
        for col, (x, body) in enumerate(zip(left, cells)):
            box = FancyBboxPatch((x, y), 3.64, .76,
                                 boxstyle="round,pad=.03,rounding_size=.12",
                                 fc=colors[row], ec="#aebfd0", lw=1.1)
            ax.add_patch(box)
            ax.text(x + 1.82, y + .38, body, ha="center", va="center",
                    fontsize=9.3, color="#183047")
            if col < 3:
                ax.annotate("", xy=(left[col + 1] - .12, y + .38),
                            xytext=(x + 3.69, y + .38),
                            arrowprops=dict(arrowstyle="->", lw=1.6, color="#647e94"))
    ax.text(0, -.2,
            "All four predict the same standardized log(HOMO–LUMO gap). "
            "The H-row and π-graph models receive the same H/occupancy information; "
            "the atom graph also receives the σ skeleton.",
            fontsize=10.4, color="#405a70")
    path = outdir / "encoder_paths.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def plot_graph_walkthrough(ex, outdir):
    """A visual counterpart to SMILES-token diagrams for the two graph models."""
    fig = setup_figure(17.5, 11.4,
        f"From molecule to graph inputs  |  {ex['smiles']}  |  dataset row {ex['index']}")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[.88, 1.12, 1.12],
                  width_ratios=[1, 1.35, 1.35])

    def card(ax, heading, lines, face="#ffffff", fontsize=10.5):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((.015, .035), .97, .9,
                    boxstyle="round,pad=.008,rounding_size=.025",
                    fc=face, ec="#b8c9d7", lw=1.3))
        ax.text(.05, .86, heading, fontsize=12, fontweight="bold",
                color="#163149", va="top", transform=ax.transAxes)
        for j, line in enumerate(lines):
            ax.text(.055, .72 - j * .145, line, fontsize=fontsize, color="#244158",
                    va="top", fontfamily="monospace", transform=ax.transAxes)

    ax = fig.add_subplot(gs[0, 0]); draw_molecule(ax, ex, "chemical")
    panel_title(ax, "1  Start with the molecule", "SDF atom order gives the node indices")
    card(fig.add_subplot(gs[0, 1]), "2  Assign graph IDs",
         ["atom IDs:  0 1 2 3 4 5 6 7 8 9",
          "π-site IDs: 0 1 2 3 4 5 6 7 8 9",
          "Here all 10 atoms are π sites.",
          "For methylfuran, the methyl C is not."], "#edf4f9", 11)
    card(fig.add_subplot(gs[0, 2]), "3  Two views of the same structure",
         ["Atom graph: X_atom + B + C",
          "π graph:   X_π + off-diagonal H",
          "N2 is highlighted below; neighbors C1,C3.",
          "No Q/K attention in either graph model."], "#edf4f9", 10.8)

    for row, mode, heading, subtitle in [
        (1, "full", "Full atom graph", "Nodes are atoms; B and C are edge tensors"),
        (2, "pi", "π-site graph", "Nodes are π orbitals; H is the weighted edge tensor"),
    ]:
        ax = fig.add_subplot(gs[row, 0]); draw_molecule(ax, ex, mode)
        pos = centered_coords(ex["sdf"]["coords"])
        for atom, color in [(2, "#f2b544"), (1, "#2c9cbf"), (3, "#2c9cbf")]:
            ax.add_patch(Circle(pos[atom], .29, fill=False, ec=color, lw=3.2, zorder=6))
        panel_title(ax, heading, subtitle)

    atom2 = ex["node"][2]
    assert ex["sdf"]["elements"][2] == "N"
    assert ex["site_classes"][2] == "N_pyridine"
    assert ex["bond"][2, 1] == 2 and ex["bond"][2, 3] == 1
    assert np.isclose(ex["coupling"][2, 1], -1) and np.isclose(ex["coupling"][2, 3], -1)
    card(fig.add_subplot(gs[1, 1]), "Node and edge arrays (exact values)",
         ["X_atom: 41 × 18  (first 10 active)",
          "X_atom[2]: element=N, class=N_pyridine",
          f"  [q/2,d/4,e/2,Hjj/2] = [{fmt(atom2[14])},{fmt(atom2[15])},{fmt(atom2[16])},{fmt(atom2[17])}]",
          "B[2,1]=2   B[2,3]=1   (bond order)",
          "C[2,1]=-1 C[2,3]=-1 (π coupling)"], "#f2f7f9", 10.5)
    card(fig.add_subplot(gs[1, 2]), "One atom-graph layer at N2",
         ["h_i = Linear_18→96(X_atom[i])",
          "degree_B(2) = 2",
          "u₂ = Wself h₂ + b + (W₂ h₁ + W₁ h₃",
          "     - Wπ h₁ - Wπ h₃) / 2",
          "h′₂ = LayerNorm(h₂ + GELU(u₂))"], "#e9f4ed", 10.2)
    pi2 = ex["pi_features"][2]
    card(fig.add_subplot(gs[2, 1]), "Site and edge arrays (exact values)",
         ["X_π: 40 × 3  (first 10 active)",
          "columns = [Hjj/2, eπ/2, degree_H/4]",
          f"X_π[2] = [{fmt(pi2[0])}, {fmt(pi2[1])}, {fmt(pi2[2])}]",
          "H[2,1]=-1  H[2,3]=-1",
          "off-diagonal H is the weighted adjacency"], "#f2f7f9", 10.5)
    card(fig.add_subplot(gs[2, 2]), "One π-graph layer at site 2",
         ["h_j = Linear_3→96(X_π[j])",
          "degree_H(2) = 2",
          "u₂ = Wself h₂ + b",
          "     + (-Wedge h₁ - Wedge h₃) / 2",
          "h′₂ = LayerNorm(h₂ + GELU(u₂))"], "#e9f4ed", 10.2)
    png = outdir / "graph_encoding_walkthrough.png"
    svg = outdir / "graph_encoding_walkthrough.svg"
    for path in (png, svg):
        fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return png, svg


def plot_overview(examples, outdir):
    fig = setup_figure(16, 13, "The same molecules in four representations")
    gs = GridSpec(len(examples), 3, figure=fig, width_ratios=[1, 1, 1.1])
    for row, ex in enumerate(examples):
        ax = fig.add_subplot(gs[row, 0]); draw_molecule(ax, ex, "chemical")
        panel_title(ax, f"{ex['title']}  •  #{ex['index']}", ex["smiles"])
        ax = fig.add_subplot(gs[row, 1]); draw_molecule(ax, ex, "pi")
        panel_title(ax, "π-site graph", f"{ex['n']} sites; {ex['pi_e']} π electrons")
        ax = fig.add_subplot(gs[row, 2]); matrix_panel(ax, ex["H"],
                    [str(j) for j in range(ex["n"])], "Hückel matrix H", -2, 0, "Blues_r")
    path = outdir / "overview.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def export_inputs(examples, feat, source, outdir):
    indices = np.array([ex["index"] for ex in examples], dtype=np.int64)
    keys = ("H", "site_mask", "site_pi_electrons", "site_counts", "atom_counts",
            "pi_electrons", "gap", "node", "bond", "pi_coupling", "smiles_ids")
    arrays = {k: feat[k][indices] for k in keys}
    arrays["source_indices"] = indices
    arrays["aux"] = np.stack([ex["aux"] for ex in examples])
    arrays["atom_mask"] = np.arange(feat["node"].shape[1])[None, :] < arrays["atom_counts"][:, None]
    arrays["matrix_row_tokens"] = np.concatenate(
        [arrays["H"], arrays["site_pi_electrons"][..., None] / 2], axis=2)
    diag = np.diagonal(arrays["H"], axis1=1, axis2=2)
    offdiag = arrays["H"] - np.eye(arrays["H"].shape[1], dtype=np.float32)[None] * diag[:, :, None]
    arrays["pi_graph_node_features"] = np.stack(
        [diag / 2, arrays["site_pi_electrons"] / 2, (offdiag != 0).sum(axis=2) / 4],
        axis=2).astype(np.float32)
    arrays["site_atom_indices"] = source["site_atom_indices"][indices]
    arrays["site_classes"] = source["site_classes"][indices]
    arrays["smiles"] = source["smiles"][indices]
    np.savez_compressed(outdir / "selected_inputs.npz", **arrays)
    records = []
    for ex in examples:
        records.append({
            "dataset_row": ex["index"], "name": ex["title"], "smiles": ex["smiles"],
            "smiles_tokens": ex["tokens"], "smiles_ids_active": ex["ids"].tolist(),
            "n_sites": ex["n"], "n_atoms": ex["a"], "n_pi_electrons": ex["pi_e"],
            "gap_beta_units": ex["gap"], "site_to_atom_index": ex["site_atoms"].tolist(),
            "site_classes": ex["site_classes"], "site_pi_electrons": ex["site_e"].tolist(),
            "H_active": ex["H"].tolist(), "pi_graph_node_features": ex["pi_features"].tolist(),
            "atom_graph_node_features": ex["node"].tolist(),
            "bond_order_active": ex["bond"].tolist(), "pi_coupling_active": ex["coupling"].tolist(),
            "aux": ex["aux"].tolist(),
        })
    (outdir / "encodings.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    vocab = json.loads((PROBE / "vocab.json").read_text())
    sdf = read_sdf_selected(SOURCE / "molecules.sdf", {x[0] for x in EXAMPLES})
    with np.load(PROBE / "features.npz", allow_pickle=False) as feat, \
         np.load(SOURCE / "dataset.npz", allow_pickle=False) as source:
        examples = [make_example(i, slug, title, feat, source, vocab, sdf[i])
                    for i, slug, title in EXAMPLES]
        export_inputs(examples, feat, source, HERE)
    paths = [plot_overview(examples, outdir), plot_encoder_paths(outdir)]
    paths.extend(plot_graph_walkthrough(examples[-1], outdir))
    for ex in examples:
        paths.extend([plot_views(ex, outdir), plot_encodings(ex, outdir)])
    print("Wrote exact arrays to", HERE / "selected_inputs.npz")
    print("Wrote readable active arrays to", HERE / "encodings.json")
    for path in paths:
        print("Wrote", path)


if __name__ == "__main__":
    main()
