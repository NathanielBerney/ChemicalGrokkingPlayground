#!/usr/bin/env python3
"""Plot the formal Hückel/Möbius orbital circuits in the pericyclic dataset.

The diagrams are topology schematics, not optimized transition-state
geometries. Every template/mode is checked against an actual dataset row.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "pericyclic_thermal_dataset"
OUT = HERE / "figures"

INK = "#19324b"
MUTED = "#536a7d"
BG = "#f7fafc"
HUE_H = "#0b897e"
HUE_M = "#8e54a7"
FORMING = "#2b9a6f"
BREAKING = "#df993b"
PI = "#294d74"
CONJUGATED = "#8a9aac"
FLIP = "#c52e78"
SITE_FILL = {"C": "#25435e", "O": "#cc515a", "H": "#d99b34"}

NAMES = {
    "diels_alder_4plus2": "Diels–Alder [4+2]",
    "alder_ene": "Alder–ene",
    "claisen_3_3": "Claisen [3,3]",
    "cope_3_3": "Cope [3,3]",
    "cycloaddition_2plus2": "[2+2] cycloaddition",
    "cycloaddition_4plus4": "[4+4] cycloaddition",
    "electrocyclic_4pi": "4π electrocyclic closure",
    "electrocyclic_6pi": "6π electrocyclic closure",
}
SOURCE_LABELS = {
    "diels_alder_4plus2": "4π + 2π",
    "alder_ene": "2π + 2π + C–H σ",
    "claisen_3_3": "2π + 2π + C–O σ",
    "cope_3_3": "2π + 2π + C–C σ",
    "cycloaddition_2plus2": "2π + 2π",
    "cycloaddition_4plus4": "4π + 4π",
    "electrocyclic_4pi": "4π",
    "electrocyclic_6pi": "6π",
}
MODE_LABELS = {
    "supra_supra": "suprafacial / suprafacial",
    "supra_antara": "suprafacial / antarafacial",
    "all_suprafacial": "all suprafacial",
    "one_antarafacial_component": "one antarafacial component",
    "conrotatory": "conrotatory",
    "disrotatory": "disrotatory",
}


def load_and_validate():
    templates = json.loads((DATA / "templates.json").read_text())
    with (DATA / "metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    first = {}
    for index, row in enumerate(rows):
        first.setdefault((row["reaction_family"], row["candidate_approach"]), (index, row))
    manifest = []
    with np.load(DATA / "dataset.npz", allow_pickle=False) as d:
        for template in templates:
            n = len(template["sites"])
            assert n == len(template["edge_roles_cyclic_order"])
            for mode, parity in template["modes"]:
                index, row = first[(template["name"], mode)]
                phase = -1 if parity else 1
                h = d["H"][index, :n, :n]
                loop = int(np.prod(np.sign(-h[np.arange(n), (np.arange(n) + 1) % n])))
                allowed = int((n % 4 == 2) == (phase == 1))
                assert loop == phase == int(row["gauge_invariant_cycle_phase"])
                assert int(d["y"][index]) == allowed == int(row["thermal_allowed"])
                assert int(row["n_active_electrons"]) == n
                assert row["topology"] == ("Huckel" if phase == 1 else "Mobius")
                manifest.append({"family": template["name"], "mode": mode,
                                 "topology": row["topology"], "phase": phase,
                                 "electrons": n, "thermal_allowed": allowed,
                                 "example_index": index, "sample_id": row["sample_id"]})
    assert len(manifest) == 16
    return templates, manifest


def role_style(role):
    if role == "pi":
        return PI, "solid", 3.3
    if role == "conjugated":
        return CONJUGATED, "solid", 2.5
    if role.startswith("forming"):
        return FORMING, (0, (5, 3)), 3.1
    if role.startswith("breaking"):
        return BREAKING, (0, (1.2, 2.2)), 3.2
    raise ValueError(role)


def circuit(ax, template, phase, *, small=False):
    """Draw a canonical orbital gauge with any phase flip at the closing edge."""
    n = len(template["sites"])
    angles = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    radius = .71 if not small else .68
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    for i, role in enumerate(template["edge_roles_cyclic_order"]):
        j = (i + 1) % n
        color, dash, width = role_style(role)
        p, q = points[i], points[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color,
                linestyle=dash, lw=width * (.8 if small else 1),
                solid_capstyle="round", zorder=1)
    if phase == -1:
        p, q = points[-1], points[0]
        midpoint = (p + q) / 2
        tangent = (q - p) / np.linalg.norm(q - p)
        normal = np.array([-tangent[1], tangent[0]])
        scale = .09 if not small else .075
        for sign in (-1, 1):
            c = midpoint + sign * .045 * tangent
            a = c - scale * (tangent + normal) / np.sqrt(2)
            b = c + scale * (tangent + normal) / np.sqrt(2)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=FLIP, lw=3.5,
                    solid_capstyle="round", zorder=6)
    for i, (point, element) in enumerate(zip(points, template["sites"])):
        ax.add_patch(Circle(point, .126 if not small else .113,
                            fc=SITE_FILL[element], ec="white", lw=1.5, zorder=4))
        ax.text(*point, element, ha="center", va="center", color="white",
                fontsize=10 if not small else 8.7, weight="bold", zorder=5)
        outward = point / max(np.linalg.norm(point), 1e-6)
        ax.text(*(point + outward * (.17 if not small else .15)), str(i),
                ha="center", va="center", fontsize=8 if not small else 7,
                color=MUTED, zorder=5)
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-.96, .98)
    ax.set_aspect("equal")
    ax.axis("off")


def legend(fig, y):
    entries = [(PI, "π bond", "solid"),
               (CONJUGATED, "conjugated link", "solid"),
               (FORMING, "forming bond", (0, (5, 3))),
               (BREAKING, "breaking bond", (0, (1.2, 2.2))),
               (FLIP, "one orbital phase inversion", "solid")]
    xs = [.06, .235, .44, .615, .79]
    for (color, label, dash), x in zip(entries, xs):
        line = plt.Line2D([x, x + .035], [y, y], color=color, lw=3.2,
                          linestyle=dash, transform=fig.transFigure, clip_on=False)
        fig.add_artist(line)
        fig.text(x + .045, y, label, va="center", fontsize=10.5, color=INK)


def allowed_card(ax, template, mode, phase):
    ax.set_facecolor("white")
    ax.add_patch(FancyBboxPatch((.015, .015), .97, .97,
                boxstyle="round,pad=.01,rounding_size=.035",
                fc="#eef8f6" if phase == 1 else "#f7f0fa",
                ec="#b8d6d0" if phase == 1 else "#d6c2dd",
                lw=1.2, transform=ax.transAxes, zorder=-1))
    ax.text(.06, .94, NAMES[template["name"]], transform=ax.transAxes,
            fontsize=13.5, weight="bold", color=INK, va="top")
    topology = "Hückel" if phase == 1 else "Möbius"
    ax.text(.06, .85, f"{len(template['sites'])} e⁻  •  {topology}  •  {SOURCE_LABELS[template['name']]}",
            transform=ax.transAxes, fontsize=10.4, color=HUE_H if phase == 1 else HUE_M,
            weight="bold", va="top")
    inset = ax.inset_axes([.19, .22, .62, .59])
    circuit(inset, template, phase)
    ax.text(.06, .13, MODE_LABELS[mode], transform=ax.transAxes,
            fontsize=9.6, color=MUTED)
    ax.text(.94, .13, "THERMAL ALLOWED", transform=ax.transAxes,
            ha="right", fontsize=9.7, color=HUE_H if phase == 1 else HUE_M,
            weight="bold")
    ax.axis("off")


def make_allowed_atlas(templates):
    fig = plt.figure(figsize=(18.5, 11.3), facecolor=BG)
    fig.text(.045, .975, "Pericyclic transition-state orbital circuits",
             ha="left", va="top", fontsize=23, color=INK, weight="bold")
    fig.text(.045, .925,
             "Eight formally thermally allowed approaches from the dataset templates  •  Hückel: 4k+2 e⁻  •  Möbius: 4k e⁻",
             fontsize=12, color=MUTED, va="top")
    legend(fig, .863)
    gs = fig.add_gridspec(2, 4, left=.035, right=.985, top=.81, bottom=.13,
                          wspace=.075, hspace=.12)
    for template, cell in zip(templates, gs):
        n = len(template["sites"])
        wanted = 1 if n % 4 == 2 else -1
        mode, parity = next((m, p) for m, p in template["modes"]
                            if (-1 if p else 1) == wanted)
        allowed_card(fig.add_subplot(cell), template, mode, wanted)
    fig.text(.05, .077,
             "Nodes are active orbitals (C, O, or transferring H). Each edge is an interaction in the proposed cyclic transition state.",
             fontsize=10.7, color=INK)
    fig.text(.05, .044,
             "The pink slash is one canonical phase flip; orbital sign gauges can move it. These are formal orbital topologies, not optimized 3D structures or barrier predictions.",
             fontsize=10.2, color=MUTED)
    return fig


def comparison_cell(ax, template, mode, phase, allowed):
    ax.set_facecolor("#eef8f6" if phase == 1 else "#f7f0fa")
    inset = ax.inset_axes([.02, .04, .39, .92])
    circuit(inset, template, phase, small=True)
    ax.text(.43, .77, MODE_LABELS[mode], fontsize=9.7,
            color=INK, transform=ax.transAxes, va="center")
    ax.text(.43, .42, "ALLOWED" if allowed else "FORBIDDEN",
            fontsize=11, weight="bold", transform=ax.transAxes,
            color="#16805e" if allowed else "#a84b52", va="center")
    ax.text(.43, .16, f"loop phase {phase:+d}", fontsize=9.3,
            color=MUTED, transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_edgecolor("#c7d6dd")
    ax.set_xticks([]); ax.set_yticks([])


def make_paired_comparison(templates):
    fig = plt.figure(figsize=(15.8, 23), facecolor=BG)
    fig.text(.045, .983, "How the approach changes the orbital circuit",
             ha="left", va="top", fontsize=22, color=INK, weight="bold")
    fig.text(.045, .958,
             "Each row holds the electron count fixed and changes the proposed orbital phase topology.",
             fontsize=11.5, color=MUTED, va="top")
    fig.text(.38, .929, "HÜCKEL  •  no phase inversion  •  Φ = +1",
             fontsize=12.3, color=HUE_H, ha="center", weight="bold")
    fig.text(.79, .929, "MÖBIUS  •  one phase inversion  •  Φ = −1",
             fontsize=12.3, color=HUE_M, ha="center", weight="bold")
    gs = fig.add_gridspec(8, 3, left=.04, right=.98, top=.91, bottom=.079,
                          width_ratios=[1.45, 2.6, 2.6], hspace=.13, wspace=.045)
    for row_index, template in enumerate(templates):
        name_ax = fig.add_subplot(gs[row_index, 0]); name_ax.axis("off")
        name_ax.text(.03, .70, NAMES[template["name"]], transform=name_ax.transAxes,
                     fontsize=11.5, weight="bold", color=INK, va="center")
        name_ax.text(.03, .38, f"{len(template['sites'])} active e⁻  •  {SOURCE_LABELS[template['name']]}",
                     transform=name_ax.transAxes, fontsize=9.8, color=MUTED, va="center")
        for col, phase in ((1, 1), (2, -1)):
            mode, parity = next((m, p) for m, p in template["modes"]
                                if (-1 if p else 1) == phase)
            allowed = int((len(template["sites"]) % 4 == 2) == (phase == 1))
            comparison_cell(fig.add_subplot(gs[row_index, col]),
                            template, mode, phase, allowed)
    legend(fig, .062)
    fig.text(.05, .038,
             "Green means formally thermally allowed; red means symmetry-forbidden for that specified concerted approach.",
             fontsize=10.6, color=INK)
    fig.text(.05, .020,
             "A formally allowed antarafacial route may still be geometrically inaccessible. No 3D transition-state geometry was optimized.",
             fontsize=10.2, color=MUTED)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    templates, manifest = load_and_validate()
    (HERE / "example_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for stem, figure in (("allowed_orbital_circuits", make_allowed_atlas(templates)),
                         ("paired_huckel_mobius_approaches", make_paired_comparison(templates))):
        for suffix in ("png", "svg", "pdf"):
            figure.savefig(OUT / f"{stem}.{suffix}", dpi=190,
                           bbox_inches="tight", facecolor=BG)
        plt.close(figure)
        print("Wrote", stem)
    print("Validated", len(manifest), "template/approach pairs against the stored matrices and labels")


if __name__ == "__main__":
    main()
