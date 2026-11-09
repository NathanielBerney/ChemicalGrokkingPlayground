#!/usr/bin/env python3
"""Reactant orbitals -> proposed thermal approach -> product connectivity.

Original schematic drawings, not computed transition-state geometries/densities.
Open-chain frontier coefficients are calculated and verified analytically. The
ene/[3,3] pages deliberately use localized pi/sigma orbitals rather than calling
them a molecular HOMO. See orbital_schemes/README.md for conventions/sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath

from make_atlas import load_and_validate, NAMES, MODE_LABELS

HERE = Path(__file__).resolve().parent
OUT = HERE / "orbital_schemes"
FIGS = OUT / "figures"
INK = "#233343"
MUTED = "#607080"
POS = "#247db3"
NEG = "#e99849"
GOOD = "#15845c"
BAD = "#ba3a50"
TWIST = "#8052a6"
BREAK = "#bc842d"
FAINT = "#e2e9ed"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "font.size": 11})

ORDER = ["diels_alder_4plus2", "cycloaddition_2plus2",
         "cycloaddition_4plus4", "electrocyclic_4pi", "electrocyclic_6pi",
         "claisen_3_3", "cope_3_3", "alder_ene"]
PRODUCTS = {
    "diels_alder_4plus2": ("Cyclohexene", "C1=CCCCC1"),
    "cycloaddition_2plus2": ("Cyclobutane", "C1CCC1"),
    "cycloaddition_4plus4": ("Cycloocta-1,5-diene", "C1=CCCC=CCC1"),
    "electrocyclic_4pi": ("Cyclobutene", "C1=CCC1"),
    "electrocyclic_6pi": ("Cyclohexa-1,3-diene", "C1=CC=CCC1"),
    "claisen_3_3": ("Pent-4-enal", "O=CCCC=C"),
    "cope_3_3": ("Rearranged hexa-1,5-diene", "C=CCCC=C"),
    "alder_ene": ("Pent-1-ene", "C=CCCC"),
}
MO_RECORDS = {}


def chain_mo(n, k):
    """Normalized kth MO of a uniform chain, Hii=0, Hij=-1, 1-based k."""
    c = np.sqrt(2 / (n + 1)) * np.sin(np.arange(1, n + 1) * k * np.pi / (n + 1))
    h = -np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    energy = -2 * np.cos(k * np.pi / (n + 1))
    assert np.allclose(h @ c, energy * c, atol=1e-12)
    assert np.isclose(c @ c, 1)
    MO_RECORDS[f"chain_{n}_psi_{k}"] = {"n": n, "k": k, "energy_beta_units": energy,
                                       "coefficients": c.tolist()}
    return c


def petal(ax, center, vector, width, color, alpha=1):
    """A p/hybrid lobe with a node at center; vector runs to the lobe tip."""
    center, vector = np.asarray(center), np.asarray(vector)
    normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
    vertices = [center, center + .20 * vector + width * normal,
                center + .94 * vector + width * normal, center + vector,
                center + .94 * vector - width * normal,
                center + .20 * vector - width * normal, center, center]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), fc=color, ec=color,
                           lw=.65, alpha=.88 * alpha, zorder=3))


def orbital(ax, center, coefficient=1, direction=(0, 1), size=.56,
            alpha=1, hybrid=False, signs=True):
    unit = np.array(direction, float)
    unit /= np.linalg.norm(unit)
    for sign in (1, -1):
        s = size * (.48 if hybrid and sign == -1 else 1)
        v = sign * s * unit
        phase = np.sign(coefficient) * sign
        petal(ax, center, v, .19 * (s / .56), POS if phase > 0 else NEG, alpha)
        if signs and (not hybrid or sign > 0):
            p = np.asarray(center) + .60 * v
            ax.text(*p, "+" if phase > 0 else "−", ha="center", va="center",
                    color="white" if phase > 0 else "#59340b", fontsize=10.3,
                    weight="bold", alpha=alpha, zorder=6)


def atom(ax, p, element="C", number=None, alpha=1, offset=(.15, -.10)):
    if element == "C":
        ax.add_patch(Circle(p, .043, fc=INK, ec="white", lw=.3, alpha=alpha, zorder=7))
    else:
        ax.text(*p, element, ha="center", va="center", fontsize=12.5,
                weight="bold", color=BAD if element == "O" else INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.1},
                alpha=alpha, zorder=7)
    if number is not None:
        ax.text(p[0] + offset[0], p[1] + offset[1], str(number),
                color=MUTED, fontsize=7.4, zorder=8, alpha=alpha,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .1})


def bond(ax, a, b, order=1, color=INK, alpha=1, dashed=False, width=1.65):
    a, b = np.asarray(a), np.asarray(b)
    delta = b - a
    normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    for d in ([0] if order == 1 else [-.046, .046]):
        p, q = a + d * normal, b + d * normal
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=width,
                ls=(0, (3.4, 2.6)) if dashed else "-", alpha=alpha,
                solid_capstyle="round", zorder=2)


def note(ax, text, y, color=MUTED, size=10.4):
    ax.text(0, y, text, color=color, fontsize=size, ha="center", va="center",
            linespacing=1.45, zorder=12)


def setup(ax):
    ax.set_xlim(-2.58, 2.58)
    ax.set_ylim(-2.32, 2.70)
    ax.set_aspect("equal")
    ax.axis("off")


def orders(template, product=False):
    ans = []
    for role in template["edge_roles_cyclic_order"]:
        if product:
            ans.append(0 if role.startswith("breaking") else 2 if role == "conjugated" else 1)
        else:
            ans.append(0 if role.startswith("forming") else 2 if role == "pi" else 1)
    return ans


def skeleton(ax, points, template, product=False, ts=False, numbers=False,
             alpha=1, show_atoms=True):
    n = len(points)
    for i, (order, role) in enumerate(zip(orders(template, product), template["edge_roles_cyclic_order"])):
        if ts and role.startswith("forming"):
            continue  # Orbital contacts are drawn separately.
        if order:
            col = BREAK if ts and role.startswith("breaking") else INK
            bond(ax, points[i], points[(i + 1) % n], order, color=col,
                 dashed=ts and role.startswith("breaking"), alpha=alpha)
    if show_atoms:
        for i, (p, el) in enumerate(zip(points, template["sites"])):
            atom(ax, p, el, i + 1 if numbers else None, alpha=alpha)


def bracket(ax):
    for side in (-1, 1):
        x = side * 2.12
        ax.plot([x - side * .13, x, x, x - side * .13],
                [-1.56, -1.56, 1.83, 1.83], color=MUTED, lw=1.1, zorder=0)
    ax.text(2.16, 1.83, "‡", fontsize=17, color=INK)


def contact(ax, a, b, matches, label=None, label_pos=None):
    col = GOOD if matches else BAD
    bond(ax, a, b, color=col, dashed=True, width=2.1)
    mid = (np.asarray(a) + b) / 2
    if not matches:
        ax.plot(mid[0], mid[1], marker="x", ms=10, mew=2, color=BAD, zorder=9)
    if label:
        p = mid if label_pos is None else label_pos
        ax.text(*p, label, color=col, fontsize=9.6, ha="center", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=10)


def rotate_arrow(ax, center, clockwise, radius=.45):
    angles = np.linspace(155, -55, 30) if clockwise else np.linspace(25, 235, 30)
    xy = np.c_[np.cos(np.radians(angles)), np.sin(np.radians(angles))] * radius + center
    ax.plot(xy[:, 0], xy[:, 1], color=TWIST, lw=1.35, zorder=9)
    ax.add_patch(FancyArrowPatch(xy[-3], xy[-1], arrowstyle="-|>", color=TWIST,
                                mutation_scale=11, lw=1.3, zorder=9))


def cyclo_geometry(name, ts):
    if name == "cycloaddition_2plus2":
        nupper, nlower = 2, 2
    elif name == "diels_alder_4plus2":
        nupper, nlower = 4, 2
    else:
        nupper, nlower = 4, 4
    shift = 0 if ts else (.12 if nlower == 4 else .28)
    upper_end_y = .62 if nlower == 4 else .47
    upper = [(-1.13, .47 + shift), (1.13, .47 + shift)] if nupper == 2 else [
        (-1.13, upper_end_y + shift), (-.75, 1.43 + shift), (.75, 1.43 + shift), (1.13, upper_end_y + shift)]
    lower = [(1.13, -.80 - shift), (-1.13, -.80 - shift)] if nlower == 2 else [
        (1.13, -.62 - shift), (.75, -1.15 - shift), (-.75, -1.15 - shift), (-1.13, -.62 - shift)]
    return np.array(upper + lower), nupper, nlower


def cyclo_orbitals(ax, template, parity, ts):
    name = template["name"]
    p, m, n = cyclo_geometry(name, ts)
    ctop = chain_mo(m, m // 2)
    clow = chain_mo(n, n // 2 + 1)
    # Set the arbitrary whole-fragment phase so the left contact is constructive.
    clow *= np.sign(-ctop[0] / clow[-1])
    coeff = np.r_[ctop, clow]
    directions = np.tile([0., 1.], (len(p), 1))
    if ts and parity:
        directions[m] *= -1  # Opposite face at right terminus, same MO coefficients.
    sizes = .45 + .12 * np.abs(coeff) / np.abs(coeff).max()
    skeleton(ax, p, template, ts=ts, show_atoms=False)
    for i in range(len(p)):
        orbital(ax, p[i], coeff[i], directions[i], size=sizes[i])
        atom(ax, p[i])
    if ts:
        right_match = bool(np.sign(-ctop[-1]) == np.sign(clow[0] * directions[m, 1]))
        left_match = bool(np.sign(-ctop[0]) == np.sign(clow[-1]))
        expected = (len(p) % 4 == 2) != bool(parity)
        assert (left_match and right_match) == expected
        for up, lo, match in [(0, len(p) - 1, left_match), (m - 1, m, right_match)]:
            a = p[up] - [0, sizes[up] * .82]
            b = p[lo] + [0, sizes[lo] * .82]
            contact(ax, a, b, match)
        note(ax, "both contacts\nin phase" if expected else "one contact\nout of phase",
             .02, GOOD if expected else BAD, size=10.2)
        if parity:
            rotate_arrow(ax, p[m] + [.49, -.02], clockwise=True, radius=.24)
            ax.text(1.76, -1.36 if n == 2 else -.95, "opposite\nface", color=TWIST,
                    fontsize=9.1, ha="center", va="center")
        note(ax, f"{'Möbius' if parity else 'Hückel'} circuit  ·  {m+n} electrons",
             2.38, TWIST if parity else INK, size=11.5)
        note(ax, "Partial bonds join the facing lobes.", -2.04, size=9.8)
        bracket(ax)
    else:
        top_label = "Ethene HOMO, ψ₁ (occupied)" if m == 2 else "Diene HOMO, ψ₂ (occupied)"
        low_label = "Ethene LUMO, ψ₂ (empty)" if n == 2 else "Diene LUMO, ψ₃ (empty)"
        note(ax, top_label, 2.40, INK, size=11.1)
        note(ax, low_label, -2.13, INK, size=11.1)
        note(ax, "+", -.15, MUTED, size=18)


def electro_geometry(n):
    if n == 4:
        return np.array([[-1.12, -.60], [-1.12, .94], [1.12, .94], [1.12, -.60]])
    return np.array([[-1.12, -.65], [-1.55, .60], [-.75, 1.48],
                     [.75, 1.48], [1.55, .60], [1.12, -.65]])


def electro_orbitals(ax, template, parity, ts):
    n = len(template["sites"])
    p = electro_geometry(n)
    c = chain_mo(n, n // 2)
    sizes = .43 + .15 * np.abs(c) / np.abs(c).max()
    directions = np.tile([0., 1.], (n, 1))
    if ts:
        directions[0] = [1, 0]   # Left terminus clockwise in the drawing.
        directions[-1] = [1, 0] if parity else [-1, 0]
    skeleton(ax, p, template, ts=ts, show_atoms=False)
    for i in range(n):
        orbital(ax, p[i], c[i], directions[i], size=sizes[i])
        atom(ax, p[i])
    if ts:
        left_phase = np.sign(c[0])
        right_phase = np.sign(c[-1]) * (-1 if parity else 1)
        matched = bool(left_phase == right_phase)
        assert matched == ((n % 4 == 2) != bool(parity))
        a, b = p[0] + [sizes[0] * .82, 0], p[-1] - [sizes[-1] * .82, 0]
        contact(ax, a, b, matched)
        note(ax, "in phase → bonding" if matched else "out of phase → antibonding",
             -1.42, GOOD if matched else BAD, size=10.8)
        for i, clockwise in [(0, True), (n - 1, bool(parity))]:
            rotate_arrow(ax, p[i], clockwise, radius=.70)
        note(ax, f"{'Möbius' if parity else 'Hückel'} circuit  ·  {n} electrons",
             2.38, TWIST if parity else INK, size=11.5)
        note(ax, "same rotation sense" if parity else "opposite rotation senses", -2.04, TWIST, size=10.3)
        bracket(ax)
    else:
        note(ax, f"Polyene HOMO, ψ{str(n // 2).translate(str.maketrans('23', '₂₃'))} (occupied)",
             2.38, INK, size=11.5)
        note(ax, "Opposite terminal phases" if n == 4 else "Same terminal phases",
             -1.57, INK, size=10.7)
        note(ax, "Follow the terminal lobes during rotation.", -2.04, size=10)


def local_geometry():
    theta = np.radians([210, 150, 90, 30, -30, -90])
    return np.c_[1.49 * np.cos(theta), 1.12 * np.sin(theta)] + [0, .12]


def local_orbitals(ax, template, parity, ts):
    p = local_geometry()
    name = template["name"]
    skeleton(ax, p, template, ts=ts, show_atoms=False)
    if not ts:
        for i in (0, 1, 4, 5):
            orbital(ax, p[i], size=.45, direction=(.35, 1))
        # Occupied sigma-bond orbital, with same-sign inward hybrid lobes.
        a, b = p[2], p[3]
        unit = (b - a) / np.linalg.norm(b - a)
        orbital(ax, a, direction=unit, size=np.linalg.norm(b - a) * .64, hybrid=True)
        if name != "alder_ene":
            orbital(ax, b, direction=-unit, size=np.linalg.norm(b - a) * .64, hybrid=True)
        else:
            ax.add_patch(Circle(b, .22, fc=POS, ec=POS, alpha=.9, zorder=4))
        note(ax, "Occupied π + σ + π bond orbitals", 2.38, INK, size=11.1)
        sigma = "C–H" if name == "alder_ene" else "C–O" if name == "claisen_3_3" else "C–C"
        ax.text(.55, 1.75, f"σ({sigma})", fontsize=10, color=INK, ha="center")
        ax.text(-1.96, .14, "π", fontsize=13, color=INK)
        ax.text(.89, -1.40, "π", fontsize=13, color=INK)
        note(ax, "Localized occupied orbitals; not one HOMO.", -2.04, size=9.9)
    else:
        # Schematic local overlap basis, deliberately distinct from frontier MOs.
        # A half twist is distributed around the ring in the projected p axes.
        for i, (point, element) in enumerate(zip(p, template["sites"])):
            if element == "H":
                ax.add_patch(Circle(point, .24, fc=POS, ec=POS, alpha=.9, zorder=4))
                continue
            theta = i * np.pi / 6 if parity else 0
            direction = [np.sin(theta), np.cos(theta)]
            orbital(ax, point, direction=direction, size=.46,
                    hybrid=i in (2, 3), signs=True)
        for i, role in enumerate(template["edge_roles_cyclic_order"]):
            if role.startswith("forming"):
                bond(ax, p[i], p[(i + 1) % 6], color=GOOD, dashed=True, width=2)
        if parity:
            a, b = p[5], p[0]
            mid = (a + b) / 2
            normal = np.array([-(b-a)[1], (b-a)[0]]) / np.linalg.norm(b-a)
            for d in (-.055, .055):
                q = mid + [d, 0]
                ax.plot([q[0] - .14 * normal[0], q[0] + .14 * normal[0]],
                        [q[1] - .14 * normal[1], q[1] + .14 * normal[1]], color=TWIST, lw=3, zorder=10)
            ax.text(-1.55, -1.28, "half twist", color=TWIST, fontsize=9.5, ha="center")
        note(ax, f"{'Möbius' if parity else 'Hückel'} array  ·  6 electrons",
             2.38, TWIST if parity else INK, size=11.5)
        note(ax, "one phase inversion" if parity else "no phase inversion", -.03,
             TWIST if parity else INK, size=10)
        note(ax, "Local overlap basis; σ centers reorganize.", -2.04, size=9.9)
        bracket(ax)
    for i, (point, el) in enumerate(zip(p, template["sites"])):
        atom(ax, point, el, i + 1, offset=(.17, -.07))


def product_panel(ax, template, allowed):
    n, name = len(template["sites"]), template["name"]
    if name in ("claisen_3_3", "cope_3_3", "alder_ene"):
        p = local_geometry()
    elif name.startswith("electrocyclic"):
        p = electro_geometry(n)
    else:
        # Preserve atom positions across the arrow so each new bond is traceable.
        p, _, _ = cyclo_geometry(name, ts=True)
    alpha = 1 if allowed else .48
    skeleton(ax, p, template, product=True,
             numbers=name in ("claisen_3_3", "cope_3_3", "alder_ene"), alpha=alpha)
    for i, role in enumerate(template["edge_roles_cyclic_order"]):
        if role.startswith("forming"):
            bond(ax, p[i], p[(i + 1) % n], color=GOOD, alpha=alpha, width=3)
    note(ax, "Product connectivity" if allowed else "Proposed product connectivity",
         2.38, INK if allowed else BAD, size=11.4)
    note(ax, PRODUCTS[name][0], -1.48, INK, size=11)
    note(ax, "New σ bond(s) highlighted in green." if allowed else
         "Forbidden for this concerted thermal route.", -2.04, GOOD if allowed else BAD, size=10)


def row(fig, y0, height, template, mode, parity, *, compact=False):
    allowed = (len(template["sites"]) % 4 == 2) != bool(parity)
    status_color = GOOD if allowed else BAD
    # Axes share a physical size so orbital lobes keep the same aspect ratio.
    widths = [.292, .292, .245]
    lefts = [.035, .384, .745]
    axes = [fig.add_axes([x, y0, width, height]) for x, width in zip(lefts, widths)]
    for ax in axes:
        setup(ax)
    name = template["name"]
    draw = (electro_orbitals if name.startswith("electrocyclic") else
            local_orbitals if name in ("claisen_3_3", "cope_3_3", "alder_ene") else cyclo_orbitals)
    draw(axes[0], template, parity, False)
    draw(axes[1], template, parity, True)
    product_panel(axes[2], template, allowed)
    for x1, x2, blocked in [(.330, .376, False), (.684, .736, not allowed)]:
        yy = y0 + .50 * height
        fig.add_artist(FancyArrowPatch((x1, yy), (x2, yy), transform=fig.transFigure,
                                      arrowstyle="-|>", mutation_scale=18,
                                      lw=1.8, color=BAD if blocked else INK))
        if blocked:
            xx = (x1 + x2) / 2
            fig.add_artist(plt.Line2D([xx - .007, xx + .007], [yy - .013, yy + .013],
                                      transform=fig.transFigure, color=BAD, lw=2.8))
        else:
            fig.text((x1 + x2) / 2, yy + .024, "Δ", fontsize=13, color=INK, ha="center")
    mode_label = MODE_LABELS[mode]
    status = "THERMALLY ALLOWED" if allowed else "THERMALLY FORBIDDEN"
    if allowed and name in ("cycloaddition_2plus2", "cycloaddition_4plus4"):
        status += "  (FORMAL)"
    ytitle = y0 + height + .010
    fig.text(.043, ytitle, status, fontsize=12 if not compact else 11,
             weight="bold", color=status_color, va="bottom")
    fig.text(.40, ytitle, mode_label, fontsize=12 if not compact else 11,
             color=INK, va="bottom")
    return allowed


def figure_legend(fig, y):
    fig.add_artist(Circle((.052, y), .0045, transform=fig.transFigure, fc=POS, ec=POS))
    fig.add_artist(Circle((.068, y), .0045, transform=fig.transFigure, fc=NEG, ec=NEG))
    fig.text(.081, y, "+ / − orbital phase (not charge)", va="center", fontsize=10.6, color=INK)
    fig.add_artist(plt.Line2D([.39, .419], [y, y], transform=fig.transFigure,
                             color=GOOD, lw=2, ls="--"))
    fig.text(.43, y, "forming overlap / bond", va="center", fontsize=10.6, color=INK)
    fig.text(.70, y, "×", color=BAD, va="center", fontsize=16, weight="bold")
    fig.text(.72, y, "phase mismatch", va="center", fontsize=10.6, color=INK)


def footnote(template):
    name = template["name"]
    if name in ("cycloaddition_2plus2", "cycloaddition_4plus4"):
        return "Antarafacial allowance is formal: geometry may prevent this approach, especially for ethene. Products show connectivity only."
    if name in ("claisen_3_3", "cope_3_3", "alder_ene"):
        return "π/σ and transition-state local-basis drawings use different representations; neither is a calculated molecular HOMO."
    if name.startswith("electrocyclic"):
        return "Rotation arrows follow the same viewing convention at both termini. Product connectivity is shown without substituent stereochemistry."
    return "Cycloaddition panels show the diagnostic HOMO–LUMO pair. Changing one approach face changes the relative overlap sign."


def make_pair(template):
    fig = plt.figure(figsize=(18.8, 12.6), facecolor="white")
    fig.text(.04, .970, NAMES[template["name"]] + "  |  orbital phases along the reaction",
             fontsize=23, weight="bold", color=INK, va="top")
    for x, title in ((.181, "REACTANT ORBITALS"), (.530, "PROPOSED TRANSITION STATE"),
                     (.867, "PRODUCT")):
        fig.text(x, .923, title, ha="center", fontsize=13, weight="bold", color=MUTED)
    modes = sorted(template["modes"], key=lambda mp: not ((len(template["sites"]) % 4 == 2) != bool(mp[1])))
    row(fig, .535, .325, template, *modes[0])
    row(fig, .145, .325, template, *modes[1])
    fig.add_artist(plt.Line2D([.04, .97], [.51, .51], transform=fig.transFigure, color=FAINT, lw=1.2))
    figure_legend(fig, .101)
    fig.text(.043, .063, footnote(template), fontsize=10.6, color=MUTED)
    fig.text(.043, .033,
             "Thermal, concerted pathways. Colored lobes are qualitative phase schematics; the middle panels are not optimized TS geometries or TS eigenorbitals.",
             fontsize=10.2, color=MUTED)
    return fig


def make_overview(templates):
    """Four diagnostic rows, all retaining reactant -> TS -> product structure."""
    selected = [("diels_alder_4plus2", "supra_supra"),
                ("cycloaddition_2plus2", "supra_supra"),
                ("electrocyclic_4pi", "conrotatory"),
                ("electrocyclic_4pi", "disrotatory")]
    fig = plt.figure(figsize=(18.8, 21), facecolor="white")
    fig.text(.04, .978, "Why the orbital phases allow—or block—the reaction",
             fontsize=25, weight="bold", color=INK, va="top")
    fig.text(.043, .944, "Reactant frontier orbitals → proposed transition state → product connectivity", fontsize=14, color=MUTED)
    figure_legend(fig, .919)
    for j, (name, mode) in enumerate(selected):
        template = templates[name]
        parity = dict(template["modes"])[mode]
        y0 = .695 - .205 * j
        fig.text(.043, y0 + .174, NAMES[name], fontsize=16.5, weight="bold", color=INK)
        row(fig, y0, .145, template, mode, parity, compact=True)
        if j < 3:
            fig.add_artist(plt.Line2D([.04, .97], [y0 - .012, y0 - .012],
                                      transform=fig.transFigure, color=FAINT, lw=1.1))
    fig.text(.043, .043,
             "Hückel + 6 electrons and Möbius + 4 electrons are thermally allowed. Red pathways are forbidden for the specified concerted approach.",
             fontsize=11, color=INK)
    fig.text(.043, .024,
             "Schematic orbital phases, not calculated transition-state densities. A crossed arrow does not prohibit a different mechanism or photochemical route.",
             fontsize=10.5, color=MUTED)
    return fig


def export(fig, stem, pdf=None):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(FIGS / f"{stem}.{suffix}", dpi=180, facecolor="white", bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", stem, flush=True)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    templates, dataset_manifest = load_and_validate()
    lookup = {t["name"]: t for t in templates}
    export(make_overview(lookup), "orbital_selection_overview")
    with PdfPages(OUT / "pericyclic_orbital_schemes.pdf") as packet:
        for name in ORDER:
            export(make_pair(lookup[name]), name, packet)
    manifest = {"drawing_type": "qualitative orbital phase schematics; no TS optimization",
                "frontier_model": "uniform open-chain Hii=0, nearest-neighbor Hij=-1",
                "lobe_sizes": "compressed illustrative scale; not density isosurfaces",
                "frontier_coefficients": MO_RECORDS,
                "dataset_checks": dataset_manifest,
                "reaction_bonds": [{"family": name, "elements": lookup[name]["sites"],
                                    "reactant_cyclic_bond_orders": orders(lookup[name]),
                                    "product_cyclic_bond_orders": orders(lookup[name], True),
                                    "expected_product_smiles": PRODUCTS[name][1]} for name in ORDER]}
    (OUT / "orbital_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified open-chain eigenvectors, frontier-contact phases, and 16 dataset labels.")


if __name__ == "__main__":
    main()
