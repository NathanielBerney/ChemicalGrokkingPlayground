#!/usr/bin/env python3
"""Explain cyclic and Möbius Hückel matrices as modular shift operators.

The n=8 example is chosen for legibility. It illustrates the indexing rule,
not the neutral closed-shell selection used in the generated datasets.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
import numpy as np


N = 8
BLUE = "#2356b8"
MAGENTA = "#a326a5"
RED = "#d64b3d"
DARK = "#18263b"
OUT = Path(__file__).resolve().parent / "constrained_huckel_datasets"


def signed_shift(n: int, seam_sign: int) -> np.ndarray:
    """T[i,i+1]=1, T[n-1,0]=seam_sign."""
    shift = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        shift[i, i + 1] = 1
    shift[n - 1, 0] = seam_sign
    return shift


def draw_ring(ax, n: int, seam_sign: int, color: str) -> None:
    theta = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    xy = np.column_stack((np.cos(theta), np.sin(theta)))
    seam_color = "#c58d1f" if seam_sign > 0 else RED
    for i in range(n):
        j = (i + 1) % n
        is_seam = i == n - 1
        edge_color = seam_color if is_seam else color
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                color=edge_color, lw=3.8 if is_seam else 2.3,
                ls="--" if is_seam and seam_sign < 0 else "-", zorder=1)
    for i, (x, y) in enumerate(xy):
        ax.add_patch(Circle((x, y), 0.115, facecolor="white", edgecolor=color,
                            lw=2.0, zorder=2))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=10,
                weight="bold", color=DARK, zorder=3)
    seam_midpoint = (xy[0] + xy[-1]) / 2
    ax.annotate("7 → 0\nwrap", seam_midpoint, xytext=(-1.65, 1.42),
                ha="center", color=seam_color, weight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=seam_color, lw=1.5))
    ax.text(0, -1.38, "signed wrap:  −1" if seam_sign < 0 else "ordinary wrap:  +1",
            ha="center", va="center", fontsize=11, color=color, weight="bold")
    ax.set_xlim(-1.8, 1.45)
    ax.set_ylim(-1.56, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_matrix(ax, h: np.ndarray, color: str) -> None:
    cmap = LinearSegmentedColormap.from_list("coupling", [BLUE, "#ffffff", RED])
    ax.imshow(h, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xlabel("site j", fontsize=10)
    ax.set_ylabel("site i", fontsize=10)
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="#c9d2e0", lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(N):
        for j in range(N):
            if h[i, j] != 0:
                ax.text(j, i, f"{int(h[i, j]):+d}", ha="center", va="center",
                        color="white" if h[i, j] < 0 else DARK,
                        weight="bold", fontsize=10)
    for x, y in ((N - 1, 0), (0, N - 1)):
        ax.add_patch(Rectangle((x - 0.48, y - 0.48), 0.96, 0.96,
                               fill=False, edgecolor="#e9b949", lw=3.1))
    ax.text(0.5, -0.16, "gold boxes = wraparound bond", transform=ax.transAxes,
            ha="center", color="#7c5200", fontsize=9)


def draw_momenta(ax, n: int, seam_sign: int, color: str) -> None:
    base = 2 * np.pi * np.arange(n) / n
    allowed = base + (np.pi / n if seam_sign < 0 else 0)
    for q in base:
        ax.axvline(q, ymin=0.23, ymax=0.64, color="#cdd6e2", lw=1.2, zorder=0)
    ax.axhline(0, color=DARK, lw=1)
    ax.vlines(allowed, 0, 0.76, color=color, lw=2.6)
    ax.scatter(allowed, np.full(n, 0.76), s=75, color=color,
               edgecolor="white", lw=1.2, zorder=3)
    if seam_sign < 0:
        ax.annotate("half-step\nshift", (np.pi / n, 0.76), (1.0, 1.11),
                    color=color, fontsize=10, weight="bold", ha="center",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    ax.set_xlim(-0.27, 2 * np.pi + 0.15)
    ax.set_ylim(-0.25, 1.48)
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", "π/2", "π", "3π/2", "2π"])
    ax.set_xlabel("orbital phase increment q", fontsize=10)
    ax.spines[["left", "right", "top"]].set_visible(False)
    boundary = r"$\psi(j+8)=+\psi(j)$" if seam_sign > 0 else r"$\psi(j+8)=-\psi(j)$"
    q_formula = r"$q_m=2\pi m/8$" if seam_sign > 0 else r"$q_m=2\pi(m+1/2)/8$"
    t_formula = r"$T_+^8=+I$" if seam_sign > 0 else r"$T_-^8=-I$"
    ax.text(0.5, -0.18, f"{boundary}     {q_formula}     {t_formula}",
            transform=ax.transAxes, ha="center", fontsize=11, color=color)


def make_figure() -> tuple[Path, Path]:
    ordinary = signed_shift(N, +1)
    twisted = signed_shift(N, -1)
    assert np.array_equal(np.linalg.matrix_power(ordinary, N), np.eye(N, dtype=int))
    assert np.array_equal(np.linalg.matrix_power(twisted, N), -np.eye(N, dtype=int))
    matrices = [-ordinary - ordinary.T, -twisted - twisted.T]
    for h, offset in zip(matrices, (0.0, 0.5)):
        q = 2 * np.pi * (np.arange(N) + offset) / N
        assert np.allclose(np.sort(np.linalg.eigvalsh(h)),
                           np.sort(-2 * np.cos(q)), atol=1e-12)

    fig = plt.figure(figsize=(16.5, 10.2), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.965, top=0.84,
                           bottom=0.15, wspace=0.19, hspace=0.33,
                           width_ratios=(1.0, 1.03, 1.63))
    fig.suptitle("From modular indexing to Hückel orbital phases", x=0.5, y=0.975,
                 fontsize=22, weight="bold", color=DARK)
    fig.text(0.5, 0.911,
             "For both rings, neighbors satisfy  j ≡ i ± 1  (mod 8).  The Möbius seam changes the sign when the index wraps.",
             ha="center", fontsize=13, color="#42546e")

    for row, (title, seam_sign, color, h) in enumerate(zip(
        ("Ordinary circulant ring", "Möbius signed ring"),
        (+1, -1), (BLUE, MAGENTA), matrices)):
        ring_ax = fig.add_subplot(grid[row, 0])
        matrix_ax = fig.add_subplot(grid[row, 1])
        phase_ax = fig.add_subplot(grid[row, 2])
        ring_ax.set_title(title, loc="left", fontsize=15, weight="bold", color=color, pad=5)
        t_label = "+" if seam_sign > 0 else "-"
        matrix_ax.set_title(
            "Hückel matrix  " + rf"$H=-(T_{{{t_label}}}+T_{{{t_label}}}^T)$",
            fontsize=12, color=DARK, pad=13)
        phase_ax.set_title("Allowed orbital phases", fontsize=12, color=DARK, pad=13)
        draw_ring(ring_ax, N, seam_sign, color)
        draw_matrix(matrix_ax, h, color)
        draw_momenta(phase_ax, N, seam_sign, color)

    fig.text(0.5, 0.055,
             "A circulant matrix uses periodic modular wraparound.  One negative wrap gives an anti-periodic boundary: its allowed q values move by half a spacing.",
             ha="center", va="center", fontsize=12, color=DARK,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#edf3fa", edgecolor="none"))
    fig.text(0.5, 0.021,
             "Illustrative n = 8 and α = 0, β = −1.  The Möbius seam denotes an orbital phase flip, not a negative chemical bond or a 3D geometry.",
             ha="center", fontsize=9, color="#67768a")
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "modular_huckel_explainer.png"
    svg = OUT / "modular_huckel_explainer.svg"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


if __name__ == "__main__":
    png_path, svg_path = make_figure()
    print(png_path)
    print(svg_path)
