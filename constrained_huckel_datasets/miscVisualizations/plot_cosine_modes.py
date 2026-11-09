#!/usr/bin/env python3
"""Plot three separate N=3 Hückel cosine modes and their frequencies."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N = 3
ALPHA = 0.0
BETA = -1.0
OUTPUT_DIR = Path(__file__).resolve().parent
COLORS = ("#2356b8", "#009e73", "#a326a5")
INK = "#18263b"


def cosine_mode(k: int, position: np.ndarray | float) -> np.ndarray | float:
    """Continuous extension of alpha + 2 beta cos(2 pi k j / N)."""
    return ALPHA + 2 * BETA * np.cos(2 * np.pi * k * position / N)


def plot_mode(k: int) -> Path:
    position = np.linspace(0, N, 900)
    sites = np.arange(N)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.21, top=0.82)
    ax.plot(position, cosine_mode(k, position), color=COLORS[k], lw=2.8)
    ax.scatter(sites, cosine_mode(k, sites), s=95, facecolor="white",
               edgecolor=COLORS[k], lw=2.2, zorder=3)
    ax.scatter([1], [cosine_mode(k, 1)], s=105, color=COLORS[k],
               edgecolor="white", lw=1.5, zorder=4)
    ax.axvline(1, color="#9aa8ba", lw=1.1, ls="--")
    ax.axhline(0, color=INK, lw=0.9, alpha=0.6)
    ax.set_xlim(0, N)
    ax.set_ylim(-2.45, 2.45)
    ax.set_xticks(range(N + 1))
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("Position j around the ring", color=INK)
    ax.set_ylabel(r"$f_k(j)=\alpha+2\beta\cos(2\pi k j/3)$", color=INK)
    ax.set_title(rf"Cosine mode $k={k}$", fontsize=17, weight="bold",
                 color=INK, pad=18)
    ax.grid(alpha=0.22)
    ax.text(0.98, 0.94, rf"$E_{k}=f_{k}(1)={cosine_mode(k, 1):+.0f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color=COLORS[k],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    fig.text(0.5, 0.07,
             "Dots mark the three sites j = 0, 1, 2; the line shows the cosine between them.",
             ha="center", fontsize=10, color="#52627a")
    path = OUTPUT_DIR / f"cosine_k{k}.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def mode_amplitude(k: int) -> float:
    """One-sided Fourier amplitude of the continuous mode."""
    sample_count = 1536
    position = np.linspace(0, N, sample_count, endpoint=False)
    spectrum = np.fft.rfft(cosine_mode(k, position))
    factor = 1 if k == 0 else 2
    return float(factor * abs(spectrum[k]) / sample_count)


def plot_frequency_spectrum() -> Path:
    modes = np.arange(N)
    frequencies = modes / N
    amplitudes = np.array([mode_amplitude(k) for k in modes])
    fig, ax = plt.subplots(figsize=(8, 4.7))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.23, top=0.82)
    bars = ax.bar(frequencies, amplitudes, width=0.13, color=COLORS,
                  edgecolor="white", linewidth=1.3)
    for k, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                f"k = {k}", ha="center", va="bottom", color=COLORS[k],
                fontsize=11, weight="bold")
    ax.set_xlim(-0.14, 0.81)
    ax.set_ylim(0, max(amplitudes) * 1.23)
    ax.set_xticks(frequencies, ["0 (DC)", "1/3", "2/3"])
    ax.set_ylabel("One-sided Fourier amplitude", color=INK)
    ax.set_xlabel("Mode frequency k/3 (cycles per site)", color=INK)
    ax.set_title("Frequency spectrum of the three cosine modes",
                 fontsize=17, weight="bold", color=INK, pad=18)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.075,
             "Frequencies come from the continuous curves; at the three sites, k = 1 and k = 2 alias.",
             ha="center", fontsize=10, color="#52627a")
    path = OUTPUT_DIR / "frequency_spectrum.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for k in range(N):
        assert np.isclose(cosine_mode(k, 1),
                          ALPHA + 2 * BETA * np.cos(2 * np.pi * k / N))
        print(plot_mode(k))
    print(plot_frequency_spectrum())


if __name__ == "__main__":
    main()
