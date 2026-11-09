#!/usr/bin/env python3
"""Draw Nanda's model and the circulant-gap architectures with measured scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from analyze_gap import final_prediction
from train_gap import ROOT, load_data, score


COLORS = {"nanda": "#334155", "transformer": "#7c3aed", "mlp": "#2563eb",
          "linear": "#059669"}


def box(ax, x, y, width, height, label, face, font=10):
    patch = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.008,rounding_size=0.015",
                           linewidth=1, edgecolor=face, facecolor=face, alpha=.10)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            fontsize=font, color="#0f172a", linespacing=1.35)


def arrow(ax, left, right, y):
    ax.add_patch(FancyArrowPatch((left, y), (right, y), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.4, color="#64748b"))


def flow(ax, items, y, color, xs, widths, height=.17):
    for x, width, label in zip(xs, widths, items):
        box(ax, x, y, width, height, label, color)
    for i in range(len(items) - 1):
        arrow(ax, xs[i] + widths[i] + .012, xs[i + 1] - .012, y + height / 2)


def model_scores(device):
    n, gap = load_data()
    extended_n = np.arange(40_006, 60_003, 4, dtype=np.int64)
    extended_gap = 4 * np.sin(np.pi / extended_n)
    choices = [
        ("Shared-bit transformer", "log_bits", ROOT / "runs_decay", "transformer"),
        ("Position-aware transformer", "log_posbits", ROOT / "runs_decay", "transformer"),
        ("Flat-bit MLP", "log_flatbits", ROOT / "runs_decay", "mlp"),
        ("log(n) MLP", "log_scalar", ROOT / "runs", "mlp"),
    ]
    results = []
    for label, variant, run_root, kind in choices:
        pred, _, test = final_prediction(variant, device, n, run_root)
        extended_pred, _, _ = final_prediction(variant, device, extended_n, run_root)
        results.append({"name": label, "kind": kind,
                        "heldout_1pct": score(pred, gap, test)["accuracy_rel_1pct"],
                        "larger_rings_1pct": score(extended_pred, extended_gap,
                                                   np.arange(len(extended_n)))["accuracy_rel_1pct"]})
    log_n = np.log(n)
    train = final_prediction("log_scalar", device, n)[1]
    test = final_prediction("log_scalar", device, n)[2]
    slope, intercept = np.polyfit(log_n[train], np.log(gap[train]), 1)
    predicted = np.exp(intercept + slope * log_n)
    extended_predicted = np.exp(intercept + slope * np.log(extended_n))
    results.append({"name": "Log-linear fit", "kind": "linear",
                    "heldout_1pct": score(predicted, gap, test)["accuracy_rel_1pct"],
                    "larger_rings_1pct": score(extended_predicted, extended_gap,
                                               np.arange(len(extended_n)))["accuracy_rel_1pct"]})
    return results


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = model_scores(device)
    fig = plt.figure(figsize=(17, 11), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 2.7], width_ratios=[3.1, 1.35])
    nanda = fig.add_subplot(grid[0, :])
    ours = fig.add_subplot(grid[1, 0])
    bars = fig.add_subplot(grid[1, 1])
    for ax in (nanda, ours):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    nanda.text(.01, .94, "Nanda et al. · modular addition mod 113", fontsize=15,
               weight="bold", color=COLORS["nanda"], va="top")
    flow(nanda, ["tokens [a, b, =]\n3 positions", "learned token +\nposition embeddings\nd = 128",
                 "1 causal attention layer\n4 heads + residual", "ReLU MLP 512\n+ residual",
                 "last token → 114 logits\ncross-entropy"],
         .44, COLORS["nanda"], [.012, .216, .42, .624, .828], [.155] * 5, .29)
    nanda.text(.01, .26, "Our local reproduction: test accuracy reached 100% after delayed generalization.",
               fontsize=10, color="#475569")
    nanda.text(.01, .12, "This classification score and the gap-regression scores below measure different tasks.",
               fontsize=10, color="#475569")

    ours.text(.01, .98, "Our circulant gap probes · same 3,000/7,000 random size split",
              fontsize=15, weight="bold", color="#0f172a", va="top")
    rows = [
        (["16 binary digits\nshared 0/1 token", "learned position + CLS", "1 attention layer\n4 heads; MLP 512", "log gap"], "transformer"),
        (["16 binary digits\nposition-specific token", "position-specific 0/1\nembedding + CLS", "1 attention layer\n4 heads; MLP 512", "log gap"], "transformer"),
        (["16 ordered bit features", "explicit bit positions", "2-layer ReLU MLP\n128 hidden units", "log gap"], "mlp"),
        (["scalar log(n)", "normalized scalar", "2-layer ReLU MLP\n128 hidden units", "log gap"], "mlp"),
        (["scalar log(n)", "slope + intercept fit", "a + b log(n)", "log gap"], "linear"),
    ]
    y_centers = [.83, .65, .47, .29, .11]
    for (items, kind), center in zip(rows, y_centers):
        flow(ours, items, center - .065, COLORS[kind],
             [.01, .265, .52, .805], [.20, .20, .23, .16], .13)
    ours.text(.01, .005, "Transformer and flat-bit MLP rows use cosine LR decay; log(n) MLP uses constant LR.",
              fontsize=9, color="#64748b")

    names = [entry["name"] for entry in results]
    yy = np.arange(len(results))
    width = .36
    bars.barh(yy - width / 2, [r["heldout_1pct"] for r in results], height=width,
              color="#2563eb", label="random held-out sizes")
    bars.barh(yy + width / 2, [r["larger_rings_1pct"] for r in results], height=width,
              color="#f97316", label="new larger rings")
    bars.set_yticks(yy, names)
    bars.invert_yaxis()
    bars.set(xlim=(0, 1.04), xlabel="Within 1% relative gap error",
             title="Measured generalization")
    bars.grid(axis="x", alpha=.2)
    bars.legend(loc="lower right", fontsize=8)
    bars.text(0, len(results) + .45, "New larger rings: n = 40,006…60,002; no retraining.",
              fontsize=9, color="#475569")
    fig.suptitle("Architecture and representation explain the gap results", fontsize=19,
                 weight="bold")
    fig.savefig(ROOT / "architecture_map.svg", bbox_inches="tight")
    fig.savefig(ROOT / "architecture_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (ROOT / "architecture_scores.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
