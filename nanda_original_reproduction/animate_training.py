#!/usr/bin/env python3
"""Animate the saved seed-0 modular-addition training trajectory.

The four GIFs show logits, input embeddings, Fourier Gini, and Fourier
frequency power at the archived model steps. No retraining is required.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from train_modular import P, OriginalStyleTransformer, gini


ROOT = Path(__file__).resolve().parent
BLUE = "#2563eb"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
GRAY = "#64748b"


def load_snapshots(run_dir: Path):
    paths = sorted((run_dir / "figure_snapshots").glob("step_*.pt"))
    if len(paths) < 2:
        raise ValueError(f"Need at least two archived snapshots in {run_dir / 'figure_snapshots'}")
    snapshots = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        step = int(checkpoint["step"])
        if step != int(path.stem.split("_")[-1]):
            raise ValueError(f"Step mismatch in {path}")
        snapshots.append((step, checkpoint["model"]))
    steps = np.array([step for step, _ in snapshots])
    if steps[0] != 0 or np.any(np.diff(steps) <= 0):
        raise ValueError("Snapshots must start at step 0 and increase strictly")
    return snapshots


def load_metrics(run_dir: Path):
    with (run_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = ("step", "train_accuracy", "test_accuracy",
              "embedding_fourier_gini", "neuron_logit_fourier_gini")
    values = {field: np.array([float(row[field]) for row in rows]) for field in fields}
    if not len(rows) or np.any(np.diff(values["step"]) <= 0):
        raise ValueError("Metrics must contain strictly increasing steps")
    return values


def final_embedding_pca(state):
    embedding = state["W_E"][:, :P].numpy().T.astype(np.float64)
    mean = embedding.mean(axis=0)
    _, singular, basis = np.linalg.svd(embedding - mean, full_matrices=False)
    explained = singular[:2] @ singular[:2] / (singular @ singular)
    projected = (embedding - mean) @ basis[:2].T
    power = np.abs(np.fft.rfft(projected, axis=0)) ** 2
    frequency = int(np.argmax(power[1:].sum(axis=1)) + 1)
    return mean, basis[:2], explained, frequency


@torch.inference_mode()
def collect_frames(snapshots, device: torch.device, fixed_b: int):
    model = OriginalStyleTransformer().to(device).eval()
    a = np.arange(P)
    tokens = torch.as_tensor(np.column_stack((a, np.full(P, fixed_b), np.full(P, P))),
                             dtype=torch.long, device=device)
    logits = []
    embeddings = []
    embedding_power = []
    logit_power = []
    for step, state in snapshots:
        weights = state["W_E"][:, :P].numpy()
        logit_map = state["W_U"][:, :P].numpy().T @ state["W_out"].numpy()
        embedding_power.append((np.abs(np.fft.rfft(weights, axis=1))[:, 1:] ** 2)
                               .sum(axis=0))
        logit_power.append((np.abs(np.fft.rfft(logit_map, axis=0))[1:] ** 2)
                           .sum(axis=1))
        embeddings.append(weights.T.astype(np.float64))
        model.load_state_dict(state)
        raw = model(tokens)[:, :P].detach().cpu().numpy().astype(np.float64)
        logits.append(raw - raw.mean(axis=1, keepdims=True))
        print(f"Analyzed step {step:,}", flush=True)
    embedding_power = np.asarray(embedding_power)
    logit_power = np.asarray(logit_power)
    return (np.asarray(logits), np.asarray(embeddings),
            embedding_power / embedding_power.sum(axis=1, keepdims=True),
            logit_power / logit_power.sum(axis=1, keepdims=True))


def frame_title(fig, name: str, step: int, final_step: int, metrics):
    if step < metrics["step"][0]:
        accuracy = "test accuracy: not measured"
    else:
        accuracy = f"test accuracy: {np.interp(step, metrics['step'], metrics['test_accuracy']):.1%}"
    fig.suptitle(f"{name}  |  step {step:,} / {final_step:,}  |  {accuracy}",
                 fontsize=12, y=.98)


def style_axis(ax):
    ax.grid(color="#cbd5e1", alpha=.35, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)


def save_gif(path: Path, count: int, fps: int, draw):
    frames = []
    for index in range(count):
        fig = draw(index)
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))
        plt.close(fig)
    duration = [round(1000 / fps)] * count
    duration[-1] = 1600
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, disposal=2, optimize=False)
    print(f"Saved {path} ({count} frames)", flush=True)


def draw_logits(index, steps, metrics, logits, fixed_a, fixed_b):
    values = logits[index]
    scale = max(float(np.percentile(np.abs(values), 99)), 1e-12)
    display = np.clip(values / scale, -1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=100,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    image = axes[0].imshow(display, origin="lower", aspect="equal", cmap="coolwarm",
                           vmin=-1, vmax=1, interpolation="nearest")
    axes[0].plot((np.arange(P) + fixed_b) % P, np.arange(P),
                 ".", markersize=1.4, color="#111827", alpha=.8)
    axes[0].set(xlabel="Output class c", ylabel="Input a",
                title=f"Logit map with b={fixed_b}\nblack dots: correct c=(a+b) mod {P}")
    fig.colorbar(image, ax=axes[0], shrink=.75, label="Centered logit / frame scale")
    target = (fixed_a + fixed_b) % P
    axes[1].plot(np.arange(P), display[fixed_a], color=BLUE, linewidth=1.4)
    axes[1].axvline(target, color=ORANGE, linestyle="--", linewidth=1.2,
                    label=f"correct class {target}")
    axes[1].set(xlim=(0, P - 1), ylim=(-1.08, 1.08), xlabel="Output class c",
                ylabel="Centered logit / frame scale",
                title=f"Logit profile for a={fixed_a}, b={fixed_b}")
    axes[1].legend(loc="upper right", fontsize=9)
    style_axis(axes[1])
    fig.text(.61, .06, f"Frame scale = {scale:.3g} raw logit units; each frame is rescaled",
             fontsize=9, color=GRAY)
    frame_title(fig, "Logit space", int(steps[index]), int(steps[-1]), metrics)
    fig.subplots_adjust(left=.07, right=.97, top=.82, bottom=.15, wspace=.28)
    return fig


def draw_embeddings(index, steps, metrics, projections, limit, frequency, explained):
    points = projections[index]
    phase = (frequency * np.arange(P)) % P
    order = np.argsort(phase)
    fig, ax = plt.subplots(figsize=(7.2, 6.3), dpi=100)
    ax.plot(points[order, 0], points[order, 1], color="#94a3b8", lw=.8, alpha=.7)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=phase, cmap="twilight",
                         vmin=0, vmax=P, s=28, edgecolors="none", zorder=2)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Final embedding PC1",
           ylabel="Final embedding PC2",
           title=f"113 integer tokens in fixed final PCA plane\n"
                 f"Final PC1+PC2 variance: {explained:.1%}; phase frequency k={frequency}")
    ax.set_aspect("equal", adjustable="box")
    style_axis(ax)
    fig.colorbar(scatter, ax=ax, shrink=.72, label=f"Phase (k × token) mod {P}")
    frame_title(fig, "Weight embedding space", int(steps[index]), int(steps[-1]), metrics)
    fig.subplots_adjust(left=.12, right=.91, top=.82, bottom=.12)
    return fig


def draw_gini(index, steps, metrics, initial):
    step = steps[index]
    x = metrics["step"]
    visible = x <= step
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), dpi=100, sharex=True)
    specs = [("embedding_fourier_gini", BLUE, "Embedding Fourier Gini"),
             ("neuron_logit_fourier_gini", PURPLE, "Neuron-logit Fourier Gini")]
    for key, color, label in specs:
        shown_x = np.concatenate(([0], x[visible]))
        shown_y = np.concatenate(([initial[key]], metrics[key][visible]))
        axes[0].plot(shown_x, shown_y, color=color, lw=2, label=label)
        axes[0].plot(shown_x[-1], shown_y[-1], "o", color=color, ms=5)
    axes[0].set(ylabel="Gini coefficient", ylim=(0, 1.02),
                title="Concentration of nonzero Fourier power")
    axes[0].legend(loc="upper left", fontsize=9)
    for key, color, label in [("train_accuracy", BLUE, "Train"),
                              ("test_accuracy", ORANGE, "Test")]:
        axes[1].plot(x[visible], metrics[key][visible], color=color, lw=2, label=label)
    axes[1].set(xlabel="Optimizer step", ylabel="Accuracy", ylim=(0, 1.03),
                xlim=(0, steps[-1]), title="Learning curve")
    axes[1].legend(loc="lower right", fontsize=9)
    for ax in axes:
        ax.axvline(step, color=GRAY, lw=1, linestyle=":")
        style_axis(ax)
    frame_title(fig, "Fourier Gini", int(step), int(steps[-1]), metrics)
    fig.subplots_adjust(left=.1, right=.97, top=.82, bottom=.12, hspace=.35)
    return fig


def draw_frequency(index, steps, metrics, embedding_power, logit_power,
                   limit, final_keys):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), dpi=100, sharex=True)
    frequencies = np.arange(1, P // 2 + 1)
    for ax, values, title, color in [
        (axes[0], embedding_power[index], "Embedding W_E", BLUE),
        (axes[1], logit_power[index], "Neuron-logit map W_L", PURPLE),
    ]:
        ax.bar(frequencies, values, width=.9, color=color)
        for key in final_keys:
            ax.axvspan(key - .5, key + .5, color=ORANGE, alpha=.14, zorder=0)
        strongest = np.argsort(values)[-4:][::-1] + 1
        ax.set(ylabel="Fraction of Fourier power", ylim=(0, limit),
               title=f"{title}  |  top current k: {', '.join(map(str, strongest))}")
        style_axis(ax)
    axes[1].set(xlim=(.3, P // 2 + .7), xlabel="Fourier frequency k (mod 113)")
    fig.text(.1, .035, f"Orange bands: final active neuron-logit frequencies {final_keys}",
             fontsize=9, color=GRAY)
    frame_title(fig, "Frequency space", int(steps[index]), int(steps[-1]), metrics)
    fig.subplots_adjust(left=.1, right=.97, top=.81, bottom=.13, hspace=.38)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "figure_run_seed0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "animations")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--fixed-a", type=int, default=17)
    parser.add_argument("--fixed-b", type=int, default=29)
    args = parser.parse_args()
    if args.fps < 1 or not (0 <= args.fixed_a < P and 0 <= args.fixed_b < P):
        parser.error("fps must be positive and fixed operands must be in 0..112")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    snapshots = load_snapshots(args.run_dir)
    metrics = load_metrics(args.run_dir)
    steps = np.array([step for step, _ in snapshots])
    if metrics["step"][-1] < steps[-1]:
        raise ValueError("Metrics do not cover the final snapshot")
    mean, basis, explained, frequency = final_embedding_pca(snapshots[-1][1])
    logits, embeddings, embedding_power, logit_power = collect_frames(
        snapshots, device, args.fixed_b)
    projections = (embeddings - mean) @ basis.T
    limit = float(np.max(np.abs(projections)) * 1.08)
    power_limit = float(max(embedding_power.max(), logit_power.max()) * 1.1)
    final_keys = (np.flatnonzero(logit_power[-1] >= .01 * logit_power[-1].max()) + 1).tolist()
    initial_gini = {"embedding_fourier_gini": gini(embedding_power[0]),
                    "neuron_logit_fourier_gini": gini(logit_power[0])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = len(steps)
    save_gif(args.output_dir / "logit_space.gif", count, args.fps,
             lambda i: draw_logits(i, steps, metrics, logits, args.fixed_a, args.fixed_b))
    save_gif(args.output_dir / "embedding_space.gif", count, args.fps,
             lambda i: draw_embeddings(i, steps, metrics, projections, limit,
                                       frequency, explained))
    save_gif(args.output_dir / "fourier_gini.gif", count, args.fps,
             lambda i: draw_gini(i, steps, metrics, initial_gini))
    save_gif(args.output_dir / "frequency_space.gif", count, args.fps,
             lambda i: draw_frequency(i, steps, metrics, embedding_power,
                                      logit_power, power_limit, final_keys))


if __name__ == "__main__":
    main()
