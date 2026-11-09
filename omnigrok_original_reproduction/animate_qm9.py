#!/usr/bin/env python3
"""Animate QM9 predictions, graph embeddings, and weight trajectories.

Uses the archived reruns in runs/grokking_total{N}_alpha{1,3}_animation.
Each animation aligns the two initialization scales at saved optimizer steps.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from reproduce_qm9 import DATA_ROOT, ROOT, TARGET, ExampleNet


OUT = ROOT / "animations"
BLUE = "#2563eb"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
GRAY = "#64748b"


def metrics_table(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def pca(rows):
    centered = rows - rows.mean(axis=0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    fraction = float(np.sum(singular[:2] ** 2) / np.sum(singular ** 2))
    return basis[:2], fraction


def weight_pca(weights):
    centered = weights - weights.mean(axis=0)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    scores = eigenvectors[:, order[:2]] * np.sqrt(eigenvalues[:2])
    for column in range(2):
        if scores[-1, column] < scores[0, column]:
            scores[:, column] *= -1
    fraction = float(eigenvalues[:2].sum() / eigenvalues.sum())
    return scores, fraction


@torch.inference_mode()
def predict_and_embed(model, loader, device):
    predictions = []
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        x = F.relu(model.conv1(batch.x, batch.edge_index, batch.edge_attr))
        x = F.relu(model.conv2(x, batch.edge_index, batch.edge_attr))
        pooled = global_add_pool(x, batch.batch)
        prediction = model.out(F.relu(model.fc_1(pooled)))
        predictions.append(prediction.cpu().numpy().ravel())
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embeddings)


def load_run(total_size, alpha, loader, device):
    run = ROOT / "runs" / f"grokking_total{total_size}_alpha{alpha}_animation"
    config = json.loads((run / "config.json").read_text())
    files = sorted((run / "snapshots").glob("step_*.pt"))
    if not (run / "last.pt").exists() or len(files) < 2:
        raise ValueError(f"Archived run is incomplete: {run}")
    metrics = metrics_table(run / "full_metrics.csv")
    model = ExampleNet().to(device).eval()
    steps = []
    predictions = []
    embeddings = []
    weights = []
    for path in files:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        step = int(checkpoint["step"])
        if step != int(path.stem.split("_")[-1]):
            raise ValueError(f"Step mismatch in {path}")
        state = checkpoint["model"]
        model.load_state_dict(state)
        prediction, embedding = predict_and_embed(model, loader, device)
        steps.append(step)
        predictions.append(prediction)
        embeddings.append(embedding)
        weights.append(np.concatenate([value.numpy().ravel() for value in state.values()]))
    steps = np.asarray(steps)
    if steps[0] != 0 or steps[-1] != metrics["step"][-1] or np.any(np.diff(steps) <= 0):
        raise ValueError(f"Incomplete or unordered snapshots in {run}")
    interval = int(config["archive_every_steps"])
    if interval < 1 or np.any(np.diff(steps[:-1]) != interval) or steps[-1] - steps[-2] > interval:
        raise ValueError(f"Missing archived steps in {run}")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    basis, embedding_fraction = pca(embeddings[-1])
    # The PCA directions are fixed to the final representation. Each frame is
    # centered and divided by its own 2-D RMS radius so large initial norms
    # do not hide changes in geometry; the raw radius is reported separately.
    projected = np.einsum("tnd,kd->tnk", embeddings - embeddings.mean(axis=1, keepdims=True), basis)
    radius = np.sqrt(np.mean(np.sum(projected ** 2, axis=2), axis=1))
    normalized = projected / np.maximum(radius[:, None, None], 1e-12)
    scores, weight_fraction = weight_pca(np.asarray(weights, dtype=np.float64))
    return {"config": config, "steps": steps, "prediction": np.asarray(predictions),
            "embedding": normalized, "embedding_radius": radius,
            "embedding_pca_fraction": embedding_fraction, "weight_scores": scores,
            "weight_pca_fraction": weight_fraction, "metrics": metrics,
            "weight_norm": np.linalg.norm(np.asarray(weights, dtype=np.float64), axis=1)}


def validate_pair(total_size, runs, truth):
    one, three = runs[1], runs[3]
    if not np.array_equal(one["steps"], three["steps"]):
        raise ValueError(f"α=1 and α=3 snapshots have different steps for N={total_size}")
    if one["config"]["train_indices"] != three["config"]["train_indices"]:
        raise ValueError(f"α=1 and α=3 use different train splits for N={total_size}")
    mask = np.zeros(total_size, dtype=bool)
    mask[one["config"]["train_indices"]] = True
    for alpha, run in runs.items():
        for index in (0, len(run["steps"]) - 1):
            step = run["steps"][index]
            metric_index = np.where(run["metrics"]["step"] == step)[0]
            if not len(metric_index):
                raise ValueError(f"Missing full MSE at step {step}")
            train_mse = float(np.mean((run["prediction"][index, mask] - truth[mask]) ** 2))
            test_mse = float(np.mean((run["prediction"][index, ~mask] - truth[~mask]) ** 2))
            logged_train = run["metrics"]["train_mse"][metric_index[0]]
            logged_test = run["metrics"]["test_mse"][metric_index[0]]
            if abs(train_mse - logged_train) > max(1e-3, 1e-4 * logged_train):
                raise ValueError(f"Prediction/train MSE mismatch for α={alpha}, step={step}")
            if abs(test_mse - logged_test) > max(1e-3, 1e-4 * logged_test):
                raise ValueError(f"Prediction/test MSE mismatch for α={alpha}, step={step}")
    return mask


def save_gif(path, count, fps, draw):
    frames = []
    for index in range(count):
        fig = draw(index)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE,
                                                colors=160))
        plt.close(fig)
    durations = [round(1000 / fps)] * count
    durations[-1] = 1700
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2, optimize=False)
    print(f"Saved {path.name}: {count} frames", flush=True)


def current_mse(run, step, key="test_mse"):
    return float(np.interp(step, run["metrics"]["step"], run["metrics"][key]))


def draw_predictions(index, total_size, truth, train_mask, runs, xscale="log"):
    step = int(runs[3]["steps"][index])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=100,
                             gridspec_kw={"height_ratios": [1.4, 1]},
                             layout="constrained")
    xlow, xhigh = float(truth.min()), float(truth.max())
    padding = .07 * (xhigh - xlow)
    for column, alpha in enumerate((1, 3)):
        run = runs[alpha]
        prediction = run["prediction"][index]
        ax = axes[0, column]
        ax.scatter(truth[~train_mask], prediction[~train_mask], color=ORANGE,
                   s=18, alpha=.7, label="test", edgecolors="none")
        ax.scatter(truth[train_mask], prediction[train_mask], color=BLUE,
                   s=18, alpha=.7, label="train", edgecolors="none")
        ax.plot([xlow - padding, xhigh + padding], [xlow - padding, xhigh + padding],
                color="#111827", linestyle="--", lw=1, label="perfect")
        ymin = float(min(prediction.min(), xlow))
        ymax = float(max(prediction.max(), xhigh))
        ypad = max(.06 * (ymax - ymin), 1)
        ax.set(xlim=(xlow - padding, xhigh + padding), ylim=(ymin - ypad, ymax + ypad),
               xlabel="True polarizability", ylabel="Predicted polarizability",
               title=f"α={alpha} · test MSE {current_mse(run, step):.3g}")
        ax.grid(alpha=.25)
        if column == 0:
            ax.legend(fontsize=8, loc="best")
        ax = axes[1, column]
        metrics = run["metrics"]
        visible = (metrics["step"] >= (1 if xscale == "log" else 0)) & (metrics["step"] <= step)
        for key, color, label in (("train_mse", BLUE, "train"),
                                  ("test_mse", ORANGE, "test")):
            ax.plot(metrics["step"][visible], np.maximum(metrics[key][visible], 1e-12),
                    color=color, lw=1.6, label=label)
        ax.set(xscale=xscale, yscale="log",
               xlim=(1 if xscale == "log" else 0, run["steps"][-1] * 1.04),
               ylim=(max(1e-3, min(metrics["train_mse"].min(), metrics["test_mse"].min()) * .6),
                     max(metrics["train_mse"].max(), metrics["test_mse"].max()) * 1.5),
               xlabel="Optimizer step", ylabel="Full-set MSE")
        ax.grid(alpha=.2, which="both")
        if column == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"QM9 predictions · {total_size//2} train / {total_size//2} test · step {step:,}\n"
                 "Prediction vertical axes adjust each frame; loss axes remain fixed",
                 fontsize=13)
    return fig


def draw_embeddings(index, total_size, truth, train_mask, runs, limit):
    step = int(runs[3]["steps"][index])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=100, layout="constrained")
    for ax, alpha in zip(axes, (1, 3)):
        run = runs[alpha]
        points = run["embedding"][index]
        ax.scatter(points[train_mask, 0], points[train_mask, 1], c=truth[train_mask],
                   cmap="viridis", vmin=truth.min(), vmax=truth.max(), s=27,
                   marker="o", alpha=.8, edgecolors="none")
        scatter = ax.scatter(points[~train_mask, 0], points[~train_mask, 1],
                             c=truth[~train_mask], cmap="viridis",
                             vmin=truth.min(), vmax=truth.max(), s=31,
                             marker="^", alpha=.8, edgecolors="none")
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Final pooled-embedding PC1 / RMS",
               ylabel="Final pooled-embedding PC2 / RMS",
               title=f"α={alpha} · final PC1+2 {run['embedding_pca_fraction']:.1%} variance\n"
                     f"raw PCA radius {run['embedding_radius'][index]:.3g} · test MSE {current_mse(run, step):.3g}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
    fig.colorbar(scatter, ax=axes, shrink=.72, label="True polarizability")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=GRAY, label="train"),
               Line2D([0], [0], marker="^", linestyle="", color=GRAY, label="test")]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle(f"QM9 pooled graph embeddings · N={total_size} · step {step:,}\n"
                 "Each model uses its final PCA directions; coordinates are centered and divided by that frame's RMS radius",
                 fontsize=12)
    return fig


def draw_weights(index, total_size, runs, limits):
    step = int(runs[3]["steps"][index])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), dpi=100, layout="constrained")
    for ax, alpha in zip(axes, (1, 3)):
        run = runs[alpha]
        scores = run["weight_scores"]
        ax.plot(scores[:index + 1, 0], scores[:index + 1, 1],
                color=BLUE if alpha == 1 else PURPLE, lw=1.7)
        ax.scatter(scores[0, 0], scores[0, 1], marker="x", color="#111827", s=55,
                   label="initial")
        ax.scatter(scores[index, 0], scores[index, 1], color=ORANGE, s=55,
                   zorder=3, label="current")
        ax.set(xlim=limits[alpha][0], ylim=limits[alpha][1],
               xlabel="Weight trajectory PC1", ylabel="Weight trajectory PC2",
               title=f"α={alpha} · PC1+2 {run['weight_pca_fraction']:.1%} variance\n"
                     f"weight norm {run['weight_norm'][index]:.2f} · test MSE {current_mse(run, step):.3g}")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"QM9 full-model weight trajectories · N={total_size} · step {step:,}\n"
                 "Each panel uses a fixed PCA plane fitted across its saved checkpoints",
                 fontsize=13)
    return fig


def make_one(total_size, fps, device):
    dataset = QM9(root=str(DATA_ROOT))[:total_size]
    truth = np.array([float(dataset[i].y[0, TARGET]) for i in range(total_size)])
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    runs = {alpha: load_run(total_size, alpha, loader, device) for alpha in (1, 3)}
    train_mask = validate_pair(total_size, runs, truth)
    steps = runs[3]["steps"]
    limit = max(3.2, float(max(np.percentile(np.abs(run["embedding"]), 99.8)
                           for run in runs.values()) * 1.1))
    limits = {}
    for alpha, run in runs.items():
        scores = run["weight_scores"]
        padding = np.maximum(np.ptp(scores, axis=0) * .1, 1e-3)
        limits[alpha] = ((float(scores[:, 0].min() - padding[0]),
                          float(scores[:, 0].max() + padding[0])),
                         (float(scores[:, 1].min() - padding[1]),
                          float(scores[:, 1].max() + padding[1])))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"analysis_total{total_size}.npz", steps=steps, truth=truth,
                        train_mask=train_mask, alpha1_prediction=runs[1]["prediction"],
                        alpha3_prediction=runs[3]["prediction"],
                        alpha1_embedding=runs[1]["embedding"],
                        alpha3_embedding=runs[3]["embedding"],
                        alpha1_weight_pca=runs[1]["weight_scores"],
                        alpha3_weight_pca=runs[3]["weight_scores"])
    count = len(steps)
    save_gif(OUT / f"predictions_total{total_size}.gif", count, fps,
             lambda i: draw_predictions(i, total_size, truth, train_mask, runs))
    save_gif(OUT / f"embeddings_total{total_size}.gif", count, fps,
             lambda i: draw_embeddings(i, total_size, truth, train_mask, runs, limit))
    save_gif(OUT / f"weights_total{total_size}.gif", count, fps,
             lambda i: draw_weights(i, total_size, runs, limits))
    return {"total_size": total_size, "frame_count": count, "first_step": int(steps[0]),
            "last_step": int(steps[-1]), "frame_interval": 1000,
            "alpha1_embedding_pca_fraction": runs[1]["embedding_pca_fraction"],
            "alpha3_embedding_pca_fraction": runs[3]["embedding_pca_fraction"],
            "alpha1_embedding_radius_initial_final": [float(runs[1]["embedding_radius"][i])
                                                      for i in (0, -1)],
            "alpha3_embedding_radius_initial_final": [float(runs[3]["embedding_radius"][i])
                                                      for i in (0, -1)],
            "alpha1_weight_pca_fraction": runs[1]["weight_pca_fraction"],
            "alpha3_weight_pca_fraction": runs[3]["weight_pca_fraction"]}


def render_linear_predictions(total_size, fps):
    with np.load(OUT / f"analysis_total{total_size}.npz") as cached:
        steps = cached["steps"].copy()
        truth = cached["truth"].copy()
        train_mask = cached["train_mask"].copy()
        predictions = {alpha: cached[f"alpha{alpha}_prediction"].copy()
                       for alpha in (1, 3)}
    runs = {}
    for alpha in (1, 3):
        directory = ROOT / "runs" / f"grokking_total{total_size}_alpha{alpha}_animation"
        runs[alpha] = {"steps": steps, "prediction": predictions[alpha],
                       "metrics": metrics_table(directory / "full_metrics.csv")}
        for index in (0, len(steps) - 1):
            step = steps[index]
            metric_index = np.flatnonzero(runs[alpha]["metrics"]["step"] == step)
            if len(metric_index) != 1:
                raise ValueError(f"Missing metric for alpha={alpha}, step={step}")
            for subset, key in ((train_mask, "train_mse"), (~train_mask, "test_mse")):
                measured = np.mean((predictions[alpha][index, subset] - truth[subset]) ** 2)
                logged = runs[alpha]["metrics"][key][metric_index[0]]
                if not np.isclose(measured, logged, rtol=1e-4, atol=1e-3):
                    raise ValueError(f"Cached prediction and {key} disagree at step {step}")
    save_gif(OUT / f"predictions_total{total_size}_linear_x.gif", len(steps), fps,
             lambda i: draw_predictions(i, total_size, truth, train_mask, runs,
                                        xscale="linear"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 400])
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--linear-x-only", action="store_true",
                        help="Render linear-step prediction GIFs from cached predictions")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    if args.linear_x_only:
        for size in args.sizes:
            render_linear_predictions(size, args.fps)
        return
    summary_path = OUT / "summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    summaries = {int(row["total_size"]): row for row in existing}
    for size in args.sizes:
        summaries[size] = make_one(size, args.fps, torch.device(args.device))
    summaries = [summaries[size] for size in sorted(summaries)]
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
