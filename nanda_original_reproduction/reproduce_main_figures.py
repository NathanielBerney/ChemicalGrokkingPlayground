#!/usr/bin/env python3
"""Recreate main-paper Figures 1–7 from the local mod-113 reproduction.

Figures use this project's trained weights, so learned frequencies and
numerical values differ from the authors' published run. The Fourier
projections for Figures 6–7 follow the paper's cos(k(a+b))/sin(k(a+b))
definitions and use the same input-independent bias correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from train_modular import D, HEADS, P, OriginalStyleTransformer, dataset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "main_paper_figures"
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def basis() -> tuple[np.ndarray, list[str]]:
    x = np.arange(P)
    rows = [np.ones(P) / math.sqrt(P)]
    labels = ["constant"]
    for k in range(1, P // 2 + 1):
        phase = 2 * np.pi * k * x / P
        rows.extend([np.sqrt(2 / P) * np.cos(phase), np.sqrt(2 / P) * np.sin(phase)])
        labels.extend([f"cos {k}", f"sin {k}"])
    return np.stack(rows), labels


def all_tokens(device: torch.device) -> torch.Tensor:
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    x = np.column_stack((a.ravel(), b.ravel(), np.full(P * P, P)))
    return torch.as_tensor(x, dtype=torch.long, device=device)


@torch.inference_mode()
def forward_cache(model: OriginalStyleTransformer, tokens: torch.Tensor):
    """Expose the same attention and MLP tensors as the reference hooks."""
    x = model.W_E[:, tokens].permute(1, 2, 0) + model.W_pos
    q = torch.einsum("hmd,btd->bhtm", model.W_Q, x)
    k = torch.einsum("hmd,btd->bhtm", model.W_K, x)
    v = torch.einsum("hmd,btd->bhtm", model.W_V, x)
    scores = torch.einsum("bhtm,bhsm->bhts", q, k) / math.sqrt(D // HEADS)
    scores = scores.masked_fill(model.causal_mask, -1e9)
    attention = scores.softmax(dim=-1)
    z = torch.einsum("bhts,bhsm->bhtm", attention, v)
    z = z.permute(0, 2, 1, 3).reshape(tokens.shape[0], 3, D)
    x = x + torch.einsum("dm,btm->btd", model.W_O, z)
    hidden = F.relu(torch.einsum("md,btd->btm", model.W_in, x) + model.b_in)
    x = x + torch.einsum("dm,btm->btd", model.W_out, hidden) + model.b_out
    logits = x[:, -1] @ model.W_U
    return logits, attention[:, :, -1, 0], hidden[:, -1]


@torch.inference_mode()
def capture_all(model, tokens, batch_size=2048):
    parts = [[], [], []]
    for batch in tokens.split(batch_size):
        for out, value in zip(parts, forward_cache(model, batch)):
            out.append(value.cpu().numpy())
    return tuple(np.concatenate(part, axis=0) for part in parts)


@torch.inference_mode()
def logits_all(model, tokens, batch_size=4096):
    return np.concatenate([model(batch).cpu().numpy()[:, :P] for batch in tokens.split(batch_size)])


def weights(model):
    W_E = model.W_E[:, :P].detach().cpu().numpy()
    W_L = (model.W_U[:, :P].T @ model.W_out).detach().cpu().numpy()
    return W_E, W_L


def keys_from_logit_map(W_L: np.ndarray, relative_power_cutoff=0.01):
    power = np.sum(np.abs(np.fft.rfft(W_L, axis=0)) ** 2, axis=1)
    # The paper's pictured seed has five large modes. Our seed 0 has four;
    # the next mode is over 600 times smaller. Use the spectral gap instead
    # of forcing an arbitrary fifth frequency into mechanistic plots.
    return [int(k) for k in range(1, len(power))
            if power[k] >= relative_power_cutoff * power[1:].max()]


def ce(logits: np.ndarray, labels: np.ndarray, mask=None) -> float:
    if mask is not None:
        logits, labels = logits[mask], labels[mask]
    values = logits.astype(np.float64)
    peak = values.max(axis=1)
    return float(np.mean(peak + np.log(np.exp(values - peak[:, None]).sum(axis=1))
                         - values[np.arange(len(labels)), labels]))


def accuracy(logits, labels, mask=None):
    if mask is not None:
        logits, labels = logits[mask], labels[mask]
    return float(np.mean(logits.argmax(axis=1) == labels))


def train_mask(seed: int, device):
    (train_x, _), _ = dataset(seed, device)
    index = train_x[:, 0].cpu().numpy() * P + train_x[:, 1].cpu().numpy()
    mask = np.zeros(P * P, dtype=bool)
    mask[index] = True
    return mask


def projected_sum_components(logits: np.ndarray, keys: list[int]):
    """Exact 2D projection onto cos/sin(k(a+b)) for every output class.

    Each residue s=a+b mod P occurs P times, so averaging logits by s
    followed by a 1D DFT gives the (k,k) component of the 2D input DFT.
    """
    cube = logits.reshape(P, P, P)
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    s = (a + b) % P
    grouped = np.stack([cube[s == residue].mean(axis=0) for residue in range(P)])
    spectrum = np.fft.rfft(grouped, axis=0)
    mean = np.broadcast_to(spectrum[0].real[None, :] / P, logits.shape)
    components = {}
    for k in keys:
        only = np.zeros_like(spectrum)
        only[k] = spectrum[k]
        components[k] = np.fft.irfft(only, n=P, axis=0)[s.ravel()]
    return mean, components


def paper_bias_correct(projected: np.ndarray, original: np.ndarray):
    return projected + (original - projected).mean(axis=0, keepdims=True)


def figure0_fourier_basis(B, keys):
    k = keys[0]
    x = np.arange(P)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), layout="constrained")
    axes[0].imshow(B, aspect="auto", cmap="coolwarm", interpolation="nearest",
                   vmin=-np.sqrt(2 / P), vmax=np.sqrt(2 / P))
    axes[0].set(xlabel="Residue 0–112", ylabel="Fourier basis component",
                title="Orthonormal Fourier basis for mod 113")
    axes[1].plot(x, B[2 * k - 1], label=f"cos {k}")
    axes[1].plot(x, B[2 * k], label=f"sin {k}")
    axes[1].set(xlabel="Residue", ylabel="Basis value", title=f"Learned frequency {k}")
    axes[1].legend()
    save(fig, "figure00_fourier_basis.png")


def figure1_algorithm(logits, keys):
    a, b = 17, 29
    k = keys[0]
    phase = [2 * np.pi * k * val / P for val in (a, b, (a + b) % P)]
    fig = plt.figure(figsize=(17, 4.4), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5])
    titles = [f"Input a={a}", f"Input b={b}", f"Composed a+b={a+b}"]
    for i, (theta, title) in enumerate(zip(phase, titles)):
        ax = fig.add_subplot(gs[0, i])
        circle = plt.Circle((0, 0), 1, fill=False, color="0.7")
        ax.add_patch(circle)
        ax.arrow(0, 0, np.cos(theta), np.sin(theta), width=.025,
                 length_includes_head=True, color=COLORS[i])
        ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), title=title)
        ax.text(0, -1.27, ["z(a)", "z(b)", "z(a) × z(b)"][i], ha="center")
        ax.set_aspect("equal")
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 3])
    values = logits[a * P + b, :P]
    ax.plot(np.arange(P), values, lw=1.1, color=COLORS[2], label="model logits")
    ax.axvline((a + b) % P, color="black", ls="--", label="a+b mod 113")
    ax.set(xlabel="Candidate output c", ylabel="Logit",
           title="Compose rotations, then read out c")
    ax.legend()
    fig.suptitle(f"Figure 1 · Fourier rotation at k={k}: z(a) × z(b) = z(a+b)")
    save(fig, "figure01_fourier_algorithm.png")


def figure2_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    for i in range(5):
        path = ROOT / f"run_seed{i}" / "metrics.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        step = np.array([int(r["step"]) for r in rows])
        color = COLORS[i]
        axes[0].plot(step, [float(r["test_accuracy"]) for r in rows], color=color, label=f"seed {i}")
        axes[0].plot(step, [float(r["train_accuracy"]) for r in rows], color=color, ls="--", alpha=.5)
        axes[1].plot(step, [float(r["test_loss"]) for r in rows], color=color)
        axes[1].plot(step, [float(r["train_loss"]) for r in rows], color=color, ls="--", alpha=.5)
    axes[0].set(xlabel="Optimizer step", ylabel="Accuracy", title="Accuracy; solid test, dashed train")
    axes[0].set_ylim(0, 1.05)
    axes[1].set(xlabel="Optimizer step", ylabel="Cross-entropy", title="Loss; solid test, dashed train")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Figure 2 · Delayed generalization across five seeds")
    save(fig, "figure02_learning_curves.png")


def figure3_weights(B, W_E, W_L, keys):
    ef = W_E @ B.T
    lf = B @ W_L
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), layout="constrained")
    for ax, values, title in [(axes[0], np.linalg.norm(ef, axis=0), "Embedding W_E"),
                              (axes[1], np.linalg.norm(lf, axis=1), "Neuron-logit map W_L")]:
        frequency = np.arange(1, P // 2 + 1)
        ax.scatter(frequency - .15, values[1::2], s=12, label="cos", alpha=.8)
        ax.scatter(frequency + .15, values[2::2], s=12, label="sin", alpha=.8)
        for k in keys:
            ax.axvspan(k - .45, k + .45, color="tomato", alpha=.12)
        ax.set(xlabel="Frequency k", ylabel="Component norm", title=title, xlim=(0, 57))
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Figure 3 · Learned weights in the Fourier basis")
    save(fig, "figure03_weight_fourier_norms.png")
    return ef, lf


def figure4_periodicity(logits, attention, hidden):
    cube = logits[:, :P].reshape(P, P, P)
    freq_norm = np.linalg.norm(np.fft.fftshift(np.fft.fft2(cube, axes=(0, 1), norm="ortho"),
                                              axes=(0, 1)), axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    images = [attention[:, 0].reshape(P, P), hidden[:, 0].reshape(P, P),
              np.log10(np.maximum(freq_norm, 1e-5))]
    titles = ["Attention weight, head 0: = → a", "MLP neuron 0 activation",
              "log₁₀ norm of 2D logit DFT"]
    for i, (ax, data, title) in enumerate(zip(axes, images, titles)):
        im = ax.imshow(data, origin="lower", cmap="viridis", aspect="equal")
        ax.set(xlabel="a" if i < 2 else "Fourier frequency in a",
               ylabel="b" if i < 2 else "Fourier frequency in b", title=title)
        if i == 2:
            ticks = [0, 28, 56, 84, 112]
            ax.set_xticks(ticks, [-56, -28, 0, 28, 56])
            ax.set_yticks(ticks, [-56, -28, 0, 28, 56])
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle("Figure 4 · Periodicity in attention, neurons, and logits")
    save(fig, "figure04_periodicity_and_logit_fft.png")
    return freq_norm


def neuron_frequency_fits(B, hidden, keys):
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    variance = np.sum(centered ** 2, axis=0)
    scores = np.empty((len(keys), hidden.shape[1]))
    term_names = ["cos a", "sin a", "cos b", "sin b", "cos a cos b", "cos a sin b",
                  "sin a cos b", "sin a sin b"]
    terms_by_freq = {}
    for j, k in enumerate(keys):
        constant, cos, sin = B[0], B[2 * k - 1], B[2 * k]
        terms = [np.outer(cos, constant), np.outer(sin, constant),
                 np.outer(constant, cos), np.outer(constant, sin),
                 np.outer(cos, cos), np.outer(cos, sin),
                 np.outer(sin, cos), np.outer(sin, sin)]
        X = np.stack([v.ravel() for v in terms], axis=1)
        coeff = X.T @ centered
        scores[j] = np.sum(coeff ** 2, axis=0) / np.maximum(variance, 1e-20)
        terms_by_freq[k] = (X, coeff)
    best = np.argmax(scores, axis=0)
    return np.array(keys)[best], scores[best, np.arange(hidden.shape[1])], terms_by_freq, term_names


def figure5_neuron_map(B, W_L, hidden, keys):
    best_freq, explained, terms, term_names = neuron_frequency_fits(B, hidden, keys)
    target = 14 if 14 in keys else keys[0]
    selected = np.flatnonzero((best_freq == target) & (explained > .85))
    if len(selected) == 0:
        selected = np.flatnonzero(best_freq == target)
    mapped = B @ W_L[:, selected]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), layout="constrained")
    axes[0].hist(explained, bins=np.linspace(0, 1, 21), color=COLORS[0], edgecolor="white")
    axes[0].axvline(.85, color="black", ls="--", label=f">85%: {int(np.sum(explained > .85))}/512")
    axes[0].set(xlabel="Best single-frequency variance explained", ylabel="Neurons",
                title="Degree-2 terms of one key frequency")
    axes[0].legend()
    limit = np.max(np.abs(mapped))
    im = axes[1].imshow(mapped, aspect="auto", origin="lower", cmap="coolwarm",
                        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[1].set(xlabel=f"Neurons assigned to k={target} ({len(selected)})",
                ylabel="Fourier output component", title="W_L in Fourier basis")
    axes[1].set_yticks([0, 2 * target - .5, P - 1],
                       ["constant", f"cos/sin {target}", f"sin {P//2}"])
    fig.colorbar(im, ax=axes[1], shrink=.75)
    fig.suptitle("Figure 5 · Neuron frequency specialization and logit map")
    save(fig, "figure05_neuron_frequency_logit_map.png")
    return best_freq, explained, terms, term_names


def table1_trig_readout(B, W_L, hidden, keys, terms, term_names):
    labels = np.arange(P * P).reshape(P, P)
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    rows = []
    for k in keys:
        X, _ = terms[k]
        phase = 2 * np.pi * k * (a + b).ravel() / P
        for direction, idx, target in [("cos", 2 * k - 1, np.cos(phase)),
                                       ("sin", 2 * k, np.sin(phase))]:
            projection = hidden @ (B[idx] @ W_L)
            centered = projection - projection.mean()
            coeff = X.T @ centered
            top = np.argsort(np.abs(coeff))[-2:][::-1]
            target_coeff = float(centered @ target / (target @ target))
            residual = centered - target_coeff * target
            fve = 1 - float(residual @ residual / (centered @ centered))
            rows.append({"frequency": k, "direction": direction,
                         "top_term_1": term_names[top[0]], "coefficient_1": float(coeff[top[0]]),
                         "top_term_2": term_names[top[1]], "coefficient_2": float(coeff[top[1]]),
                         "sum_wave_coefficient": target_coeff, "fraction_variance_explained": fve})
    with (OUT / "table01_trig_readout.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def figure6_frequency_ablation(logits, labels, mask, keys):
    raw = logits[:, :P]
    mean, all_components = projected_sum_components(raw, list(range(1, P // 2 + 1)))
    original_loss = ce(raw, labels, ~mask)
    restricted = paper_bias_correct(sum((all_components[k] for k in keys),
                                       np.zeros_like(raw)), raw)
    restricted_loss = ce(restricted, labels, ~mask)
    ablated = [ce(raw - all_components[k], labels, ~mask) for k in range(1, P // 2 + 1)]
    fig, ax = plt.subplots(figsize=(12, 4.6), layout="constrained")
    x = np.arange(1, P // 2 + 1)
    ax.scatter(x, ablated, c=["tomato" if k in keys else COLORS[0] for k in x], s=28)
    ax.axhline(original_loss, color="black", ls="--", label=f"original {original_loss:.2g}")
    ax.axhline(restricted_loss, color="green", ls=":", label=f"restricted {restricted_loss:.2g}")
    ax.set(xlabel="Ablated frequency k", ylabel="Held-out cross-entropy",
           title="Figure 6 · Remove one cos/sin(k(a+b)) component")
    ax.set_yscale("log")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, "figure06_frequency_ablation.png")
    return {"original_test_loss": original_loss, "restricted_test_loss": restricted_loss,
            "key_frequencies": keys, "test_loss_excluding_each_frequency": ablated,
            "restricted_test_accuracy": accuracy(restricted, labels, ~mask)}


def figure7_progress(device, tokens, labels, mask, keys, snapshots: Path):
    files = sorted(snapshots.glob("step_*.pt"))
    if not files or int(files[-1].stem.split("_")[-1]) != 40_000:
        return None
    records = []
    model = OriginalStyleTransformer().to(device).eval()
    for path in files:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        raw = logits_all(model, tokens)
        _, comps = projected_sum_components(raw, keys)
        key_part = sum(comps.values(), np.zeros_like(raw))
        restricted = paper_bias_correct(key_part, raw)
        excluded = raw - key_part
        W_E, W_L = weights(model)
        embedding_power = np.sum(np.abs(np.fft.rfft(W_E, axis=1)) ** 2, axis=0)[1:]
        logit_power = np.sum(np.abs(np.fft.rfft(W_L, axis=0)) ** 2, axis=1)[1:]
        def gini(v):
            v = np.sort(v)
            n = len(v)
            return float(((2 * np.arange(1, n + 1) - n - 1) @ v) / (n * v.sum()))
        records.append({"step": int(checkpoint["step"]),
                        "train_loss": ce(raw, labels, mask),
                        "test_loss": ce(raw, labels, ~mask),
                        "restricted_loss": ce(restricted, labels, ~mask),
                        "restricted_accuracy": accuracy(restricted, labels, ~mask),
                        "excluded_train_loss": ce(excluded, labels, mask),
                        "embedding_fourier_gini": gini(embedding_power),
                        "neuron_logit_fourier_gini": gini(logit_power),
                        "sum_squared_weights": float(sum((p.detach() ** 2).sum().item()
                                                         for p in model.parameters()))})
    with (OUT / "figure07_progress.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    x = [r["step"] for r in records]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    def plot(ax, names, ylog=False, title=""):
        for name, label in names:
            ax.plot(x, [r[name] for r in records], label=label)
        if ylog:
            ax.set_yscale("log")
        ax.set(xlabel="Optimizer step", title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    plot(axes[0, 0], [("excluded_train_loss", "excluded train"),
                      ("train_loss", "train"), ("test_loss", "test")], True,
         "Key frequencies removed")
    plot(axes[0, 1], [("restricted_loss", "restricted test"),
                      ("test_loss", "test"), ("train_loss", "train")], True,
         "Only key frequencies retained")
    plot(axes[1, 0], [("embedding_fourier_gini", "embedding"),
                      ("neuron_logit_fourier_gini", "neuron-logit")], False,
         "Fourier sparsity")
    plot(axes[1, 1], [("sum_squared_weights", "all parameters")], False,
         "Sum of squared weights")
    fig.suptitle("Figure 7 · Progress measures through grokking")
    save(fig, "figure07_progress_measures.png")
    return records


def make_pdf_bundle():
    files = sorted(OUT.glob("figure0[0-7]_*.png"))
    images = [Image.open(path).convert("RGB") for path in files]
    images[0].save(OUT / "main_figures.pdf", save_all=True,
                   append_images=images[1:], resolution=150)
    for image in images:
        image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "run_seed0" / "last.pt")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "figure_run_seed0" / "figure_snapshots")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_float32_matmul_precision("high")
    model = OriginalStyleTransformer().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    tokens = all_tokens(device)
    labels = (tokens[:, 0] + tokens[:, 1]).remainder(P).cpu().numpy()
    mask = train_mask(0, device)
    logits, attention, hidden = capture_all(model, tokens)
    B, basis_labels = basis()
    W_E, W_L = weights(model)
    keys = keys_from_logit_map(W_L)
    figure0_fourier_basis(B, keys)
    figure1_algorithm(logits, keys)
    figure2_learning_curves()
    figure3_weights(B, W_E, W_L, keys)
    figure4_periodicity(logits, attention, hidden)
    best_freq, explained, terms, term_names = figure5_neuron_map(B, W_L, hidden, keys)
    readout = table1_trig_readout(B, W_L, hidden, keys, terms, term_names)
    ablation = figure6_frequency_ablation(logits, labels, mask, keys)
    progress = figure7_progress(device, tokens, labels, mask, keys, args.snapshots)
    make_pdf_bundle()
    report = {"checkpoint_step": int(checkpoint["step"]), "key_frequencies": keys,
              "neurons_over_85pct_single_frequency": int(np.sum(explained > .85)),
              "neuron_frequency_counts": {str(k): int(np.sum(best_freq == k)) for k in keys},
              "trig_readout_fve": {f"{r['frequency']}_{r['direction']}": r["fraction_variance_explained"]
                                   for r in readout}, "ablation": ablation,
              "figure07_complete": progress is not None,
              "figure07_snapshots": len(progress) if progress else 0}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "ablation"}, indent=2))


if __name__ == "__main__":
    main()
