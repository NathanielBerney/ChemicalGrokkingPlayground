#!/usr/bin/env python3
"""Ablate leading output Fourier frequencies in a mod-113 checkpoint.

This tests whether a sparse Fourier subspace of the model's class logits
carries held-out performance. The paper also performs finer 2D input-logit
and neuron ablations; this is a compact reproduction of its central check.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_modular import P, OriginalStyleTransformer, dataset


def score(logits: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy": float(np.mean(logits.argmax(axis=1) == labels)),
        "cross_entropy": float(np.mean(np.logaddexp.reduce(logits.astype(np.float64), axis=1)
                                       - logits[np.arange(len(labels)), labels])),
    }


def run(checkpoint_path: Path, seed: int, output: Path, gpu: int = 0, k: int = 5) -> dict:
    device = torch.device(f"cuda:{gpu}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = OriginalStyleTransformer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    (_, _), (test_x, test_y) = dataset(seed, device)
    with torch.inference_mode():
        logits = torch.cat([model(batch).cpu() for batch in test_x.split(2048)]).numpy()
    labels = test_y.cpu().numpy()
    W_L = (model.W_U[:, :P].T @ model.W_out).detach().cpu().numpy()
    power = (np.abs(np.fft.rfft(W_L, axis=0)) ** 2).sum(axis=1)
    key = np.sort(np.argsort(power[1:])[-k:] + 1)
    spectrum = np.fft.rfft(logits[:, :P], axis=1)
    restricted_spec = np.zeros_like(spectrum)
    restricted_spec[:, 0] = spectrum[:, 0]
    restricted_spec[:, key] = spectrum[:, key]
    excluded_spec = spectrum.copy()
    excluded_spec[:, key] = 0
    restricted = np.column_stack((np.fft.irfft(restricted_spec, n=P, axis=1), logits[:, P]))
    excluded = np.column_stack((np.fft.irfft(excluded_spec, n=P, axis=1), logits[:, P]))
    report = {
        "checkpoint_step": int(checkpoint["step"]),
        "key_output_frequencies": key.tolist(),
        "neuron_logit_frequency_power": power.tolist(),
        "original": score(logits, labels),
        "restricted_to_key_frequencies": score(restricted, labels),
        "excluding_key_frequencies": score(excluded, labels),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("run_seed0/last.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("run_seed0/fourier_ablation.json"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--key-frequencies", type=int, default=5)
    args = parser.parse_args()
    run(args.checkpoint, args.seed, args.output, args.gpu, args.key_frequencies)


if __name__ == "__main__":
    main()
