#!/usr/bin/env python3
"""Fourier ablation probe for a trained n=42 circulant matrix transformer.

The paper ablates Fourier components of modular-addition logits. HOMO/LUMO
regression has only two output coordinates, so this adaptation instead ablates
Fourier components of the model's first row-projection weights. It is an
exploratory analogue, not the paper's exact progress measure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from grok_circulant_n42 import MatrixTransformer, evaluate, load_data


def ablated_weight(weight: np.ndarray, key: np.ndarray, mode: str) -> np.ndarray:
    spectrum = np.fft.rfft(weight, axis=1)
    if mode == "restricted":
        filtered = np.zeros_like(spectrum)
        filtered[:, 0] = spectrum[:, 0]
        filtered[:, key] = spectrum[:, key]
    elif mode == "excluded":
        filtered = spectrum.copy()
        filtered[:, key] = 0
    else:
        raise ValueError(mode)
    return np.fft.irfft(filtered, n=weight.shape[1], axis=1).astype(np.float32)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    device = torch.device(f"cuda:{args.gpu}")
    final = torch.load(args.final, map_location="cpu", weights_only=False)
    seed = int(args.seed)
    x_np, y_np, train_np, test_np, mean_np, std_np = load_data(args.dataset, seed, args.train_fraction)
    if not np.array_equal(final["train_indices"], train_np):
        raise ValueError("Final checkpoint split differs from the requested split")
    weight = final["model"]["row_embed.weight"].numpy()
    spectrum = np.fft.rfft(weight, axis=1)
    power = (np.abs(spectrum) ** 2).sum(axis=0)
    key = np.argsort(power[1:])[-args.key_frequencies:] + 1
    key = np.sort(key)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    test = torch.from_numpy(test_np).to(device)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    report = {"key_frequencies": key.tolist(),
              "fourier_power_by_frequency": power.tolist(),
              "ablation_target": "row_embed.weight Fourier modes; exploratory adaptation"}

    for name, path in (("early", args.early), ("final", args.final)):
        if path is None:
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        block = {"step": int(checkpoint["step"])}
        for mode in ("original", "restricted", "excluded"):
            model = MatrixTransformer().to(device)
            model.load_state_dict(checkpoint["model"])
            if mode != "original":
                original = model.row_embed.weight.detach().cpu().numpy()
                replacement = ablated_weight(original, key, mode)
                with torch.no_grad():
                    model.row_embed.weight.copy_(torch.from_numpy(replacement).to(device))
            rmse, mae = evaluate(model, x, y, test, mean, std)
            block[mode] = {"test_rmse": rmse.tolist(), "test_mae": mae.tolist()}
        report[name] = block
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/circulant_n42.npz"))
    parser.add_argument("--early", type=Path, default=Path("circulant_grokking_runs/seed0_tmux/checkpoint_early.pt"))
    parser.add_argument("--final", type=Path, default=Path("circulant_grokking_runs/seed0_tmux/last.pt"))
    parser.add_argument("--output", type=Path, default=Path("circulant_grokking_runs/seed0_tmux/fourier_probe.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--key-frequencies", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
