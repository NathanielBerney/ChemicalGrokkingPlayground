"""Small numerical and GIF helpers shared by the four-family animations."""

from __future__ import annotations

import csv

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def dct_ortho(values):
    """Orthonormal DCT-II along the first axis, using NumPy's FFT."""
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    reflected = np.concatenate((values, values[::-1]), axis=0)
    coefficients = np.fft.rfft(reflected, axis=0)[:count]
    shape = (count,) + (1,) * (values.ndim - 1)
    phase = np.exp(-1j * np.pi * np.arange(count).reshape(shape) / (2 * count))
    result = .5 * np.real(coefficients * phase)
    result[0] *= np.sqrt(1 / count)
    result[1:] *= np.sqrt(2 / count)
    return result


def gini(values):
    power = np.sort(np.asarray(values, dtype=np.float64).ravel())
    total = power.sum()
    if total < 1e-20:
        return 0.0
    count = len(power)
    return float(np.dot(2 * np.arange(1, count + 1) - count - 1, power)
                 / (count * total))


def pca_basis(rows):
    centered = np.asarray(rows, dtype=np.float64) - np.mean(rows, axis=0)
    gram = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    values = np.maximum(eigenvalues[order], 0)
    basis = eigenvectors[:, order[:2]]
    fraction = float(values[:2].sum() / max(values.sum(), 1e-30))
    return basis, fraction


def metrics_table(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows])
            for key in rows[0]}


def save_gif(path, count, fps, draw):
    frames = []
    for index in range(count):
        figure = draw(index)
        figure.canvas.draw()
        rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE,
                                                colors=160))
        plt.close(figure)
    duration = [round(1000 / fps)] * count
    duration[-1] = 1700
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, disposal=2, optimize=False)
    print(f"Saved {path}: {count} frames", flush=True)
