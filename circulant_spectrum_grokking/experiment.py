"""Shared, Fourier-free model and reproducible matrix/spectrum data handling."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / 'dataset/circulant_n42.npz'


def load_data(path=DATASET, seed=0, train_fraction=.3):
    with np.load(path, allow_pickle=False) as data:
        x = data['matrices'].astype(np.float32)
        sorted_y = data['eigenvalues'].astype(np.float64)
    n = x.shape[-1]
    if x.ndim != 3 or x.shape[1] != n:
        raise ValueError('Expected square matrices')
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be between zero and one')
    np.testing.assert_allclose(x, x.transpose(0, 2, 1), atol=1e-7)
    for i in range(n):
        np.testing.assert_allclose(x[:, i], np.roll(x[:, 0], i, axis=1), atol=1e-7)
    # FFT is used ONLY to construct mode-ordered labels and audit the dataset.
    fourier = np.fft.fft(x[:, 0].astype(np.float64), axis=1)
    np.testing.assert_allclose(fourier.imag, 0, atol=1e-10)
    mode_y = fourier.real
    np.testing.assert_allclose(np.sort(mode_y, axis=1), sorted_y, atol=1e-10)
    # Group numerically equal spectra, including isomorphic offset patterns.
    _, groups = np.unique(np.round(sorted_y, 7), axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    group_order = rng.permutation(groups.max() + 1)
    counts = np.bincount(groups)
    cumulative = np.cumsum(counts[group_order])
    cut = np.argmin(abs(cumulative[:-1] - train_fraction * len(x))) + 1
    mask = np.isin(groups, group_order[:cut])
    train, test = np.flatnonzero(mask), np.flatnonzero(~mask)
    # Same single affine scale for BOTH tasks; no per-mode weighting.
    mean, std = float(sorted_y[train].mean()), float(sorted_y[train].std())
    return dict(x=x, sorted=sorted_y, mode=mode_y, train=train, test=test,
                groups=groups, mean=mean, std=max(std, 1e-8),
                fft_max_error=float(abs(np.sort(mode_y, axis=1)-sorted_y).max()),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


class Block(nn.Module):
    def __init__(self, width, heads, mlp):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, mlp), nn.ReLU(), nn.Linear(mlp, width))

    def forward(self, h):
        a, _ = self.attn(h, h, h, need_weights=False)
        h = h + a
        return h + self.ff(h)


class MatrixTransformer(nn.Module):
    def __init__(self, n=42, width=128, heads=4, mlp=512, layers=2):
        super().__init__()
        self.row_embed = nn.Linear(n, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.randn(1, n+1, width)*.02)
        self.blocks = nn.ModuleList([Block(width, heads, mlp) for _ in range(layers)])
        self.out = nn.Linear(width, n)

    def forward(self, matrix, return_hidden=False, intervention=None):
        h = self.row_embed(matrix)
        h = torch.cat((self.cls.expand(len(matrix), -1, -1), h), 1) + self.pos
        states = []
        for stage in range(len(self.blocks)+1):
            if intervention is not None and stage == intervention[0]:
                # Remove a rank-r site subspace; retain CLS. Diagnostic only.
                q = intervention[1]
                rows = h[:, 1:]
                rows = rows - torch.einsum('nr,brd->bnd', q,
                                            torch.einsum('nr,bnd->brd', q, rows))
                h = torch.cat((h[:, :1], rows), 1)
            if return_hidden:
                states.append(h)
            if stage < len(self.blocks):
                h = self.blocks[stage](h)
        pred = self.out(h[:, 0])
        return (pred, states) if return_hidden else pred


def build_model(config):
    return MatrixTransformer(**{k: config[k] for k in ('n','width','heads','mlp','layers')})


def gini(values, axis=-1):
    v = np.sort(np.maximum(np.asarray(values, dtype=np.float64), 0), axis=axis)
    n = v.shape[axis]
    shape = [1]*v.ndim
    shape[axis] = n
    numerator = (v*(2*np.arange(1,n+1)-n-1).reshape(shape)).sum(axis=axis)
    return np.divide(numerator, n*v.sum(axis=axis), out=np.zeros_like(numerator),
                     where=v.sum(axis=axis)>1e-25)


def real_mode_basis(n, k):
    t = 2*np.pi*k*np.arange(n)/n
    if k == 0:
        return np.ones((n,1))/np.sqrt(n)
    if n % 2 == 0 and k == n//2:
        return np.cos(t)[:,None]/np.sqrt(n)
    return np.column_stack((np.cos(t),np.sin(t)))*np.sqrt(2/n)


def rfft_power(values, axis):
    n = values.shape[axis]
    power = abs(np.fft.rfft(values, axis=axis, norm='ortho'))**2
    weights = np.full(n//2+1, 2.)
    weights[0] = 1
    if n % 2 == 0:
        weights[-1] = 1
    shape = [1]*power.ndim
    shape[axis] = len(weights)
    return power*weights.reshape(shape)


def mode_ablation_selectivity(output_change):
    """Fraction of output-change energy in the expected k, n-k pair, per stage."""
    power=np.asarray(output_change)**2
    n=power.shape[-1]
    mask=np.zeros(power.shape[-2:])
    for k in range(1,n//2+1):
        mask[k-1,k]=mask[k-1,(-k)%n]=1
    fractions=(power*mask).sum(-1)/np.maximum(power.sum(-1),1e-25)
    return fractions.mean(-1)
