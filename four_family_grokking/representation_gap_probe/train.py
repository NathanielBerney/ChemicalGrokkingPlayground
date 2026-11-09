#!/usr/bin/env python3
"""Matched molecular HOMO-LUMO-gap training dynamics on one GPU per run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
KINDS = ("matrix_rows", "matrix_rows_permaug", "matrix_rows_2layer",
         "matrix_spectral", "matrix_frontier", "pi_graph",
         "molecular_graph", "smiles", "molecular_graph_raw", "smiles_raw")


class MatrixRows(nn.Module):
    """Nanda-style row attention, adapted to a masked 40-site Hamiltonian."""

    def __init__(self):
        super().__init__()
        d = 128
        self.row = nn.Linear(41, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 41, d) * .02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, d))
        self.head = nn.Sequential(nn.Linear(d + 2, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, H, site_e, site_mask, aux, return_hidden=False):
        x = torch.cat((self.cls.expand(len(H), -1, -1),
                       self.row(torch.cat((H, site_e[..., None] / 2), -1))), 1) + self.pos
        pad = torch.cat((torch.zeros((len(H), 1), dtype=torch.bool, device=H.device), ~site_mask), 1)
        update, _ = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)
        x = x + update
        x = x + self.ff(x)
        hidden = torch.cat((x[:, 0], aux), 1)
        pred = self.head(hidden).squeeze(-1)
        return (pred, hidden) if return_hidden else pred


class MatrixRowsTwoLayer(nn.Module):
    """Parameter-matched deeper row attention with two contextualization rounds."""

    def __init__(self):
        super().__init__()
        d = 96
        self.row = nn.Linear(41, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 41, d) * .02)
        self.blocks = nn.ModuleList()
        for _ in range(2):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(d),
                "attn": nn.MultiheadAttention(d, 4, batch_first=True),
                "norm2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 192), nn.GELU(), nn.Linear(192, d)),
            }))
        self.head = nn.Sequential(nn.Linear(d + 2, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, H, site_e, site_mask, aux, return_hidden=False):
        x = torch.cat((self.cls.expand(len(H), -1, -1),
                       self.row(torch.cat((H, site_e[..., None] / 2), -1))), 1) + self.pos
        pad = torch.cat((torch.zeros((len(H), 1), dtype=torch.bool, device=H.device), ~site_mask), 1)
        for block in self.blocks:
            h = block["norm1"](x)
            update, _ = block["attn"](h, h, h, key_padding_mask=pad, need_weights=False)
            x = x + update
            x = x + block["ff"](block["norm2"](x))
        hidden = torch.cat((x[:, 0], aux), 1)
        pred = self.head(hidden).squeeze(-1)
        return (pred, hidden) if return_hidden else pred


class PrecomputedEigenvalueMLP(nn.Module):
    """Already-diagonalized H eigenvalues plus occupancy markers."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(122, 128), nn.GELU(), nn.Linear(128, 128),
                                 nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, 1)

    def forward(self, eig, homo_marker, lumo_marker, aux, return_hidden=False):
        x = torch.cat((eig / 3, homo_marker, lumo_marker, aux), 1)
        hidden = self.net(x)
        pred = self.head(hidden).squeeze(-1)
        return (pred, hidden) if return_hidden else pred


class FrontierEnergiesControl(nn.Module):
    """Near-oracle control: HOMO/LUMO energies are selected before this network."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.head = nn.Linear(64, 1)

    def forward(self, eig, homo_marker, lumo_marker, aux, return_hidden=False):
        homo = (eig * homo_marker).sum(1, keepdim=True)
        lumo = (eig * lumo_marker).sum(1, keepdim=True)
        hidden = self.net(torch.cat((homo, lumo, aux), 1))
        pred = self.head(hidden).squeeze(-1)
        return (pred, hidden) if return_hidden else pred


class GraphLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_linear = nn.Linear(d, d)
        self.bond_linear = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(3))
        self.pi_linear = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, bond, coupling, mask):
        degree = (bond > 0).sum(-1).clamp_min(1)[..., None]
        update = self.self_linear(h)
        for order, linear in enumerate(self.bond_linear, 1):
            update = update + torch.bmm((bond == order).to(h.dtype), linear(h)) / degree
        update = update + torch.bmm(coupling, self.pi_linear(h)) / degree
        return self.norm(h + F.gelu(update)) * mask[..., None]


class PiGraphLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_linear = nn.Linear(d, d)
        self.edge_linear = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, coupling, mask):
        degree = coupling.ne(0).sum(-1).clamp_min(1)[..., None]
        message = torch.bmm(coupling, self.edge_linear(h)) / degree
        return self.norm(h + F.gelu(self.self_linear(h) + message)) * mask[..., None]


class PiGraph(nn.Module):
    """Permutation-invariant GNN using exactly H and occupancy, no sigma atoms."""

    def __init__(self):
        super().__init__()
        d = 96
        self.embed = nn.Linear(3, d)
        self.layers = nn.ModuleList(PiGraphLayer(d) for _ in range(6))
        self.pool = nn.Sequential(nn.Linear(2 * d + 2, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, 1)

    def forward(self, H, site_e, site_mask, aux, return_hidden=False):
        diagonal = torch.diagonal(H, dim1=1, dim2=2)
        coupling = H - torch.diag_embed(diagonal)
        degree = coupling.ne(0).sum(-1).float() / 4
        features = torch.stack((diagonal / 2, site_e / 2, degree), -1)
        h = self.embed(features) * site_mask[..., None]
        for layer in self.layers:
            h = layer(h, coupling, site_mask)
        avg = h.sum(1) / site_mask.sum(1).clamp_min(1)[..., None]
        mx = h.masked_fill(~site_mask[..., None], -1e6).max(1).values
        hidden = self.pool(torch.cat((avg, mx, aux), 1))
        pred = self.head(hidden).squeeze(-1)
        return (pred, hidden) if return_hidden else pred


class MolecularGraph(nn.Module):
    """Message passing on full SDF atom/bond graphs with π-site edge couplings."""

    def __init__(self):
        super().__init__()
        d = 96
        self.embed = nn.Linear(18, d)
        self.layers = nn.ModuleList(GraphLayer(d) for _ in range(4))
        self.pool = nn.Sequential(nn.Linear(2 * d + 2, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, 1)

    def forward(self, node, bond, coupling, atom_mask, aux, return_hidden=False):
        h = self.embed(node) * atom_mask[..., None]
        for layer in self.layers:
            h = layer(h, bond, coupling, atom_mask)
        avg = h.sum(1) / atom_mask.sum(1).clamp_min(1)[..., None]
        mx = h.masked_fill(~atom_mask[..., None], -1e6).max(1).values
        hidden = self.pool(torch.cat((avg, mx, aux), 1))
        pred = self.head(hidden).squeeze(-1)
        return (pred, hidden) if return_hidden else pred


class SmilesTransformer(nn.Module):
    """Sequence transformer over lexical canonical-SMILES tokens."""

    def __init__(self, vocab_size, max_len):
        super().__init__()
        d = 96
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * .02)
        layer = nn.TransformerEncoderLayer(d, 4, 192, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.pool = nn.Sequential(nn.Linear(d + 2, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, 1)

    def forward(self, ids, aux, return_hidden=False):
        x = self.embed(ids) + self.pos[:, :ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=ids.eq(0))
        hidden = self.pool(torch.cat((x[:, 0], aux), 1))
        pred = self.head(hidden).squeeze(-1)
        return (pred, hidden) if return_hidden else pred


def load_data(device):
    with np.load(HERE / "features.npz", allow_pickle=False) as d:
        data = {k: torch.from_numpy(d[k]).to(device) for k in d.files if k not in ("train", "test")}
        train = d["train"].copy()
        test = d["test"].copy()
    data["atom_mask"] = torch.arange(data["node"].shape[1], device=device)[None] < data["atom_counts"][:, None]
    data["aux"] = torch.stack((data["pi_electrons"].float() / 40, data["site_counts"].float() / 40), 1)
    size = data["eigenvalues"].shape[1]
    occ = data["pi_electrons"] // 2 - 1
    if not bool(torch.all((occ >= 0) & (occ + 1 < data["site_counts"]))):
        raise ValueError("Open-shell or invalid HOMO/LUMO occupancy")
    data["homo_marker"] = F.one_hot(occ, size).float()
    data["lumo_marker"] = F.one_hot(occ + 1, size).float()
    return data, train, test


def model_call(model, kind, d, ids, return_hidden=False):
    if kind in ("matrix_rows", "matrix_rows_permaug", "matrix_rows_2layer", "pi_graph"):
        return model(d["H"][ids], d["site_pi_electrons"][ids], d["site_mask"][ids], d["aux"][ids], return_hidden)
    if kind in ("matrix_spectral", "matrix_frontier"):
        return model(d["eigenvalues"][ids], d["homo_marker"][ids], d["lumo_marker"][ids], d["aux"][ids], return_hidden)
    if kind == "molecular_graph":
        return model(d["node"][ids], d["bond"][ids], d["pi_coupling"][ids], d["atom_mask"][ids], d["aux"][ids], return_hidden)
    if kind == "molecular_graph_raw":
        node = d["node"][ids].clone()
        node[..., 6:14] = 0  # no π-site class
        node[..., 16:] = 0  # no π-electron or Hückel diagonal annotation
        aux = torch.stack((d["atom_counts"][ids].float() / 40,
                           torch.zeros_like(d["atom_counts"][ids].float())), 1)
        return model(node, d["bond"][ids], torch.zeros_like(d["pi_coupling"][ids]),
                     d["atom_mask"][ids], aux, return_hidden)
    if kind == "smiles_raw":
        return model(d["smiles_ids"][ids], torch.zeros_like(d["aux"][ids]), return_hidden)
    return model(d["smiles_ids"][ids], d["aux"][ids], return_hidden)


def permuted_matrix_batch(d, ids):
    """Relabel the active sites jointly on both H axes and occupancy vector."""
    H = d["H"][ids]
    site_e = d["site_pi_electrons"][ids]
    width = H.shape[1]
    valid = torch.arange(width, device=H.device)[None] < d["site_counts"][ids, None]
    rank = torch.rand(H.shape[:2], device=H.device) + (~valid).float() * 2
    perm = torch.argsort(rank, dim=1)
    H = H.gather(1, perm[:, :, None].expand(-1, -1, width))
    H = H.gather(2, perm[:, None, :].expand(-1, width, -1))
    site_e = site_e.gather(1, perm)
    return H, site_e


def score(pred, true):
    rel = np.abs(pred - true) / true
    return {
        "accuracy_rel_1pct": float(np.mean(rel < .01)),
        "accuracy_rel_5pct": float(np.mean(rel < .05)),
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "median_relative_error": float(np.median(rel)),
        "p90_relative_error": float(np.quantile(rel, .9)),
    }


@torch.inference_mode()
def predict(model, kind, d, ids, center, scale, batch=256, hidden=False):
    model.eval()
    preds, reps = [], []
    for start in range(0, len(ids), batch):
        ix = torch.as_tensor(ids[start:start + batch], device=d["gap"].device)
        out = model_call(model, kind, d, ix, hidden)
        if hidden:
            out, h = out
            reps.append(h.float().cpu().numpy())
        preds.append(out.float().cpu().numpy())
    pred = np.exp(np.concatenate(preds).astype(np.float64) * scale + center)
    return (pred, np.concatenate(reps)) if hidden else pred


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d, train, test = load_data(device)
    train = np.sort(train)
    test = np.sort(test)
    y = d["gap"].cpu().numpy()
    center = float(np.log(y[train]).mean())
    scale = float(np.log(y[train]).std())
    targets = (torch.log(d["gap"]) - center) / scale
    models = {
        "matrix_rows": MatrixRows,
        "matrix_rows_permaug": MatrixRows,
        "matrix_rows_2layer": MatrixRowsTwoLayer,
        "matrix_spectral": PrecomputedEigenvalueMLP,
        "matrix_frontier": FrontierEnergiesControl,
        "pi_graph": PiGraph,
        "molecular_graph": MolecularGraph,
        "molecular_graph_raw": MolecularGraph,
        "smiles": lambda: SmilesTransformer(int(d["smiles_ids"].max().item()) + 1, d["smiles_ids"].shape[1]),
        "smiles_raw": lambda: SmilesTransformer(int(d["smiles_ids"].max().item()) + 1, d["smiles_ids"].shape[1]),
    }
    model = models[args.kind]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 100)
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["outdir"] = str(config["outdir"])
    config.update({"train_size": len(train), "test_size": len(test), "target": "standardized log gap",
                   "split": "spectral-group 70/30", "device": str(device), "parameters": sum(p.numel() for p in model.parameters())})
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2))
    log_path = args.outdir / "history.csv"
    fields = ["step", "seconds", "lr"] + [f"{split}_{metric}" for split in ("train", "test") for metric in score(y[:2], y[:2])]
    start = time.monotonic()
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                row = {"step": step, "seconds": round(time.monotonic() - start, 2), "lr": optimizer.param_groups[0]["lr"]}
                for name, ids in (("train", train), ("test", test)):
                    p = predict(model, args.kind, d, ids, center, scale)
                    row.update({f"{name}_{k}": v for k, v in score(p, y[ids]).items()})
                writer.writerow(row)
                f.flush()
                print(f"{args.kind} {step:5d}/{args.steps} train1%={row['train_accuracy_rel_1pct']:.3f} test1%={row['test_accuracy_rel_1pct']:.3f} trainMAE={row['train_mae']:.4f} testMAE={row['test_mae']:.4f} {row['seconds']:.0f}s", flush=True)
            if step in (0, args.steps // 4, args.steps // 2, args.steps):
                torch.save({"model": model.state_dict(), "step": step, "config": config,
                            "target_center": center, "target_scale": scale}, args.outdir / f"checkpoint_{step:05d}.pt")
            if step == args.steps:
                break
            model.train()
            ids = torch.as_tensor(rng.choice(train, size=args.batch_size, replace=False), device=device)
            optimizer.zero_grad(set_to_none=True)
            if args.kind == "matrix_rows_permaug":
                permuted_H, permuted_e = permuted_matrix_batch(d, ids)
                pred = model(permuted_H, permuted_e, d["site_mask"][ids], d["aux"][ids])
            else:
                pred = model_call(model, args.kind, d, ids)
            loss = F.mse_loss(pred, targets[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress = min((step + 1) / (args.decay_steps or args.steps), 1.0)
            optimizer.param_groups[0]["lr"] = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
    # Save per-molecule final predictions and learned representations for PCA/probes.
    ids = np.arange(len(y))
    pred, hidden = predict(model, args.kind, d, ids, center, scale, hidden=True)
    np.savez_compressed(args.outdir / "final_probe.npz", pred=pred, true=y, hidden=hidden,
                        train=train, test=test, site_counts=d["site_counts"].cpu().numpy(),
                        pi_electrons=d["pi_electrons"].cpu().numpy())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--decay-steps", type=int, default=None,
                   help="Cosine decay duration; hold at 0.1× LR afterward.")
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
