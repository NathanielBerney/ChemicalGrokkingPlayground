#!/usr/bin/env python3
"""Prepare matched Hückel-matrix, molecular-graph, and SMILES views."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "heteroatom_huckel_dataset"
OUT = HERE / "features.npz"
ELEMENTS = ("C", "N", "O", "S", "F", "Cl")
SITE_CLASSES = ("C", "N_pyridine", "N_pyrrole", "N_nitro", "O_carbonyl", "O_ether", "O_anion", "S_thio")
TOKEN_RE = re.compile(r"\[[^\]]+\]|Cl|Br|%\d{2}|[A-Za-z]|\d|[=#@+\-()/\\.]")


def parse_sdf(path):
    with path.open() as f:
        block = []
        for line in f:
            if line.strip() == "$$$$":
                counts = block[3]
                na, nb = int(counts[:3]), int(counts[3:6])
                atom_lines = block[4:4 + na]
                bond_lines = block[4 + na:4 + na + nb]
                atoms = [line[31:34].strip() for line in atom_lines]
                bonds = [(int(line[:3]) - 1, int(line[3:6]) - 1, int(line[6:9])) for line in bond_lines]
                charges = np.zeros(na, dtype=np.float32)
                for line in block[4 + na + nb:]:
                    if line.startswith("M  CHG"):
                        count = int(line[6:9])
                        for j in range(count):
                            pos = 9 + 8 * j
                            charges[int(line[pos:pos + 4]) - 1] = int(line[pos + 4:pos + 8])
                yield atoms, bonds, charges
                block = []
            else:
                block.append(line)


def main():
    with np.load(SOURCE / "dataset.npz", allow_pickle=False) as d:
        H = np.nan_to_num(d["matrices"].astype(np.float32), nan=0.0)
        evals = np.nan_to_num(d["eigenvalues"].astype(np.float32), nan=0.0)
        counts = d["site_counts"].astype(np.int64)
        atom_counts = d["atom_counts"].astype(np.int64)
        pi_e = d["n_pi_electrons"].astype(np.int64)
        site_e = d["site_pi_electrons"].astype(np.float32)
        site_atoms = d["site_atom_indices"].astype(np.int64)
        classes = d["site_classes"]
        gap = d["gap"].astype(np.float32)
        smiles = d["smiles"]
    assert len(set(smiles.tolist())) == len(smiles)
    assert np.allclose(gap, np.load(SOURCE / "dataset.npz")["lumo"] - np.load(SOURCE / "dataset.npz")["homo"], atol=1e-5)
    n, max_atoms = len(smiles), int(atom_counts.max())
    node = np.zeros((n, max_atoms, 6 + 8 + 4), dtype=np.float32)
    bond = np.zeros((n, max_atoms, max_atoms), dtype=np.uint8)
    pi_coupling = np.zeros((n, max_atoms, max_atoms), dtype=np.float32)
    sites = np.zeros((n, H.shape[1]), dtype=bool)
    molecule_rows = list(parse_sdf(SOURCE / "molecules.sdf"))
    assert len(molecule_rows) == n
    for i, (atoms, bonds, charges) in enumerate(molecule_rows):
        assert len(atoms) == atom_counts[i]
        for a, element in enumerate(atoms):
            node[i, a, ELEMENTS.index(element)] = 1
            node[i, a, 14] = charges[a] / 2
        for a, b, order in bonds:
            bond[i, a, b] = bond[i, b, a] = order
        node[i, :len(atoms), 15] = (bond[i, :len(atoms)] > 0).sum(axis=1) / 4
        for j in range(counts[i]):
            a = site_atoms[i, j]
            assert 0 <= a < len(atoms)
            node[i, a, 6 + SITE_CLASSES.index(classes[i, j])] = 1
            node[i, a, 16] = site_e[i, j] / 2
            node[i, a, 17] = H[i, j, j] / 2
            sites[i, j] = True
            for k in range(counts[i]):
                b = site_atoms[i, k]
                if a != b:
                    pi_coupling[i, a, b] = H[i, j, k]
        assert np.allclose(pi_coupling[i], pi_coupling[i].T)
    # Lexical SMILES tokens preserve brackets, bond symbols, and ring closures.
    tokens = [TOKEN_RE.findall(str(s)) for s in smiles]
    assert all("".join(t) == s for t, s in zip(tokens, smiles))
    vocab = {t: j + 2 for j, t in enumerate(sorted(set(t for seq in tokens for t in seq)))}
    smiles_ids = np.zeros((n, max(map(len, tokens)) + 1), dtype=np.int64)
    smiles_ids[:, 0] = 1  # CLS; 0 = PAD.
    for i, seq in enumerate(tokens):
        smiles_ids[i, 1:len(seq) + 1] = [vocab[t] for t in seq]
    # All duplicate Hückel spectra and occupancies remain in one split.
    groups = defaultdict(list)
    for i in range(n):
        key = (int(counts[i]), int(pi_e[i]), tuple(np.round(evals[i, :counts[i]], 5)))
        groups[key].append(i)
    rng = np.random.default_rng(20261029)
    keys = list(groups)
    rng.shuffle(keys)
    train, test = [], []
    for key in keys:
        (train if len(train) < 7000 else test).extend(groups[key])
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    assert set(train).isdisjoint(test) and len(train) + len(test) == n
    np.savez_compressed(OUT, H=H, eigenvalues=evals, site_mask=sites,
                        site_pi_electrons=site_e, site_counts=counts,
                        atom_counts=atom_counts, pi_electrons=pi_e, gap=gap,
                        node=node, bond=bond, pi_coupling=pi_coupling,
                        smiles_ids=smiles_ids, train=train, test=test)
    (HERE / "vocab.json").write_text(json.dumps({"PAD": 0, "CLS": 1, **vocab}, indent=2))
    (HERE / "split.json").write_text(json.dumps({
        "seed": 20261029, "train": len(train), "test": len(test),
        "spectral_groups": len(groups), "source": str(SOURCE / "dataset.npz"),
        "policy": "group by rounded eigenvalue spectrum plus pi electron count",
    }, indent=2))
    print(f"Wrote {OUT}: {len(train)} train, {len(test)} test, {len(groups)} spectral groups; {len(vocab)} SMILES tokens")


if __name__ == "__main__":
    main()
