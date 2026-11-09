#!/usr/bin/env python3
"""
Generate unique, chemically plausible conjugated heteroatom molecules and
their Hückel Hamiltonians.

Outputs (in --outdir):
    dataset.npz      matrices, adjacency, bond orders, eigenvalues, masks, ...
    metadata.csv     one row per molecule (SMILES, HOMO/LUMO, ...)
    molecules.sdf    2D structures with properties
    parameters.json  parameters, generator settings, statistics

How molecules are built (instead of random cycle+chord graphs):
  1. Start from a library of real aromatic rings (benzene, pyridine,
     pyrrole, furan, thiophene, azoles, azines, ...).
  2. Fuse additional 5/6-membered aromatic rings onto free C-C bonds
     (naphthalene, quinoline, indole, benzothiazole, ... patterns).
  3. Optionally link two such cores by a direct bond, vinylene,
     ethynylene or azo spacer (biaryls, stilbenes, azobenzenes, ...).
  4. Decorate with common substituents (NH2, OH, OMe, CHO, CN, NO2, ...).
  5. Reject implausible motifs (peroxides, N-N single bonds, N-O/N-S bonds
     other than nitro, N-N-N chains, ...), then sanitize with RDKit.

Hamiltonian (units of |beta_CC|, alpha_C = 0, beta_CC = -1):
  * Only atoms in the conjugated pi system are included (RDKit
    conjugation perception). sp3 atoms and sigma-only halogens are not
    part of H; the site list is stored so the matrices can be mapped back.
  * Each site gets a type keyed on (element, number of pi electrons), e.g.
    pyridine-type N (1 e) vs pyrrole/amine-type N (2 e).
  * H_ii = h_X * beta_CC          (h > 0 -> more electronegative -> LOWER
                                   energy, because beta_CC is negative)
  * H_ij = k_ij * beta_CC * s     k_ij = sqrt(k_i * k_j),
                                   s = single_bond_scale for non-aromatic
                                   single bonds between two 1-electron sites
                                   (alternating bond lengths in conjugated
                                   chains), otherwise 1.
  * Orbitals are filled (2 e each, closed shell) to give HOMO, LUMO, gap
    and total pi energy.

IMPORTANT CAVEATS
  * The (h, k) values are textbook-range, illustrative numbers. Hückel
    heteroatom parameters vary a lot between sources; replace PARAMS with
    your preferred literature set (and cite it) for quantitative work.
  * RDKit sanitization plus the blocklist is NOT a stability or synthetic
    accessibility test. Fused systems are generated in 2D topology only;
    sterically impossible overlaps (e.g. long helicenes) are not screened.
  * NPZ matrices are NaN-padded; use site_counts to recover each matrix.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------
# Hückel parameters
# --------------------------------------------------------------------------
BETA_CC = -1.0  # energy unit (negative by convention)

# (site class) -> (h, k).  alpha_X = alpha_C + h * beta_CC,
# beta_CX = k * beta_CC.  Bond between two non-carbon sites uses sqrt(k_i*k_j).
PARAMS = {
    "C":         (0.0, 1.0),  # sp2/sp carbon, 1 pi electron
    "N_pyridine": (0.5, 1.0),  # =N-, 1 pi electron
    "N_pyrrole":  (1.5, 0.8),  # -NH-, -NR-, -NH2: 2 pi electrons
    "N_nitro":    (2.0, 1.0),  # N+ of NO2, 1 pi electron
    "O_carbonyl": (1.0, 1.0),  # C=O, N=O: 1 pi electron
    "O_ether":    (2.0, 0.8),  # -O-, -OH, furan O: 2 pi electrons
    "O_anion":    (1.5, 0.8),  # O- of NO2: 2 pi electrons
    "S_thio":     (1.0, 0.7),  # thiophene S, -SH: 2 pi electrons
}
PI_ELECTRONS = {
    "C": 1, "N_pyridine": 1, "N_pyrrole": 2, "N_nitro": 1,
    "O_carbonyl": 1, "O_ether": 2, "O_anion": 2, "S_thio": 2,
}

# --------------------------------------------------------------------------
# Building-block libraries
# --------------------------------------------------------------------------
# (SMILES, sampling weight)
RING_LIBRARY = [
    ("c1ccccc1", 6.0),        # benzene
    ("c1ccncc1", 2.0),        # pyridine
    ("c1cncnc1", 1.0),        # pyrimidine
    ("c1cnccn1", 1.0),        # pyrazine
    ("c1ccnnc1", 0.5),        # pyridazine
    ("c1ncncn1", 0.3),        # 1,3,5-triazine
    ("c1cc[nH]c1", 1.5),      # pyrrole
    ("c1ccoc1", 1.5),         # furan
    ("c1ccsc1", 1.5),         # thiophene
    ("c1c[nH]cn1", 1.0),      # imidazole
    ("c1cc[nH]n1", 0.7),      # pyrazole
    ("c1cocn1", 0.7),         # oxazole
    ("c1cscn1", 0.7),         # thiazole
]

# Substituents: SMILES whose atom 0 bonds to the core. (SMILES, weight)
SUBSTITUENTS = [
    ("N", 3.0),               # amino
    ("N(C)C", 1.5),           # dimethylamino
    ("O", 3.0),               # hydroxy
    ("OC", 2.0),              # methoxy
    ("S", 0.7),               # thiol
    ("C", 3.0),               # methyl (sigma only)
    ("CC", 1.0),              # ethyl (sigma only)
    ("C(F)(F)F", 0.7),        # CF3 (sigma only)
    ("F", 1.0),               # fluoro (sigma only)
    ("Cl", 1.0),              # chloro (sigma only)
    ("C=O", 1.5),             # formyl
    ("C(=O)C", 1.0),          # acetyl
    ("C(=O)O", 1.0),          # carboxylic acid
    ("C(=O)N", 0.7),          # amide
    ("C#N", 2.0),             # cyano
    ("[N+](=O)[O-]", 1.5),    # nitro
    ("C=C", 1.0),             # vinyl
]

SPACERS = [
    ("direct", 5.0),
    ("vinylene", 2.5),
    ("ethynylene", 1.5),
    ("azo", 1.0),
]

# Substructures that are not considered chemically plausible here.
BLOCKLIST_SMARTS = {
    "peroxide/O-O": "[#8]~[#8]",
    "S-S": "[#16]~[#16]",
    "O-S": "[#8]~[#16]",
    "N-S": "[#7]~[#16]",
    "N-O (non-nitro)": "[#7;!$([N+](=O)[O-])]~[#8]",
    "N-N single (hydrazine-like)": "[#7]-!:[#7]",
    "N-N-N chain": "[#7]~[#7]~[#7]",
}
BLOCKLIST = {k: Chem.MolFromSmarts(v) for k, v in BLOCKLIST_SMARTS.items()}

HETERO_NUMS = (7, 8, 16)  # N, O, S


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--count", type=int, default=10_000)
    p.add_argument("--min-heavy", type=int, default=6,
                   help="Minimum heavy atoms per molecule.")
    p.add_argument("--max-heavy", type=int, default=50,
                   help="Maximum heavy atoms per molecule.")
    p.add_argument("--max-fused-rings", type=int, default=3,
                   help="Max extra rings fused onto each core.")
    p.add_argument("--link-prob", type=float, default=0.35,
                   help="Probability of joining two cores through a spacer.")
    p.add_argument("--max-substituents", type=int, default=4)
    p.add_argument("--min-heteroatoms", type=int, default=1,
                   help="Min N/O/S atoms (counting substituents).")
    p.add_argument("--max-heteroatoms", type=int, default=8)
    p.add_argument("--single-bond-scale", type=float, default=0.9,
                   help="beta scale for non-aromatic single bonds between "
                        "two 1-electron pi sites.")
    p.add_argument("--allow-all-carbon", action="store_true",
                   help="Allow molecules with zero heteroatoms.")
    p.add_argument("--unique-spectra", action="store_true",
                   help="Also reject molecules whose pi spectrum duplicates "
                        "an accepted one (drops sigma-only variants).")
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--outdir", type=Path,
                   default=Path("heteroatom_huckel_dataset"))
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Default: max(50000, count*200).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------
# Molecule construction
# --------------------------------------------------------------------------
def weighted_choice(rng, items):
    values, weights = zip(*items)
    return rng.choices(values, weights=weights, k=1)[0]


def sanitize_copy(rw):
    """Return a sanitized Mol or None."""
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def make_ring_atom(spec):
    atomic = {"c": 6, "n": 7, "nH": 7, "o": 8, "s": 16}[spec]
    a = Chem.Atom(atomic)
    a.SetIsAromatic(True)
    if spec == "nH":
        a.SetNumExplicitHs(1)
        a.SetNoImplicit(True)
    return a


def is_pyrrole_type(atom):
    """Aromatic atom that donates a lone pair (o, s, nH, N-substituted n)."""
    sym = atom.GetSymbol()
    if sym in ("O", "S"):
        return True
    if sym == "N":
        return atom.GetTotalNumHs() > 0 or atom.GetDegree() == 3
    return False


def fusable_bonds(mol):
    """Aromatic C-C bonds in exactly one ring whose atoms are both CH."""
    ri = mol.GetRingInfo()
    out = []
    for b in mol.GetBonds():
        if not b.GetIsAromatic() or ri.NumBondRings(b.GetIdx()) != 1:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if all(a.GetAtomicNum() == 6 and a.GetDegree() == 2
               and a.GetTotalNumHs() == 1 for a in (a1, a2)):
            host = next(r for r in ri.BondRings() if b.GetIdx() in r)
            out.append((b, len(host)))
    return out


def new_ring_path(rng, size):
    """Return atom specs for the (size-2) new atoms of a fused ring."""
    if size == 6:
        return [rng.choices(["c", "n"], weights=[0.8, 0.2])[0]
                for _ in range(4)]
    # 5-membered: donor next to the fusion, then two c/n sites
    donor = rng.choices(["nH", "o", "s"], weights=[1.0, 1.0, 1.0])[0]
    tail = [rng.choices(["c", "n"], weights=[0.75, 0.25])[0] for _ in range(2)]
    return [donor] + tail


def fuse_ring(mol, rng):
    options = fusable_bonds(mol)
    if not options:
        return None
    bond, host_size = rng.choice(options)
    new_size = rng.choice([5, 6]) if rng.random() < 0.3 else 6
    if host_size == 5 and new_size == 5:
        return None  # avoid pentalene-like 5-5 fusions

    a, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

    if host_size == 5:
        # fuse next to the host's donor atom only (indole-like, not isoindole)
        ri = mol.GetRingInfo()
        host_atoms = next(r for r in ri.AtomRings()
                          if a in r and c in r and len(r) == 5)
        def next_to_donor(idx):
            return any(is_pyrrole_type(nb) and nb.GetIdx() in host_atoms
                       for nb in mol.GetAtomWithIdx(idx).GetNeighbors())
        if next_to_donor(a) == next_to_donor(c):
            return None

    if rng.random() < 0.5:
        a, c = c, a

    specs = new_ring_path(rng, new_size)
    rw = Chem.RWMol(mol)
    new_idx = [rw.AddAtom(make_ring_atom(s)) for s in specs]
    chain = [a] + new_idx + [c]
    for u, v in zip(chain[:-1], chain[1:]):
        rw.AddBond(u, v, Chem.BondType.AROMATIC)
        rw.GetBondBetweenAtoms(u, v).SetIsAromatic(True)
    return sanitize_copy(rw)


def build_core(rng, args):
    mol = Chem.MolFromSmiles(weighted_choice(rng, RING_LIBRARY))
    n_fuse = rng.choices(range(args.max_fused_rings + 1),
                         weights=[4.0] + [2.0, 1.0, 0.5, 0.3][:args.max_fused_rings])[0]
    for _ in range(n_fuse):
        for _try in range(6):
            fused = fuse_ring(mol, rng)
            if fused is not None and fused.GetNumHeavyAtoms() <= args.max_heavy:
                mol = fused
                break
    return mol


def link_candidates(mol):
    """sp2 CH carbons and pyrrole-type nH atoms that can take a new bond."""
    out = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 6 and a.GetDegree() == 2 and a.GetTotalNumHs() == 1 \
                and (a.GetIsAromatic() or a.GetHybridization()
                     == Chem.HybridizationType.SP2):
            out.append(a.GetIdx())
        elif a.GetAtomicNum() == 7 and a.GetIsAromatic() \
                and a.GetTotalNumHs() == 1:
            out.append(a.GetIdx())
    return out


def clear_h(rw, idx):
    atom = rw.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() == 7 and atom.GetIsAromatic():
        atom.SetNumExplicitHs(0)


def link_cores(mol_a, mol_b, rng):
    ca, cb = link_candidates(mol_a), link_candidates(mol_b)
    if not ca or not cb:
        return None
    ia, ib = rng.choice(ca), rng.choice(cb)
    spacer = weighted_choice(rng, SPACERS)

    rw = Chem.RWMol(Chem.CombineMols(mol_a, mol_b))
    ib += mol_a.GetNumAtoms()
    clear_h(rw, ia)
    clear_h(rw, ib)
    S, D, T = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE

    if spacer == "direct":
        rw.AddBond(ia, ib, S)
    else:
        if spacer == "vinylene":
            x, y, order = 6, 6, D
        elif spacer == "ethynylene":
            x, y, order = 6, 6, T
        else:  # azo
            x, y, order = 7, 7, D
        i1 = rw.AddAtom(Chem.Atom(x))
        i2 = rw.AddAtom(Chem.Atom(y))
        rw.AddBond(ia, i1, S)
        rw.AddBond(i1, i2, order)
        rw.AddBond(i2, ib, S)
    return sanitize_copy(rw)


def add_substituents(mol, rng, max_sub):
    n_sub = rng.randint(0, max_sub)
    for _ in range(n_sub):
        cands = [i for i in link_candidates(mol)
                 if mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
        if not cands:
            break
        idx = rng.choice(cands)
        sub = Chem.MolFromSmiles(weighted_choice(rng, SUBSTITUENTS))
        rw = Chem.RWMol(Chem.CombineMols(mol, sub))
        rw.AddBond(idx, mol.GetNumAtoms(), Chem.BondType.SINGLE)
        new = sanitize_copy(rw)
        if new is not None:
            mol = new
    return mol


def plausible(mol):
    return not any(mol.HasSubstructMatch(p) for p in BLOCKLIST.values())


def build_candidate(rng, args):
    core = build_core(rng, args)
    if rng.random() < args.link_prob:
        other = build_core(rng, args)
        if core.GetNumHeavyAtoms() + other.GetNumHeavyAtoms() + 2 > args.max_heavy:
            return None
        core = link_cores(core, other, rng)
        if core is None:
            return None
    mol = add_substituents(core, rng, args.max_substituents)
    n_heavy = mol.GetNumHeavyAtoms()
    if not (args.min_heavy <= n_heavy <= args.max_heavy):
        return None
    if len(Chem.GetMolFrags(mol)) != 1 or not plausible(mol):
        return None
    return mol


# --------------------------------------------------------------------------
# Pi-system perception and Hamiltonian
# --------------------------------------------------------------------------
def classify_site(atom):
    """Return the PARAMS class of a pi site, or None if not modeled."""
    sym = atom.GetSymbol()
    has_multiple = any(
        b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
        for b in atom.GetBonds()
    )
    if sym == "C":
        return "C"
    if sym == "N":
        if atom.GetFormalCharge() == 1 and has_multiple:
            return "N_nitro"
        if atom.GetIsAromatic():
            connections = atom.GetDegree() + atom.GetTotalNumHs()
            return "N_pyrrole" if connections == 3 else "N_pyridine"
        return "N_pyridine" if has_multiple else "N_pyrrole"
    if sym == "O":
        if atom.GetFormalCharge() == -1:
            return "O_anion"
        if atom.GetIsAromatic():
            return "O_ether"
        return "O_carbonyl" if has_multiple else "O_ether"
    if sym == "S":
        if atom.GetIsAromatic() or not has_multiple:
            return "S_thio"
    return None  # halogens, unsupported elements -> not a pi site


def pi_system(mol):
    """Return (site_atom_indices, site_classes) for conjugated atoms."""
    sites = set()
    for b in mol.GetBonds():
        if b.GetIsConjugated():
            sites.add(b.GetBeginAtomIdx())
            sites.add(b.GetEndAtomIdx())
    classes = {}
    for idx in sorted(sites):
        cls = classify_site(mol.GetAtomWithIdx(idx))
        if cls is not None:
            classes[idx] = cls
    return sorted(classes), classes


def build_hamiltonian(mol, single_bond_scale):
    sites, classes = pi_system(mol)
    if len(sites) < 4:
        return None
    pos = {s: i for i, s in enumerate(sites)}
    n = len(sites)
    H = np.zeros((n, n))
    A = np.zeros((n, n))
    BO = np.zeros((n, n))

    for s in sites:
        H[pos[s], pos[s]] = PARAMS[classes[s]][0] * BETA_CC

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in pos or j not in pos or not b.GetIsConjugated():
            continue
        ci, cj = classes[i], classes[j]
        k = math.sqrt(PARAMS[ci][1] * PARAMS[cj][1])
        if (b.GetBondType() == Chem.BondType.SINGLE
                and PI_ELECTRONS[ci] == 1 and PI_ELECTRONS[cj] == 1):
            k *= single_bond_scale
        pi_, pj = pos[i], pos[j]
        H[pi_, pj] = H[pj, pi_] = k * BETA_CC
        A[pi_, pj] = A[pj, pi_] = 1.0
        BO[pi_, pj] = BO[pj, pi_] = b.GetBondTypeAsDouble()

    # every site must be connected to at least one other site
    if (A.sum(axis=1) == 0).any():
        return None

    n_pi = sum(PI_ELECTRONS[classes[s]] for s in sites)
    return {
        "H": H, "A": A, "BO": BO, "sites": sites,
        "classes": [classes[s] for s in sites],
        "n_pi": n_pi,
    }


def orbital_summary(eigs, n_pi):
    n_occ = n_pi // 2
    homo = float(eigs[n_occ - 1]) if n_occ >= 1 else float("nan")
    lumo = float(eigs[n_occ]) if n_occ < len(eigs) else float("nan")
    return {
        "homo": homo,
        "lumo": lumo,
        "gap": lumo - homo,
        "pi_energy": float(2.0 * eigs[:n_occ].sum()),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive.")
    if args.min_heavy < 5 or args.max_heavy < args.min_heavy:
        raise SystemExit("Require 5 <= min-heavy <= max-heavy.")
    if args.max_heteroatoms < args.min_heteroatoms:
        raise SystemExit("max-heteroatoms must be >= min-heteroatoms.")

    min_het = 0 if args.allow_all_carbon else max(1, args.min_heteroatoms)

    rng = random.Random(args.seed)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    target_attempts = (args.max_attempts if args.max_attempts is not None
                       else max(50_000, args.count * 200))

    accepted = []
    seen_smiles, seen_spectra = set(), set()
    stats = Counter()
    attempts = 0

    writer = Chem.SDWriter(str(outdir / "molecules.sdf"))
    try:
        while len(accepted) < args.count and attempts < target_attempts:
            attempts += 1
            mol = build_candidate(rng, args)
            if mol is None:
                stats["rejected_construction"] += 1
                continue

            n_het = sum(1 for a in mol.GetAtoms()
                        if a.GetAtomicNum() in HETERO_NUMS)
            if not (min_het <= n_het <= args.max_heteroatoms):
                stats["rejected_heteroatom_count"] += 1
                continue

            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if smiles in seen_smiles:
                stats["rejected_duplicate"] += 1
                continue

            ham = build_hamiltonian(mol, args.single_bond_scale)
            if ham is None:
                stats["rejected_no_pi_system"] += 1
                continue
            if ham["n_pi"] % 2 != 0:
                stats["rejected_open_shell"] += 1
                continue

            eigs = np.linalg.eigvalsh(ham["H"])
            if args.unique_spectra:
                key = tuple(np.round(eigs, 6))
                if key in seen_spectra:
                    stats["rejected_duplicate_spectrum"] += 1
                    continue
                seen_spectra.add(key)

            orb = orbital_summary(eigs, ham["n_pi"])
            idx = len(accepted)
            het_counts = dict(Counter(
                a.GetSymbol() for a in mol.GetAtoms()
                if a.GetAtomicNum() in HETERO_NUMS))

            out_mol = Chem.Mol(mol)
            rdDepictor.Compute2DCoords(out_mol)
            out_mol.SetProp("_Name", f"hetero_huckel_{idx:05d}")
            out_mol.SetProp("canonical_smiles", smiles)
            out_mol.SetProp("pi_sites", str(len(ham["sites"])))
            out_mol.SetProp("pi_electrons", str(ham["n_pi"]))
            out_mol.SetProp("homo", f"{orb['homo']:.6f}")
            out_mol.SetProp("lumo", f"{orb['lumo']:.6f}")
            out_mol.SetProp("gap", f"{orb['gap']:.6f}")
            out_mol.SetProp("pi_site_atom_indices",
                            json.dumps(ham["sites"]))
            out_mol.SetProp("heteroatom_counts_json",
                            json.dumps(het_counts, sort_keys=True))
            out_mol.SetProp("validation",
                            "RDKit sanitization + blocklist; stability not assessed")
            writer.write(out_mol)

            seen_smiles.add(smiles)
            accepted.append({
                "matrix_id": idx,
                "smiles": smiles,
                "n_heavy": mol.GetNumHeavyAtoms(),
                "n_rings": mol.GetRingInfo().NumRings(),
                "het_counts": het_counts,
                "eigs": eigs,
                **ham, **orb,
            })

            if not args.quiet and len(accepted) % 250 == 0:
                print(f"Accepted {len(accepted):,}/{args.count:,}; "
                      f"attempts={attempts:,}")
    finally:
        writer.close()

    if not accepted:
        raise SystemExit("No valid structures generated.")
    if len(accepted) != args.count:
        print(f"WARNING: generated {len(accepted):,} of {args.count:,} "
              f"requested molecules after {attempts:,} attempts.",
              file=sys.stderr)

    # ---- NPZ (NaN/-1/empty padded) ---------------------------------------
    count = len(accepted)
    ms = max(len(x["sites"]) for x in accepted)
    matrices = np.full((count, ms, ms), np.nan, dtype=np.float32)
    adjacency = np.full_like(matrices, np.nan)
    bond_orders = np.full_like(matrices, np.nan)
    eigenvalues = np.full((count, ms), np.nan, dtype=np.float32)
    site_atom_indices = np.full((count, ms), -1, dtype=np.int16)
    site_classes = np.full((count, ms), "", dtype="<U12")
    site_pi_electrons = np.zeros((count, ms), dtype=np.int8)
    site_counts = np.zeros(count, dtype=np.int16)
    atom_counts = np.zeros(count, dtype=np.int16)
    n_pi_arr = np.zeros(count, dtype=np.int16)
    homo = np.zeros(count, dtype=np.float32)
    lumo = np.zeros(count, dtype=np.float32)
    gap = np.zeros(count, dtype=np.float32)
    pi_energy = np.zeros(count, dtype=np.float32)
    smiles_arr = np.empty(
        count, dtype=f"<U{max(len(x['smiles']) for x in accepted)}")

    for i, it in enumerate(accepted):
        n = len(it["sites"])
        matrices[i, :n, :n] = it["H"]
        adjacency[i, :n, :n] = it["A"]
        bond_orders[i, :n, :n] = it["BO"]
        eigenvalues[i, :n] = it["eigs"]
        site_atom_indices[i, :n] = it["sites"]
        site_classes[i, :n] = it["classes"]
        site_pi_electrons[i, :n] = [PI_ELECTRONS[c] for c in it["classes"]]
        site_counts[i] = n
        atom_counts[i] = it["n_heavy"]
        n_pi_arr[i] = it["n_pi"]
        homo[i], lumo[i] = it["homo"], it["lumo"]
        gap[i], pi_energy[i] = it["gap"], it["pi_energy"]
        smiles_arr[i] = it["smiles"]

    np.savez_compressed(
        outdir / "dataset.npz",
        matrices=matrices, adjacency=adjacency, bond_orders=bond_orders,
        eigenvalues=eigenvalues,
        site_atom_indices=site_atom_indices, site_classes=site_classes,
        site_pi_electrons=site_pi_electrons,
        site_counts=site_counts, atom_counts=atom_counts,
        n_pi_electrons=n_pi_arr,
        homo=homo, lumo=lumo, gap=gap, pi_energy=pi_energy,
        smiles=smiles_arr,
        params_json=np.array(json.dumps(PARAMS)),
        seed=np.array(args.seed),
    )

    # ---- CSV --------------------------------------------------------------
    fields = ["matrix_id", "smiles", "heavy_atoms", "pi_sites", "pi_electrons",
              "rings", "heteroatom_counts", "homo", "lumo", "gap", "pi_energy"]
    with (outdir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in accepted:
            w.writerow({
                "matrix_id": it["matrix_id"], "smiles": it["smiles"],
                "heavy_atoms": it["n_heavy"], "pi_sites": len(it["sites"]),
                "pi_electrons": it["n_pi"], "rings": it["n_rings"],
                "heteroatom_counts": json.dumps(it["het_counts"], sort_keys=True),
                "homo": f"{it['homo']:.6f}", "lumo": f"{it['lumo']:.6f}",
                "gap": f"{it['gap']:.6f}", "pi_energy": f"{it['pi_energy']:.6f}",
            })

    # ---- JSON -------------------------------------------------------------
    parameters = {
        "requested_count": args.count,
        "generated_count": count,
        "attempts": attempts,
        "rejection_stats": dict(stats),
        "seed": args.seed,
        "settings": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
        "beta_cc": BETA_CC,
        "site_parameters_h_k": PARAMS,
        "pi_electrons_per_site_class": PI_ELECTRONS,
        "off_diagonal_rule": "beta_ij = sqrt(k_i*k_j)*beta_cc; "
                             "x single_bond_scale for non-aromatic single "
                             "bonds between two 1-electron sites",
        "diagonal_rule": "alpha_i = h_i * beta_cc (alpha_C = 0)",
        "ring_library": [s for s, _ in RING_LIBRARY],
        "substituents": [s for s, _ in SUBSTITUENTS],
        "spacers": [s for s, _ in SPACERS],
        "blocklist": BLOCKLIST_SMARTS,
        "energy_units": "|beta_CC| (arbitrary Hückel units)",
        "caveats": (
            "Parameters are illustrative textbook-range values, not a single "
            "cited parameterization. RDKit sanitization and the blocklist do "
            "not assess strain, stability, synthetic accessibility, or 3D "
            "geometry. Halogens and sp3 atoms are not part of the pi system."
        ),
    }
    (outdir / "parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8")

    print("\nDataset generation finished.")
    print(f"Unique accepted molecules: {count:,}")
    print(f"Attempts: {attempts:,}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v:,}")
    print(f"Heavy atoms: {min(x['n_heavy'] for x in accepted)}–"
          f"{max(x['n_heavy'] for x in accepted)}")
    print(f"Pi sites:    {min(len(x['sites']) for x in accepted)}–{ms}")
    print(f"Output dir:  {outdir}")


if __name__ == "__main__":
    main()