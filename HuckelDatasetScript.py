"""
Generate a Hückel-matrix dataset: 25% circulant (Hückel rings), 25% Möbius rings,
25% Toeplitz (linear chains, tridiagonal Toeplitz), 25% heteroatom-containing systems.

Conventions
-----------
alpha = 0, beta = -1  ->  energies in units of |beta|, H = -W, ascending order.
W has h_X on the diagonal and k_CX (times +/-1 for Möbius sign flips) off-diagonal.
Labels are always computed on the UNPADDED matrix. Padding is only a storage format.
"""
import numpy as np
import csv

# ---------------------------------------------------------------- parameters
# type code: (h, pi electrons contributed).  Code 0 is reserved for PAD.
ATOM_TYPES = {
    "C":  (1, 0.0, 1),
    "N1": (2, 0.5, 1),   # pyridine-type N
    "N2": (3, 1.5, 2),   # pyrrole-type N
    "O1": (4, 1.0, 1),   # carbonyl-type O
    "O2": (5, 2.0, 2),   # furan/ether-type O
}
# symmetric carbon-X coupling constants (Streitwieser-style; verify against your source)
K_CX = {"C": 1.0, "N1": 1.0, "N2": 0.8, "O1": 1.0, "O2": 0.8}

def k_pair(a, b):
    """Resonance-integral ratio for bond a-b. Hetero-hetero falls back to geometric mean."""
    if a == "C":
        return K_CX[b]
    if b == "C":
        return K_CX[a]
    return float(np.sqrt(K_CX[a] * K_CX[b]))

N_MIN, N_MAX = 3, 16
TOL = 1e-8

# ---------------------------------------------------------------- core builders
def build_W(atoms, bonds, flipped=()):
    n = len(atoms)
    W = np.zeros((n, n))
    for i, a in enumerate(atoms):
        W[i, i] = ATOM_TYPES[a][1]
    flip = {tuple(sorted(b)) for b in flipped}
    for i, j in bonds:
        s = -1.0 if tuple(sorted((i, j))) in flip else 1.0
        W[i, j] = W[j, i] = s * k_pair(atoms[i], atoms[j])
    return W

def ring_bonds(n):
    return [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

def chain_bonds(n):
    return [(i, i + 1) for i in range(n - 1)]

def analyze(W, atoms, charge):
    """Return (n_electrons, energies, homo_idx, gap, degenerate) or None if not closed-shell."""
    n = len(atoms)
    n_e = sum(ATOM_TYPES[a][2] for a in atoms) - charge
    if n_e % 2 or n_e < 2 or n_e > 2 * n - 2:
        return None
    E = np.linalg.eigvalsh(-W)           # alpha=0, beta=-1
    homo = n_e // 2 - 1
    gap = E[homo + 1] - E[homo]
    return n_e, E, homo, gap, gap < TOL

def pick_charge(atoms, rng, allow_degenerate=True):
    """Sample a charge in [-2, 2] giving a closed-shell electron count (even, 2..2n-2)."""
    n = len(atoms)
    base = sum(ATOM_TYPES[a][2] for a in atoms)
    options = [q for q in range(-2, 3)
               if (base - q) % 2 == 0 and 2 <= base - q <= 2 * n - 2]
    return int(rng.choice(options)) if options else None

# ---------------------------------------------------------------- family samplers
def sample_circulant(rng):
    n = int(rng.integers(N_MIN, N_MAX + 1))
    atoms = ["C"] * n
    return dict(family="circulant", atoms=atoms, W=build_W(atoms, ring_bonds(n)),
                twists=0, charge=pick_charge(atoms, rng))

def sample_mobius(rng):
    n = int(rng.integers(N_MIN, N_MAX + 1))
    atoms = ["C"] * n
    # one sign flip across the closing bond = one Möbius twist
    W = build_W(atoms, ring_bonds(n), flipped=[(n - 1, 0)])
    return dict(family="mobius", atoms=atoms, W=W, twists=1,
                charge=pick_charge(atoms, rng))

def sample_toeplitz(rng):
    n = int(rng.integers(2, N_MAX + 1))
    atoms = ["C"] * n
    return dict(family="toeplitz", atoms=atoms, W=build_W(atoms, chain_bonds(n)),
                twists=0, charge=pick_charge(atoms, rng))

def sample_hetero(rng):
    n = int(rng.integers(N_MIN, N_MAX + 1))
    is_ring = bool(rng.integers(0, 2))
    atoms = ["C"] * n
    n_het = int(rng.integers(1, max(2, n // 3) + 1))
    het_types = [t for t in ATOM_TYPES if t != "C"]
    for pos in rng.choice(n, size=min(n_het, n), replace=False):
        atoms[int(pos)] = str(rng.choice(het_types))
    bonds = ring_bonds(n) if is_ring else chain_bonds(n)
    # optional Möbius closure on hetero rings (kept rare, still labeled)
    twists = int(is_ring and rng.random() < 0.2)
    flipped = [(n - 1, 0)] if twists else []
    return dict(family="hetero", atoms=atoms, W=build_W(atoms, bonds, flipped),
                twists=twists, charge=pick_charge(atoms, rng))

SAMPLERS = {"circulant": sample_circulant, "mobius": sample_mobius,
            "toeplitz": sample_toeplitz, "hetero": sample_hetero}

# ---------------------------------------------------------------- analytic checks
def check_analytic(entry):
    n = len(entry["atoms"])
    E = np.sort(np.linalg.eigvalsh(-entry["W"]))
    k = np.arange(n)
    if entry["family"] == "circulant":
        ref = -2 * np.cos(2 * np.pi * k / n)
    elif entry["family"] == "mobius":
        ref = -2 * np.cos((2 * k + 1) * np.pi / n)
    elif entry["family"] == "toeplitz":
        ref = -2 * np.cos((k + 1) * np.pi / (n + 1))
    else:
        return
    assert np.allclose(E, np.sort(ref), atol=1e-9), entry["family"]

# ---------------------------------------------------------------- dataset assembly
def generate(n_total=500, seed=0, n_max=N_MAX):
    rng = np.random.default_rng(seed)
    per = n_total // 4
    plan = [f for f in SAMPLERS for _ in range(per)]
    plan += list(rng.choice(list(SAMPLERS), size=n_total - len(plan)))  # remainder
    rng.shuffle(plan)

    recs = []
    for fam in plan:
        while True:  # resample until a valid closed-shell example is produced
            e = SAMPLERS[fam](rng)
            if e["charge"] is None:
                continue
            res = analyze(e["W"], e["atoms"], e["charge"])
            if res is not None:
                break
        check_analytic(e)
        n_e, E, homo, gap, degenerate = res
        e.update(n_electrons=n_e, energies=E, homo=homo, gap=gap, degenerate=degenerate)
        recs.append(e)

    N = len(recs)
    W = np.zeros((N, n_max, n_max), dtype=np.float32)
    mask = np.zeros((N, n_max), dtype=bool)
    atom_codes = np.zeros((N, n_max), dtype=np.int8)       # 0 = PAD
    energies = np.full((N, n_max), np.nan, dtype=np.float32)
    meta = []
    for idx, e in enumerate(recs):
        n = len(e["atoms"])
        W[idx, :n, :n] = e["W"]
        mask[idx, :n] = True
        atom_codes[idx, :n] = [ATOM_TYPES[a][0] for a in e["atoms"]]
        energies[idx, :n] = e["energies"]
        meta.append(dict(id=idx, family=e["family"], n_atoms=n, charge=e["charge"],
                         n_electrons=e["n_electrons"], twists=e["twists"],
                         homo_index=e["homo"], gap=float(e["gap"]),
                         homo_energy=float(e["energies"][e["homo"]]),
                         lumo_energy=float(e["energies"][e["homo"] + 1]),
                         degenerate=bool(e["degenerate"]),
                         atoms="".join(f"[{a}]" for a in e["atoms"])))
    return dict(W=W, mask=mask, atom_codes=atom_codes, energies=energies), meta

def save(arrays, meta, stem="huckel_dataset"):
    np.savez_compressed(f"{stem}.npz", **arrays,
                        n_atoms=np.array([m["n_atoms"] for m in meta]),
                        n_electrons=np.array([m["n_electrons"] for m in meta]),
                        charge=np.array([m["charge"] for m in meta]),
                        gap=np.array([m["gap"] for m in meta], dtype=np.float32),
                        degenerate=np.array([m["degenerate"] for m in meta]),
                        family=np.array([m["family"] for m in meta]))
    with open(f"{stem}_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader(); w.writerows(meta)

def make_pairs(n_total=500, seed=0, convention="W"):
    """Return (pairs, meta).

    pairs : list of (matrix, (homo, lumo))
            matrix : 2-D np.ndarray (n x n), unpadded, natural size
            homo, lumo : orbital energies in units of |beta| (alpha = 0)
    convention : "W" -> h on diagonal, +k off-diagonal (matches your figure);
                 "H" -> H = -W, so homo/lumo are literally eigenvalues of the matrix.
    meta : list of dicts (family, charge, n_electrons, degenerate, atoms, ...).
           n_electrons is NOT recoverable from the matrix, so keep it if you need it.
    """
    arrays, meta = generate(n_total=n_total, seed=seed)
    pairs = []
    for i, m in enumerate(meta):
        n = m["n_atoms"]
        Wi = arrays["W"][i, :n, :n].astype(np.float64)
        mat = Wi if convention == "W" else -Wi
        pairs.append((mat, (m["homo_energy"], m["lumo_energy"])))
    return pairs, meta

if __name__ == "__main__":
    import pickle
    pairs, meta = make_pairs(n_total=500, seed=0)
    with open("huckel_pairs.pkl", "wb") as f:
        pickle.dump({"pairs": pairs, "meta": meta}, f)
    with open("huckel_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader(); w.writerows(meta)
    from collections import Counter
    print("samples:", len(pairs))
    print("families:", dict(Counter(m["family"] for m in meta)))
    print("degenerate (gap=0):", sum(m["degenerate"] for m in meta))
    X, y = pairs[1]
    print("example input shape:", X.shape, "| ndim:", X.ndim, "| output:", y)