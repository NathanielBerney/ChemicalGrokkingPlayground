
#!/usr/bin/env python3
"""
Generate a dataset of symmetric binary circulant Hückel matrices.

Default:
    n = 42
    samples = 10_000
    alpha = 0
    beta = -1
    seed = 42

Outputs:
    <prefix>.csv  : metadata, eigenvalues, HOMO, LUMO, gap
    <prefix>.npz  : matrices, spectra, first rows, offset masks

Requirements:
    pip install numpy

Important:
    This is a mathematical dataset. No chemical-validity filtering
    is applied. Graph-isomorphic matrices are not deduplicated.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def canonical_offsets(offsets, n):
    """Canonicalize undirected cyclic offsets to 1,...,floor(n/2)."""
    return tuple(sorted({
        min(int(k) % n, (-int(k)) % n)
        for k in offsets
        if int(k) % n != 0
    }))


def circulant_adjacency(n, offsets):
    """Create a symmetric binary circulant adjacency matrix."""
    if n < 3:
        raise ValueError("n must be at least 3.")

    offsets = canonical_offsets(offsets, n)
    A = np.zeros((n, n), dtype=np.float64)

    for k in offsets:
        for i in range(n):
            A[i, (i + k) % n] = 1.0
            A[i, (i - k) % n] = 1.0

    np.fill_diagonal(A, 0.0)

    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)

    return A


def sample_offset_sets(n, n_samples, seed, balanced=True):
    """
    Sample unique offset sets.

    balanced=True:
        Distribute samples as evenly as possible across the number
        of selected offsets.

    balanced=False:
        Sample uniformly from all binary offset patterns.

    Uniqueness is by offset set, not graph isomorphism.
    """
    rng = np.random.default_rng(seed)
    max_offset = n // 2
    offsets = np.arange(1, max_offset + 1)

    total_patterns = 2**max_offset

    if n_samples > total_patterns:
        raise ValueError(
            f"Only {total_patterns:,} patterns exist for n={n}."
        )

    if not balanced:
        # Uniform random sampling from the binary pattern space.
        # Rejection sampling guarantees uniqueness.
        selected = set()

        while len(selected) < n_samples:
            mask = rng.integers(
                0, 2, size=max_offset, dtype=np.int8
            )
            candidate = tuple(
                (np.flatnonzero(mask) + 1).tolist()
            )
            selected.add(candidate)

        return sorted(selected, key=lambda x: (len(x), x))

    # Balance the number of examples with k selected offsets.
    capacities = [
        math.comb(max_offset, k)
        for k in range(max_offset + 1)
    ]
    quotas = [0] * (max_offset + 1)

    # Round-robin allocation respects each stratum's capacity.
    remaining = n_samples

    while remaining:
        progressed = False

        for k in range(max_offset + 1):
            if quotas[k] < capacities[k]:
                quotas[k] += 1
                remaining -= 1
                progressed = True

                if remaining == 0:
                    break

        if not progressed:
            raise ValueError("Insufficient unique patterns.")

    selected = set()

    for k, quota in enumerate(quotas):
        bucket = set()

        while len(bucket) < quota:
            if k == 0:
                candidate = ()
            else:
                candidate = tuple(sorted(
                    rng.choice(
                        offsets, size=k, replace=False
                    ).tolist()
                ))

            bucket.add(candidate)

        selected.update(bucket)

    return sorted(selected, key=lambda x: (len(x), x))


def is_connected(A):
    """Test graph connectivity using depth-first search."""
    n = A.shape[0]
    visited = {0}
    stack = [0]

    while stack:
        i = stack.pop()

        for j in np.flatnonzero(A[i]):
            j = int(j)

            if j not in visited:
                visited.add(j)
                stack.append(j)

    return len(visited) == n


def generate_dataset(
    n=42,
    n_samples=10_000,
    seed=42,
    alpha=0.0,
    beta=-1.0,
    out_prefix="circulant_n42",
    balanced=True,
):
    """Generate and save a circulant Hückel matrix dataset."""

    if n < 3:
        raise ValueError("n must be at least 3.")

    if n % 2 != 0:
        raise ValueError(
            "This version assumes even n for HOMO/LUMO filling."
        )

    if beta == 0:
        raise ValueError("beta must be nonzero.")

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    npz_path = prefix.with_suffix(".npz")

    offset_sets = sample_offset_sets(
        n=n,
        n_samples=n_samples,
        seed=seed,
        balanced=balanced,
    )

    max_offset = n // 2

    matrices = np.empty(
        (n_samples, n, n), dtype=np.float32
    )
    spectra = np.empty(
        (n_samples, n), dtype=np.float64
    )
    first_rows = np.empty(
        (n_samples, n), dtype=np.float32
    )
    offset_masks = np.zeros(
        (n_samples, max_offset), dtype=np.uint8
    )

    # Assumes n electrons, two electrons per spatial orbital.
    homo_idx = n // 2 - 1
    lumo_idx = n // 2

    fields = [
        "sample_id",
        "n",
        "topology",
        "offsets",
        "n_offsets",
        "degree",
        "edge_count",
        "alpha",
        "beta",
        "eigenvalues_ascending",
        "homo_index_zero_based",
        "lumo_index_zero_based",
        "homo_energy",
        "lumo_energy",
        "homo_lumo_gap",
        "frontier_degeneracy",
        "connected",
    ]

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

        for idx, selected_offsets in enumerate(offset_sets):
            A = circulant_adjacency(
                n, selected_offsets
            )

            # Hückel Hamiltonian:
            # H_ii = alpha
            # H_ij = beta when a coupling is present
            H = alpha * np.eye(n) + beta * A

            # Symmetric matrix eigenvalues, in ascending order.
            eig = np.linalg.eigvalsh(H)

            matrices[idx] = H.astype(np.float32)
            spectra[idx] = eig
            first_rows[idx] = H[0].astype(np.float32)

            for k in selected_offsets:
                offset_masks[idx, k - 1] = 1

            homo = float(eig[homo_idx])
            lumo = float(eig[lumo_idx])
            gap = lumo - homo

            # Number of orbitals degenerate with either frontier
            # energy, using a small numerical tolerance.
            tol = 1e-8
            frontier_degeneracy = max(
                np.count_nonzero(
                    np.abs(eig - homo) < tol
                ),
                np.count_nonzero(
                    np.abs(eig - lumo) < tol
                ),
            )

            writer.writerow({
                "sample_id": f"circ_n{n}_{idx:06d}",
                "n": n,
                "topology": "symmetric_binary_circulant",
                "offsets": json.dumps(
                    list(selected_offsets)
                ),
                "n_offsets": len(selected_offsets),
                "degree": int(np.count_nonzero(A[0])),
                "edge_count": int(np.count_nonzero(A) // 2),
                "alpha": alpha,
                "beta": beta,
                "eigenvalues_ascending": json.dumps(
                    eig.tolist()
                ),
                "homo_index_zero_based": homo_idx,
                "lumo_index_zero_based": lumo_idx,
                "homo_energy": homo,
                "lumo_energy": lumo,
                "homo_lumo_gap": gap,
                "frontier_degeneracy": int(
                    frontier_degeneracy
                ),
                "connected": is_connected(A),
            })

    np.savez_compressed(
        npz_path,
        matrices=matrices,
        eigenvalues=spectra,
        first_rows=first_rows,
        offset_masks=offset_masks,
        offsets_json=np.array([
            json.dumps(list(x)) for x in offset_sets
        ]),
        n=np.array(n),
        alpha=np.array(alpha),
        beta=np.array(beta),
        seed=np.array(seed),
        balanced_sampling=np.array(balanced),
        model=np.array(
            "H = alpha*I + beta*A; "
            "A is binary symmetric circulant"
        ),
        electron_count=np.array(n),
        homo_index_zero_based=np.array(homo_idx),
        lumo_index_zero_based=np.array(lumo_idx),
    )

    print(f"Generated {n_samples:,} unique matrices.")
    print(f"Matrix dimension: {n} x {n}")
    print(f"Alpha: {alpha}")
    print(f"Beta: {beta}")
    print(f"Balanced sampling: {balanced}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"NPZ: {npz_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--n", type=int, default=42)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument(
        "--out-prefix",
        default="circulant_n42",
    )
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        help="Sample random offset subsets without balancing their sizes.",
    )

    args = parser.parse_args()

    generate_dataset(
        n=args.n,
        n_samples=args.samples,
        seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        out_prefix=args.out_prefix,
        balanced=not args.unbalanced,
    )


if __name__ == "__main__":
    main()
