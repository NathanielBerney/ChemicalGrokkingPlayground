
#!/usr/bin/env python3
"""Inspect a few matrices from the n=42 circulant Hückel dataset."""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Print sample matrices from a circulant Hückel dataset."
    )
    parser.add_argument(
        "--file",
        default="/home/nberney/OPSandbox/dataset/circulant_n42.npz",
        help="Path to the .npz dataset",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Zero-based matrix indices to inspect",
    )
    parser.add_argument(
        "--show-full",
        action="store_true",
        help="Print all eigenvalues, not just HOMO/LUMO",
    )
    args = parser.parse_args()

    data = np.load(args.file, allow_pickle=False)

    matrices = data["matrices"]
    eigenvalues = data["eigenvalues"]
    offsets_json = data["offsets_json"]

    n_samples, n, _ = matrices.shape
    homo_idx = n // 2 - 1
    lumo_idx = n // 2

    print(f"Dataset: {args.file}")
    print(f"Number of matrices: {n_samples:,}")
    print(f"Matrix dimensions: {n} x {n}")
    print(f"Stored arrays: {list(data.files)}")

    for idx in args.samples:
        if not 0 <= idx < n_samples:
            print(f"\nSkipping index {idx}: valid indices are 0–{n_samples - 1}.")
            continue

        A = matrices[idx]
        eigs = eigenvalues[idx]
        offsets = offsets_json[idx]

        degree_sequence = A.sum(axis=1).astype(int)
        edge_count = int(np.triu(A, k=1).sum())

        print("\n" + "=" * 72)
        print(f"MATRIX #{idx}")
        print("=" * 72)
        print(f"Selected circulant offsets: {offsets}")
        print(f"Degree of each vertex: {degree_sequence[0]}")
        print(f"Number of edges: {edge_count}")
        print(f"Symmetric: {np.array_equal(A, A.T)}")
        print(f"Diagonal all zero: {np.all(np.diag(A) == 0)}")

        print("\nAdjacency / Hückel matrix:")
        print(A.astype(int))

        print("\nOrbital energy summary:")
        print(f"HOMO (index {homo_idx}): {eigs[homo_idx]: .6f}")
        print(f"LUMO (index {lumo_idx}): {eigs[lumo_idx]: .6f}")
        print(f"HOMO-LUMO gap:            "
              f"{eigs[lumo_idx] - eigs[homo_idx]: .6f}")

        if args.show_full:
            print("\nFull sorted eigenvalue spectrum:")
            print(np.array2string(eigs, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
