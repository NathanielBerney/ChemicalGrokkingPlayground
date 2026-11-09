
#!/usr/bin/env python3
"""
Inspect circulant Hückel matrices and attempt to convert their graphs
into carbon-only molecular structures.

Input:
    circulant_n42_10000.npz

Outputs:
    rand/
        matrix_00000.png-independent.txt  # optional matrix text
        molecule_00000.sdf                # accepted structures
        molecule_00000.smi                # SMILES
        conversion_report.csv             # all conversion outcomes

Important:
A matrix edge is interpreted as a C-C bond. A perfect matching is used
to select candidate double bonds. RDKit sanitization checks basic chemical
valence, but does not prove chemical stability or conjugation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def print_matrix(A, title="Matrix"):
    """Print a complete matrix without NumPy truncating its entries."""
    print(f"\n{title} ({A.shape[0]} x {A.shape[1]}):")
    print(np.array2string(
        A.astype(int),
        max_line_width=200,
        threshold=A.size,
        formatter={"int": lambda x: f"{x:2d}"},
    ))


def convert_to_molecule(A, index, out_dir):
    """
    Attempt a carbon-only molecular interpretation of adjacency matrix A.

    Returns:
        status, message, smiles
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdchem
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "Install RDKit and NetworkX: pip install rdkit networkx"
        ) from exc

    A = np.asarray(A)
    n = A.shape[0]

    # Basic graph checks
    if A.shape != (n, n):
        return "rejected", "Matrix is not square", ""

    if not np.array_equal(A, A.T):
        return "rejected", "Matrix is not symmetric", ""

    if np.any(np.diag(A) != 0):
        return "rejected", "Self-connections are not allowed", ""

    if not np.all(np.isin(A, [0, 1])):
        return "rejected", "Expected a binary adjacency matrix", ""

    degrees = A.sum(axis=1).astype(int)

    # For the simple carbon model used here, each carbon has at most
    # three skeletal neighbors and one double bond.
    if np.any(degrees > 3):
        return (
            "rejected",
            f"Maximum degree is {degrees.max()}; exceeds carbon model",
            "",
        )

    if np.any(degrees == 0):
        return "rejected", "Contains isolated carbon atoms", ""

    # Find a perfect matching to assign one double bond per carbon.
    # This is a candidate Kekulé-like bond assignment, not a unique answer.
    graph = nx.from_numpy_array(A)

    # Report connected components.
    components = list(nx.connected_components(graph))
    component_count = len(components)

    print(
        f"Graph has {component_count} connected component(s) "
        f"with sizes {[len(c) for c in components]}"
    )

    # Find a maximum-cardinality matching for candidate double bonds.
    # This must run regardless of the number of components.
    matching = nx.max_weight_matching(graph, maxcardinality=True)

    if len(matching) * 2 != n:
        return (
            "rejected",
            "No perfect matching found for double-bond assignment",
            "",
        )

    double_bonds = {
        tuple(sorted(edge)) for edge in matching
    }

    # Build an RDKit molecule using all graph edges.
    rw = Chem.RWMol()
    for _ in range(n):
        atom = Chem.Atom("C")
        rw.AddAtom(atom)

    for i, j in graph.edges():
        edge = tuple(sorted((i, j)))
        bond_type = (
            rdchem.BondType.DOUBLE
            if edge in double_bonds
            else rdchem.BondType.SINGLE
        )
        rw.AddBond(int(i), int(j), bond_type)

    mol = rw.GetMol()

    try:
        # Assign implicit hydrogens and check basic valence rules.
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return "rejected", f"RDKit sanitization failed: {exc}", ""

    smiles = Chem.MolToSmiles(mol)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = out_dir / f"molecule_{index:05d}.sdf"
    smi_path = out_dir / f"molecule_{index:05d}.smi"

    writer = Chem.SDWriter(str(sdf_path))
    mol.SetProp("_Name", f"circulant_matrix_{index:05d}")
    mol.SetProp("source_matrix_index", str(index))
    mol.SetProp("selected_double_bonds", json.dumps(
        sorted([list(edge) for edge in double_bonds])
    ))
    writer.write(mol)
    writer.close()

    smi_path.write_text(
        f"{smiles}\tcirculant_matrix_{index:05d}\n",
        encoding="utf-8",
    )

    return "accepted", f"Saved {sdf_path.name}", smiles


def main():
    parser = argparse.ArgumentParser(
        description="Inspect matrices and attempt molecular conversion."
    )
    parser.add_argument(
        "--file",
        default="/home/nberney/OPSandbox/dataset/circulant_n42.npz",
        help="Input dataset NPZ file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Zero-based matrix indices to inspect",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print each complete matrix in the terminal",
    )
    parser.add_argument(
        "--make-molecules",
        action="store_true",
        help="Attempt molecular conversion and save accepted structures",
    )
    parser.add_argument(
        "--out-dir",
        default="rand",
        help="Directory for structures and conversion report",
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
    print(f"Available matrices: {n_samples:,}")
    print(f"Matrix size: {n} x {n}")

    out_dir = Path(args.out_dir)
    if args.make_molecules:
        out_dir.mkdir(parents=True, exist_ok=True)

    report = []

    for idx in args.samples:
        if not 0 <= idx < n_samples:
            print(f"\nSkipping invalid index {idx}")
            continue

        H = matrices[idx]

        # The generator uses H = alpha*I + beta*A.
        # For the default alpha=0, beta=-1, recover A = -H.
        alpha = float(data["alpha"]) if "alpha" in data.files else 0.0
        beta = float(data["beta"]) if "beta" in data.files else -1.0

        if beta == 0:
            raise ValueError("Cannot recover adjacency matrix when beta=0.")

        A = (H - alpha * np.eye(H.shape[0])) / beta
        A = np.rint(A).astype(int)

        if not np.all(np.isin(A, [0, 1])):
            raise ValueError(
                f"Matrix {idx}: recovered adjacency matrix is not binary. "
                "Check the alpha/beta parameters and dataset format."
            )
        eigs = eigenvalues[idx]
        degree = int(A[0].sum())
        edge_count = int(np.triu(A, k=1).sum())

        print("\n" + "=" * 65)
        print(f"MATRIX {idx}")
        print("=" * 65)
        print(f"Offsets: {offsets_json[idx]}")
        print(f"Degree of each vertex: {degree}")
        print(f"Edges: {edge_count}")
        print(f"HOMO: {eigs[homo_idx]:.6f}")
        print(f"LUMO: {eigs[lumo_idx]:.6f}")
        print(f"Gap:  {eigs[lumo_idx] - eigs[homo_idx]:.6f}")

        if args.print_matrix:
            print_matrix(A, f"Adjacency matrix #{idx}")

        if args.make_molecules:
            status, message, smiles = convert_to_molecule(
                A, idx, out_dir
            )
            print(f"Molecular conversion: {status}")
            print(message)
            if smiles:
                print(f"SMILES: {smiles}")

            report.append({
                "matrix_index": idx,
                "offsets": str(offsets_json[idx]),
                "degree": degree,
                "edge_count": edge_count,
                "status": status,
                "message": message,
                "smiles": smiles,
            })

    if args.make_molecules:
        report_path = out_dir / "conversion_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "matrix_index", "offsets", "degree", "edge_count",
                    "status", "message", "smiles",
                ],
            )
            writer.writeheader()
            writer.writerows(report)

        print(f"\nConversion report saved to: {report_path}")


if __name__ == "__main__":
    main()
