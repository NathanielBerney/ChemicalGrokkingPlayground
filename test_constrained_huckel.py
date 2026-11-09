"""Independent small-matrix checks for the compact Hückel generator."""

import networkx as nx
import numpy as np

from constrained_huckel import frontier, hamiltonian_sparse, molecular_graph, n_values


def test_frontier_matches_direct_diagonalization():
    for family, sizes in {
        "circulant": (6, 10, 14, 18),
        "mobius": (4, 8, 12, 16),
        "linear": (4, 6, 8, 10),
    }.items():
        for n in sizes:
            eig = np.linalg.eigvalsh(hamiltonian_sparse(family, n).toarray())
            homo, lumo, gap = frontier(family, n)
            assert np.allclose((homo, lumo, gap),
                               (eig[n // 2 - 1], eig[n // 2], eig[n // 2] - eig[n // 2 - 1]),
                               atol=1e-12)


def test_graphs_have_valid_carbon_valence_and_planar_connectivity():
    for family, n in (("circulant", 6), ("circulant", 38),
                      ("mobius", 16), ("linear", 4), ("linear", 24)):
        bonds, hydrogens = molecular_graph(family, n)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(bonds[:, :2])
        assert nx.is_connected(graph)
        assert nx.check_planarity(graph)[0]
        assert max(dict(graph.degree()).values()) <= 2
        assert np.all(hydrogens >= 0)
        valence = hydrogens.astype(int)
        np.add.at(valence, bonds[:, 0], bonds[:, 2])
        np.add.at(valence, bonds[:, 1], bonds[:, 2])
        assert np.all(valence == 4)
        if family == "mobius":
            H = hamiltonian_sparse(family, n)
            assert H[0, n - 1] == 1


def test_default_graph_keys_are_distinct_across_families():
    keys = set()
    for family in ("circulant", "mobius", "linear"):
        sizes = n_values(family, 10_000)
        assert len(np.unique(sizes)) == 10_000
        kind = "path" if family == "linear" else "cycle"
        family_keys = {(kind, int(n)) for n in sizes}
        assert keys.isdisjoint(family_keys)
        keys.update(family_keys)
    assert len(keys) == 30_000
