# Graph-distinct all-carbon Hückel datasets

Run from this directory:

```bash
python -m pip install numpy scipy matplotlib
python constrained_huckel.py
```

The optional checks use `pytest` and `networkx`:
`python -m pytest -q test_constrained_huckel.py`.

This writes **10,000 distinct molecular graphs per family** to
`constrained_huckel_datasets/`, in order: ordinary circulant rings, Möbius
rings, and linear conjugated chains. Each family has `dataset.csv`, a compact
`dataset.npz` of input sizes and frontier labels, `example.npz` with one small
explicit Hamiltonian, `parameters.json`, and `visualization.png` (linear axes)
and `visualization_log10.png` (log10 size and gap axes). Combined gap plots
are saved as `comparison.png` (linear axes) and `comparison_log10.png`
(log10 axes). The frontier-energy y-axis stays linear in both versions because
HOMO energies can be negative. Use `--count`, `--family`, or the
`--*-min-n` options to change the run.

## Graph and chemistry model

Each row has a different even number of carbons. The `graph_key` identifies
its complete carbon connectivity: `cycle:n` for rings and `path:n` for chains.
For each edge `(i,i+1)`, the bond order is 2 when `i` is even and 1 otherwise;
the ring closure `(n-1,0)` is single. Implicit hydrogen count is `4` minus the
sum of bond orders at each carbon. Every carbon therefore has valence 4 and no
more than two carbon neighbors. All graphs are connected, and paths/cycles
have planar graph embeddings. Planarity is a graph constraint for the ordinary
ring and linear families; it is **not imposed on the Möbius family**. Carbon
and hydrogen counts distinguish all
30,000 graphs across the three default families: the ordinary and Möbius
rings use different congruence classes of `n`.

The ordinary rings use `n = 4k+2` (starting at 6), Möbius rings use `n = 4k`
(starting at 16), and chains use even `n` (starting at 4). These give neutral
closed-shell Hückel fillings with one π electron per carbon. They are model
graphs, not a catalog of experimentally stable compounds. Graph planarity
does not establish an optimized planar molecular geometry. In particular,
**a Möbius orbital twist requires nonplanar geometry**; its 2D diagram is only
a projection of the connectivity and signed Hückel boundary.

This strict graph-distinct target forces large sizes: the 10,000th ring has
about 40,000 carbon atoms, and the 10,000th chain about 20,000. There is no
artificial `n = 42` ceiling. Those sizes are idealized extrapolations; they
are not claimed to be realistic isolated molecules. This dataset should be
used with size-aware train/test splits because gap is strongly determined by
`n`.

## Matrix and labels

`H_ii = alpha` and ordinary edge couplings are `H_ij = beta` (defaults 0 and
-1). The Möbius closing edge has `H_(n-1),0 = -beta`, representing one
anti-periodic orbital phase flip. All spectra are for neutral `n` π electrons,
two electrons per spatial orbital. `homo_index_zero_based = n/2 - 1` and
`lumo_index_zero_based = n/2`. The positive HOMO–LUMO gap is reported in the
same energy units as beta.

Dense matrices across 30,000 variable-size graphs would require terabytes.
The dataset instead stores an exact graph rule and closed-form frontier
energies. Reconstruct any Hamiltonian in O(n) storage:

```python
from constrained_huckel import hamiltonian_sparse, molecular_graph

H = hamiltonian_sparse("mobius", 16)  # SciPy CSR matrix
bonds, implicit_hydrogens = molecular_graph("mobius", 16)
```

`example.npz` gives an explicit small matrix, bonds, and hydrogens for each
family. The existing `HeteroatomsScript.py` and its output are unchanged.

## Scope of the checks

The construction guarantees connectivity, degree, Kekulé bond orders, carbon
valence, and graph planarity for these paths and cycles. It does not run
geometry optimization, certify planarity in 3D, or establish synthetic
accessibility. The Möbius phase-flip model is a Hückel idealization; real
twisting can change orbital overlaps and bond lengths.

The `4k+2` ordinary-ring and `4k` Möbius-ring electron rules follow the
standard Hückel and Möbius aromaticity models. Relevant background:
[IUPAC Hückel rule](https://old.goldbook.iupac.org/html/H/H02867.html),
[Möbius aromatic hydrocarbon synthesis](https://www.nature.com/articles/nature02224),
and [ring-current analysis of Möbius annulenes](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b00440).
