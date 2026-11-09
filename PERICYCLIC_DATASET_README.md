# Pericyclic thermal-symmetry dataset, version 1

Run `python pericyclic_dataset.py` from this directory. By default it writes
10,000 candidate cyclic transition-state orbital matrices from **10,000
non-isomorphic, valence-checked reactant graphs** to
`pericyclic_thermal_dataset/`, with exactly 5,000 labels of each kind. The
generator accepts `--count`, `--seed`, and `--outdir`.

## Input and output

- `dataset.npz`: `H` is a `(10000, 8, 8)` array of **dimensionless generalized
  Hückel-style active-orbital matrices**. Each sample uses its upper-left
  `n_sites × n_sites` block; remaining entries are zero padding. `site_mask`
  and `n_sites` identify active orbitals. `y` is the binary target, with
  `1 = formally thermally allowed` and `0 = formally thermally forbidden` for
  the specified candidate approach. `site_codes` identifies C, O, and a
  transferring H orbital; `site_order` maps displayed matrix indices to the
  cyclic template ordering.
- `metadata.csv`: one row per matrix in the same order, with reaction family,
  substituted reactant SMILES, parent reactant SMILES, a collision-safe graph
  ID, candidate approach, electron sources, phase parity, electron count,
  frontier energies, and explicit limitations.
- `reactant_graphs.jsonl`: one exact atom-and-bond graph per matrix in the
  same order, including implicit hydrogen counts. `reactant_graph_id` in the
  CSV is unique; the Weisfeiler–Lehman `reactant_graph_hash` may collide for
  different graphs, so the generator checks graph isomorphism before adding a
  row.
- `templates.json`: the active sites and cyclic edge roles for all eight
  reaction templates. The edge at index `i` goes from site `i` to
  `(i+1) mod n`.
- `parameters.json`: model and source information.
- `overview.png` and `matrix_examples.png`: class balance and canonical matrix
  examples.

Example:

```python
import numpy as np

with np.load("pericyclic_thermal_dataset/dataset.npz") as data:
    i = 0
    n = int(data["n_sites"][i])
    H = data["H"][i, :n, :n]
    allowed = int(data["y"][i])
```

## Reaction families and rule

| Family | Active electrons | Candidate approaches |
|---|---:|---|
| Diels–Alder [4+2] | 6 | supra/supra: 1; supra/antara: 0 |
| Alder–ene | 6 | all suprafacial: 1; one antarafacial component: 0 |
| Claisen [3,3] | 6 | supra/supra: 1; supra/antara: 0 |
| Cope [3,3] | 6 | supra/supra: 1; supra/antara: 0 |
| [2+2] cycloaddition | 4 | supra/supra: 0; supra/antara: 1 |
| [4+4] cycloaddition | 8 | supra/supra: 0; supra/antara: 1 |
| 4π electrocyclic closure | 4 | disrotatory: 0; conrotatory: 1 |
| 6π electrocyclic closure | 6 | disrotatory: 1; conrotatory: 0 |

The label rule is `1` for a **Hückel (+1 cyclic phase, even inversions)**
array with `4k+2` active electrons, or a **Möbius (−1 cyclic phase, odd
inversions)** array with `4k` active electrons. Otherwise the label is `0`.
The phase is computed from the product of signs around the cycle after
removing the conventional negative coupling sign. The generator randomizes
orbital phase gauge, cyclic rotation, and reversal, so a positive entry at a
particular matrix corner is not a dependable label shortcut. The loop sign
and label remain invariant.

There are **eight reaction-center templates**, each with two proposed
approaches. Each row decorates carbon sites with a unique combination of
small substituents, then checks graph isomorphism and atomic valence. The
10,000 reactant graphs and 10,000 matrices are distinct. Overlap magnitudes,
orbital gauge, and small substituent site shifts also vary. These numerical
parameters are illustrative, not fitted to quantum-chemical transition
states.

## Interpretation and limits

Concerted pericyclic reactions are described by a cyclic **transition state**,
not an isolable intermediate. This matrix represents the proposed *active
orbital interaction network*, including nascent or breaking interactions.
It is not a Hückel matrix of reactants alone. Ene and Claisen require a
generalized active-orbital model because their electron circuits include
σ bonding and, for ene, a transferring hydrogen; a carbon π-only model would
omit part of the pathway.

The label answers **only** whether the specified idealized approach is
thermally symmetry-allowed by the Woodward–Hoffmann/Hückel–Möbius rule.
Neither label predicts activation barrier, yield, kinetic feasibility,
geometric accessibility, or whether a different stepwise mechanism occurs.
In particular, a formally allowed antarafacial [2+2] approach may be hard to
realize geometrically. No 3D transition states were optimized. Opposite-face
approaches in the dataset are formal counterfactual candidates. The generated
reactant SMILES are written from acyclic, valence-checked graphs; RDKit
sanitization and geometry optimization were not run.

Because the matrices and labels are generated from this same rule, this is a
**synthetic rule-learning benchmark**, not an independently measured reaction
database. Validation on it measures rule recovery; it does not establish
generalization to experimentally observed chemistry.

Sources for the underlying classifications: [IUPAC definition of pericyclic
reaction](https://www.old.goldbook.iupac.org/html/P/P04491.html),
[Woodward and Hoffmann's orbital-symmetry treatment](https://onlinelibrary.wiley.com/doi/10.1002/anie.196907811),
[original electrocyclic stereochemistry paper](https://roaldhoffmann.com/sites/default/files/fromd6/18s_1.pdf),
and [Alder–ene six-electron treatment](https://pmc.ncbi.nlm.nih.gov/articles/PMC10087545/).
