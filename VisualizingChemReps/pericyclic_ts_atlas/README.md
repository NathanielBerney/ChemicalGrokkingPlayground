# Hückel and Möbius pericyclic transition-state circuits

For the **orbital-lobe reaction schemes**, use the [new reactant → transition-state → product figures](orbital_schemes/README.md). They pair allowed and forbidden approaches for all eight families: [overview](orbital_schemes/figures/orbital_selection_overview.png) · [eight-page PDF](orbital_schemes/pericyclic_orbital_schemes.pdf).

The [allowed-circuit atlas](figures/allowed_orbital_circuits.png) shows the formally thermally allowed orbital topology for each of the eight reaction families in the existing pericyclic dataset. The [paired comparison](figures/paired_huckel_mobius_approaches.png) shows both proposed approaches for every family, including the formally forbidden counterpart. PNG, SVG, and PDF exports of both figures are in [`figures/`](figures/).

The circles are **active orbitals**, not a 3D molecular structure. Solid dark edges are π bonds, gray edges are conjugated links, green dashed edges are forming interactions, and orange dotted edges are breaking interactions. A pink slash represents one orbital phase inversion in a convenient canonical sign gauge. The slash can move under a change of orbital sign convention; the product of interaction signs around the full cycle is invariant.

For a thermal concerted approach, the idealized rule used by this dataset is:

| Cyclic orbital topology | Active electrons | Formal thermal label |
|---|---:|---:|
| Hückel, phase product +1 | 4k+2 | allowed |
| Möbius, phase product −1 | 4k | allowed |
| Hückel, phase product +1 | 4k | forbidden |
| Möbius, phase product −1 | 4k+2 | forbidden |

This is an atlas of **candidate orbital interaction networks** from [`templates.json`](../../pericyclic_thermal_dataset/templates.json), not optimized transition-state geometries. It depicts each unadorned reaction template; dataset rows also contain substituents and random orbital sign gauges. The script checks all 16 template/approach combinations against a stored matrix, phase product, electron count, and label; those row IDs are recorded in [`example_manifest.json`](example_manifest.json). A formal “allowed” label does not establish a feasible geometry or a low barrier. Antarafacial [2+2] and [4+4] circuits are especially geometry-sensitive.

The exact template and label construction is in [`pericyclic_dataset.py`](../../pericyclic_dataset.py). The scientific selection rule follows Woodward and Hoffmann’s [electrocyclic stereochemistry paper](https://doi.org/10.1021/ja01080a054) and their broader [orbital-symmetry treatment](https://onlinelibrary.wiley.com/doi/10.1002/anie.196907811). Those are **scientific sources for the rule**, not the source code for these figures.

Regenerate from the OPSandbox root:

```bash
python VisualizingChemReps/pericyclic_ts_atlas/make_atlas.py
```

Requires NumPy and Matplotlib. The figures use the local `pericyclic_thermal_dataset/` files and do not require a GPU.
