# Pericyclic reactions drawn with orbital lobes

Each row runs **reactant orbitals → proposed transition state → product connectivity**. Each reaction has an allowed and a forbidden approach. Blue/orange and +/− indicate orbital phase, not electrical charge. A crossed reaction arrow and faded product mark a forbidden concerted thermal pathway.

Start with the [four-pathway overview](figures/orbital_selection_overview.png) or the [eight-page PDF](pericyclic_orbital_schemes.pdf). All figures also have editable SVG and individual PDF exports in [`figures/`](figures/).

| Reaction | Paired allowed/forbidden figure | Reactant orbital representation |
|---|---|---|
| Diels–Alder [4+2] | [PNG](figures/diels_alder_4plus2.png) · [SVG](figures/diels_alder_4plus2.svg) | Diene HOMO ψ₂ + ethene LUMO ψ₂ |
| [2+2] cycloaddition | [PNG](figures/cycloaddition_2plus2.png) · [SVG](figures/cycloaddition_2plus2.svg) | Ethene HOMO ψ₁ + ethene LUMO ψ₂ |
| [4+4] cycloaddition | [PNG](figures/cycloaddition_4plus4.png) · [SVG](figures/cycloaddition_4plus4.svg) | Diene HOMO ψ₂ + diene LUMO ψ₃ |
| 4π electrocyclic closure | [PNG](figures/electrocyclic_4pi.png) · [SVG](figures/electrocyclic_4pi.svg) | Butadiene HOMO ψ₂ |
| 6π electrocyclic closure | [PNG](figures/electrocyclic_6pi.png) · [SVG](figures/electrocyclic_6pi.svg) | Hexatriene HOMO ψ₃ |
| Claisen [3,3] | [PNG](figures/claisen_3_3.png) · [SVG](figures/claisen_3_3.svg) | Localized occupied π, σ(C–O), π orbitals |
| Cope [3,3] | [PNG](figures/cope_3_3.png) · [SVG](figures/cope_3_3.svg) | Localized occupied π, σ(C–C), π orbitals |
| Alder–ene | [PNG](figures/alder_ene.png) · [SVG](figures/alder_ene.svg) | Localized occupied π, σ(C–H), π orbitals |

## What is calculated and what is schematic

For cycloadditions the diagnostic donor–acceptor interaction is shown as **HOMO–LUMO**, rather than HOMO–HOMO. Electrocyclizations follow the terminal lobes of the occupied polyene HOMO. Their coefficients are calculated from the uniform open-chain model with `Hii=0` and nearest-neighbor `Hij=−1`:

`c_j(k) = sqrt(2/(n+1)) sin(j k π/(n+1))`, with `j,k=1,…,n`.

The script checks the eigenvector equation and normalization and checks the facing-lobe signs for both candidate approaches. Whole-fragment orbital signs are arbitrary; the lower cycloaddition component is gauged to make the left contact constructive. An antarafacial approach reverses which face is contacted at the right terminus, not the identity of the unperturbed reactant MO. A purple arrow marks this face change. Electrocyclic arrows use a common viewing convention at both ends.

The center columns show **qualitative frontier overlap or local orbital arrays**, not eigenvectors of the dataset TS matrix, density isosurfaces, or optimized 3D geometries. In particular, changing one face is a schematic representation of an antarafacial approach, not a calculation of the geometric deformation needed to achieve it. Lobe sizes are illustrative, with compressed coefficient scaling for legibility. The reaction drawings retain the atom ordering across each arrow.

The three π/σ reaction pages use localized occupied bond orbitals on the left and a local overlap basis in the center, including hybrid/1s contributions and reorganization of the σ centers. Those pages are explicitly **not a single HOMO calculation**. Their twisted lobes are a schematic depiction of the formal cyclic phase topology. The orientation angles are illustrative; they do not define overlap integrals or a TS geometry. Small numbers track active atoms, including the transferred H in the ene reaction.

Hückel/Möbius topology refers to the cyclic **basis-overlap** phase product. It must not be inferred by simply counting color changes in one selected HOMO, which can have its own nodes. The labels reproduce the existing dataset's formal thermal rule: Hückel/4k+2 and Möbius/4k are allowed. This establishes neither a barrier nor geometric accessibility. The formally allowed antarafacial [2+2] and [4+4] cases are flagged explicitly; an antarafacial approach is particularly restrictive for ethene.

Products show **connectivity**, without substituent stereochemistry. A forbidden row does not claim that the product is impossible through another mechanism or under irradiation. The unsubstituted Cope product is constitutionally identical to its reactant; the atom numbers show the bond exchange.

## Validation and provenance

- [`orbital_manifest.json`](orbital_manifest.json) records analytic MO coefficients, the 16 checked dataset approaches, and the exact reactant/product bond orders used to draw the structures.
- [`product_validation.json`](product_validation.json) records a separate RDKit check: valid valences, equality of the drawn reactants to the stored template SMILES, agreement with the intended product SMILES, and conservation of molecular formula for all eight families.
- The new rendering code is [`../make_orbital_schemes.py`](../make_orbital_schemes.py), written for this request using NumPy/Matplotlib. No paper figure, external drawing code, or quantum-chemistry orbital mesh was copied.
- Hoffmann and Woodward's *The Conservation of Orbital Symmetry* (1968), [author-hosted PDF](https://roaldhoffmann.com/sites/default/files/fromd6/29.pdf), was consulted for the orbital-phase reasoning, electrocyclic rotations, and cycloaddition/sigmatropic selection rules. Their 1965 [electrocyclic paper](https://doi.org/10.1021/ja01080a054) is also relevant.
- Historical sources for the related formulations are Zimmerman's [Möbius orbital-correlation paper I (1966)](https://doi.org/10.1021/ja00959a052) and Houk's [frontier-orbital treatment of cycloadditions (1975)](https://doi.org/10.1021/ar50095a001). Bibliographic records were checked; the paywalled full texts were not used as implementation instructions.

Regenerate from the OPSandbox root:

```bash
PYTHONDONTWRITEBYTECODE=1 python VisualizingChemReps/pericyclic_ts_atlas/make_orbital_schemes.py
/home/nate/anaconda3/envs/sigmodia/bin/python VisualizingChemReps/pericyclic_ts_atlas/orbital_schemes/validate_products.py
```

Rendering needs NumPy/Matplotlib; the separate product check needs RDKit. No training or GPU use is involved.
