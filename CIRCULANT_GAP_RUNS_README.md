# Circulant HOMO–LUMO gap experiments

The new gap experiments live in
[`four_family_grokking/circulant_gap_probe/`](four_family_grokking/circulant_gap_probe/README.md).
The linked README contains the run map, results, metric definitions, tmux
session names, and commands to regenerate the figures.

The original n=42 **matrix-input** circulant experiment remains in
[`CIRCULANT_GROKKING_README.md`](CIRCULANT_GROKKING_README.md). The new gap
experiments use the 10,000 distinct-size, valence-constrained circulant
dataset and encode its Hamiltonian by `n`; no dense 40,000×40,000 matrix is
materialized.

Key new figures are in the gap-probe directory:

- `architecture_map.svg` and `.png`: Nanda's architecture beside our gap
  architectures, with random holdout and larger-ring scores.
- `gap_accuracy.png`: strict 1% and 5% relative train/test gap accuracy.
- `gap_by_size.png`: gap predictions and relative error versus ring size.
- `architecture_comparison.png`: bit-input architecture controls.
- `fourier_progress/`: checkpointed Fourier spectra, Gini trajectories,
  output-mode filtering, and PCA for the gap models.

The modular-arithmetic PCA control is in
[`nanda_original_reproduction/pca_analysis/`](nanda_original_reproduction/pca_analysis/README.md).
Final-state PCA for all four chemical families is in
[`four_family_grokking/pca_final_families/`](four_family_grokking/pca_final_families/README.md).

The matched **dense Hückel matrix vs π graph vs full molecular graph vs SMILES**
gap experiments on 10,000 distinct heteroatom molecules are in
[`four_family_grokking/representation_gap_probe/`](four_family_grokking/representation_gap_probe/README.md).
This directory contains train/test learning curves, PCA, site-permutation
probes, two-seed comparisons, and structure-only controls.
