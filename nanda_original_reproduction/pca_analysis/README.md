# PCA of the modular-addition reproduction

This folder applies PCA to the existing seed-0 modular-addition snapshots in
`../figure_run_seed0/figure_snapshots/`. It follows the style of the
learned-embedding PCA visualization in
[Liu et al., *Towards Understanding Grokking*](https://arxiv.org/html/2205.10343v2),
which builds on the grokking task introduced by
[Power et al.](https://arxiv.org/abs/2201.02177). It is a new analysis of
this workspace's [Nanda-style reproduction](../README.md).

## Figures

- `embedding_pca_evolution.png`: all 113 integer-token embeddings at steps
  0, 5,000, 10,000, 13,000, and 40,000, projected into the **final**
  embedding PCA plane. The common projection and scale make formation of
  structure visible across time.
- `embedding_pca_frequency_pairs.png`: final PC1–PC2, PC3–PC4, and PC5–PC6
  planes, colored by each pair's dominant modular Fourier phase.
- `weight_trajectory_pca.png`: PCA of each complete 226,816-parameter
  checkpoint vector, colored by step and held-out accuracy.
- `pca_structure_progress.png`: the fraction of embedding Fourier power at
  the final first-PC-pair frequency beside train/test accuracy.
- `pca_summary.json`: coordinates, explained variance, and frequency
  measurements used by the figures.

## Findings

The first two PCs of the final input embeddings explain **25.6%** of their
variance. Together they form a phase-ordered ring: **90.6%** of their Fourier
power is at modular frequency **41**. The ring becomes clear near step
13,000, when held-out accuracy passes 99% (first recorded at step 12,700).
Other PCs carry additional frequency mixtures, so a two-dimensional PCA is
only one view of the learned algorithm.

The first two PCs of the **full weight trajectory** explain **86.7%** of
variance across its 41 saved checkpoints. That path is descriptive; it does
not by itself identify the Fourier circuit or separate learning from weight
decay.

Recreate the figures with:

```bash
cd /mnt/das_storage/nates_workspace/OPGrokkingDatasets/OPSandbox/nanda_original_reproduction/pca_analysis
/home/nate/anaconda3/envs/grokking/bin/python analyze_nanda_pca.py
```
