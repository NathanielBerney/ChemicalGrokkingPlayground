# PCA of all four chemical benchmark models

This directory analyzes final checkpoints from the original four-family
study. Circulant, Möbius, and linear use their log-target 40,000-step models
from `../runs_log/`; pericyclic uses its 120,000-step classifier from
`../runs/`. These are final-state PCA views; checkpoint trajectories for the
new circulant gap models are in
[`../circulant_gap_probe/fourier_progress/`](../circulant_gap_probe/fourier_progress/README.md).

## Figures and data

- `frontier_hidden_pca.png`: final readout features for circulant, Möbius,
  and linear, colored by `log₁₀(n)`.
- `frontier_input_embedding_pca.png`: effective 0/1 bit-token embeddings
  for those three models, colored by binary place value.
- `pericyclic_hidden_pca.png`: top-two-PC view colored by thermal label
  and number of active transition-state sites.
- `pericyclic_signal_pca.png`: PC1 versus the readout-aligned PC7, plus
  PC7 label distributions.
- `pericyclic_input_weight_pca.png`: eight row-projection and site-position
  vectors in separate PCA planes.
- `nanda_vs_chemical_pca.png`: side-by-side modular embedding ring, gap-model
  bit embeddings, and pericyclic label signal.
- `pca_summary.json`: explained variance, readout alignment, and numerical
  correlations.
- `analyze_family_pca.py`: regeneration script.
- `compare_nanda_chemical_pca.py`: combined comparison figure.

## What appeared

The frontier-energy readout states largely encode size. PC1+PC2 explain
**99.6%**, **99.7%**, and **99.9%** of hidden variance for circulant,
Möbius, and linear; PC1 correlates with `log(n)` at **0.9992**, **0.9997**,
and **0.9998**. Their input bit-token PCA planes show parallel 0/1 pairs
because all three architectures share the same two bit embeddings across
positions. That geometry is imposed by tokenization, unlike the learned
modular-frequency ring in the [Nanda PCA control](../../nanda_original_reproduction/pca_analysis/README.md).

The pericyclic classifier is different. Its top two hidden PCs explain
**48.1%** of variance, but contain only **0.22%** of the final linear
readout weight. Labels therefore look mixed in the usual PC1–PC2 plot.
**PC7** explains only **3.43%** of hidden variance yet carries **81.4%** of
the readout weight, and its correlation with the binary label is **−0.852**
(PCA sign is arbitrary). `pericyclic_signal_pca.png` reveals the label
separation that a top-variance PCA view would miss.

PCA is descriptive. A small number of principal components can summarize
size, weight decay, or nuisance variation without identifying the learned
chemical decision rule.

Regenerate with:

```bash
cd /mnt/das_storage/nates_workspace/OPGrokkingDatasets/OPSandbox/four_family_grokking/pca_final_families
/home/nate/anaconda3/envs/grokking/bin/python analyze_family_pca.py
/home/nate/anaconda3/envs/grokking/bin/python compare_nanda_chemical_pca.py
```
