# Fourier and Gini progress probes for the circulant gap

This folder contains Nanda-inspired analyses of the new gap models. It is
separate from the original gap runs in the parent folder. The three models
were retrained with the same seed-0 3,000/7,000 split and their original
optimizer settings, saving model-only snapshots at step 0, steps 200–800,
and every 1,000 steps through 40,000.

```text
fourier_progress/
├── runs/log_bits/             Shared-bit transformer, cosine learning rate
├── runs/log_posbits/          Position-aware transformer, cosine learning rate
├── runs/log_scalar/           log(n) MLP, constant learning rate
├── analyze_fourier.py         Model probes and figure generation
├── analysis.json              Per-step numerical measures and final filters
├── fourier_progress.png       Accuracy, Gini, and spectral error over steps
├── final_spectra.png          Model/target cosine spectra and gap curves
├── spectral_alignment.png     Model/target coefficient alignment over steps
├── mode_filtering.png         Output-mode filtering effects on held-out scores
├── nanda_gini_reference.png   Nanda and gap Gini curves, separately labeled
├── two_fourier_axes.png       Physical ring-site DFT versus size-axis DCT
├── analyze_pca.py             Embedding, hidden-state, and weight PCA
├── chemical_embedding_pca.png         Effective bit tokens across checkpoints
├── chemical_embedding_pca_final_zoom.png  Final bit-token geometry
├── chemical_bit_effect_norms.png      Effect of each binary place
├── chemical_hidden_pca.png            Final readout features by ring size
├── chemical_weight_trajectory_pca.png Full checkpoint-vector trajectories
└── pca_summary.json           PCA measurements and coordinates
```

Each `runs/<variant>/` contains `config.json`, `metrics.csv`, `latest.pt`,
`last.pt`, and `snapshots/step_*.pt`. The full console logs are
`runs/<variant>.log`. The tmux sessions are
`grok_gap_fourier_log_bits_gpu0`, `grok_gap_fourier_log_posbits_gpu1`, and
`grok_gap_fourier_log_scalar_gpu2`. All three runs have finished; the tmux
panes remain open at Bash prompts.

## Meaning of the Fourier measures

Nanda et al. analyzed Fourier power across the **113 modular input/output
classes** and used Gini coefficients to track concentration in those
learned embeddings and neuron-logit maps. Our input vocabulary is just
binary digits, so that weight-space probe has no equivalent 113-class
axis. Here the model is evaluated at all 10,000 ordered ring sizes, and an
orthonormal cosine Fourier transform (DCT-II) is applied along the **size
axis** to both its predicted `log(gap)` and its final hidden features.

We record:

- **Output-function Gini:** concentration of nonconstant cosine-mode power
  in predicted `log(gap)`.
- **Hidden-state Gini:** concentration of summed cosine-mode power in the
  final 128-dimensional readout features.
- **Low-mode and remaining error:** orthogonal contributions to full-grid
  log-gap MSE from modes 0–32 and 33 onward. Their sum equals the full MSE
  by Parseval's identity. The exact target has about 98.7% of its
  nonconstant power in modes 1–32.
- **Output-mode filtering:** reconstruction using modes 0–32, or removing
  modes 1–32 while retaining the mean, then scoring held-out gap accuracy.
- **Spectral alignment:** cosine similarity of the model and target's
  nonconstant coefficients. This checks whether concentrated power is in
  the correct modes and phases.

These are **functional output probes**. They do not ablate weights or
demonstrate that the model learned physical ring-site Fourier eigenvectors.
The physical orbital basis is shown separately in `two_fourier_axes.png`
and in the parent project’s
[`circulant_fourier_basis.png`](../../fourier_bases/circulant_fourier_basis.png).
The size-axis DCT avoids the artificial endpoint jump that a periodic DFT
would create between the smallest and largest rings. The script also records
a periodic-FFT Gini as a sensitivity check.

## What the Fourier probes found

The exact `log(gap)` function already has a **0.99938** cosine-power Gini;
the gap models begin with output-function Ginis around **0.991–0.9999**.
Their Gini curves therefore sit near the ceiling before accurate learning.
This differs sharply from the Nanda reproduction's embedding and
neuron-logit Ginis, which rise substantially during modular grokking.

The exact target puts **98.71%** of its nonconstant power in size modes
1–32. At step 40,000, the shared-bit and position-aware transformers have
low-mode relative RMSE of **0.22%** and **0.37%**, but remaining-mode relative
RMSE of **24.7%** and **36.0%**. The `log(n)` MLP reaches **0.05%** and
**0.49%**, respectively. Thus matching overall spectral concentration is
much easier than matching the detailed spectrum. The final transformer
functions visibly oscillate on small rings in `final_spectra.png`.

Filtering each model's output to modes 0–32 leaves only **27.6–28.5%**
held-out accuracy at the strict 1% gap threshold. Filtering the *exact
target* the same way gives **28.4%**. Removing modes 1–32 instead leaves
about **2.6%** accuracy. The high modes carry little aggregate power but
matter for pointwise 1% precision. These filters use predictions on the
whole size grid without labels and are post-hoc diagnostics, not trained
predictors or weight interventions.

At the final checkpoints, held-out 1% accuracy for `n≤1000` is **32.2%**
for shared-bit attention, **52.8%** for position-aware attention, and
**99.4%** for the `log(n)` MLP. This explains why aggregate accuracy can
look strong while the physically most distinct small rings remain hard.

## PCA follow-up

The [modular-addition PCA control](../../../nanda_original_reproduction/pca_analysis/README.md)
shows a frequency-colored ring in learned integer embeddings near its
generalization transition. The gap models show different structures:

- Shared-bit effective token pairs are parallel **by construction**;
  all 16 positions use the same two token embeddings plus a position vector.
- Position-aware bit pairs fan out, but do not form the modular ring. Their
  first two final embedding PCs explain **76.5%** of variance, up from
  **12.7%** at initialization. The model's full weight norm also shrinks
  from **68.1** to **20.3**, so PCA concentration alone cannot establish
  that the new geometry is the learned chemical rule.
- Final hidden features are strongly size-ordered: PC1+PC2 explain
  **99.3%** (shared-bit), **99.0%** (position-aware), and approximately
  **100%** (`log(n)` MLP). Their shapes are branches or a smooth size curve,
  rather than a periodic ring.
- PC1+PC2 explain **66.2%**, **91.2%**, and **98.0%** of the three full
  checkpoint weight trajectories. Those trajectories include optimizer
  and weight-decay effects.

The result is consistent with the task: these models learn a scalar
size-to-gap mapping from binary or logarithmic size. They never receive
the ring-site Hamiltonian or its orbital Fourier modes.

Regenerate the figures after all three tmux runs finish with:

```bash
cd /mnt/das_storage/nates_workspace/OPGrokkingDatasets/OPSandbox/four_family_grokking/circulant_gap_probe/fourier_progress
/home/nate/anaconda3/envs/grokking/bin/python analyze_fourier.py
/home/nate/anaconda3/envs/grokking/bin/python analyze_pca.py
```

The comparison with [Nanda et al.](https://arxiv.org/abs/2301.05217)
uses this workspace’s
[`nanda_original_reproduction/run_seed0/metrics.csv`](../../../nanda_original_reproduction/run_seed0/metrics.csv).
Those Gini curves and ours are plotted in adjacent panels because the axes
and their meanings differ.
