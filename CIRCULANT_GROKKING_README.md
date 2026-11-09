# Circulant HOMO/LUMO grokking experiment

This adapts [Nanda et al., *Progress measures for grokking via mechanistic
interpretability*](https://arxiv.org/abs/2301.05217) to the original
`dataset/circulant_n42.npz` set. It uses the 10,000 mathematical n=42 offset
patterns and predicts the two orbital energies at eigenvalue indices 20 and
21. This is the original mathematical dataset, which does **not** impose
carbon valence or graph planarity.

## Method

- Input: the raw `42 × 42` Hückel matrix, treated as 42 row tokens. The model
  never receives the precomputed eigenvalues as input.
- Output: HOMO and LUMO energies in units of `|β|`; train targets are
  standardized using only training-set statistics.
- Split: random, reproducible 30% train / 70% held-out test, seed 0. All
  samples have different offset patterns.
- Architecture: one transformer layer with 128-wide embeddings, four
  attention heads, and a 512-wide ReLU MLP, with no LayerNorm; a CLS readout
  produces two scalars.
- Optimizer: full-batch AdamW, learning rate `0.001`, weight decay `1.0`, up
  to 40,000 steps. This follows the paper's main settings where applicable.
  The **42-token matrix input and mean-squared-error regression** differ
  materially from the paper's three-token modular-addition classification.
- Monitoring: train/test RMSE and MAE every 200 steps, plus the Fourier Gini
  coefficient of the input row-projection weights. The latter is an
  exploratory analogue of the paper's Fourier sparsity progress measure.

The exact mathematical baseline is the DFT of a circulant matrix's first
row. Sorting those 42 eigenvalues and taking indices 20 and 21 reproduces
the dataset targets with approximately `1.3 × 10⁻¹⁵` RMSE. Thus the dataset
has a known Fourier solution; a neural model must discover an approximation
to it from the training examples.

## Run and control

The run uses **one H200 NVL**, selected by `CUDA_VISIBLE_DEVICES=0`. The
existing `/home/nate/anaconda3/envs/grokking` environment supplies PyTorch.
It runs in the `circulant_grok_seed0` tmux session:

```bash
tmux attach -t circulant_grok_seed0
```

Press Ctrl-C inside tmux to stop the run, or use
`tmux kill-session -t circulant_grok_seed0`. The run saves a checkpoint every
2,000 steps at `circulant_grokking_runs/seed0_tmux/latest.pt`. To resume after
stopping, run `grok_circulant_n42.py` with `--resume` pointing to that file,
using the same seed, output directory, and train fraction.

Results are written to `circulant_grokking_runs/seed0_tmux/metrics.csv`. Run
`python analyze_circulant_grok.py` to refresh `analysis.json` and
`learning_curves.png`.

The paper's restricted/excluded losses involve ablating Fourier components
of a 113-class modular-addition output. This regression task has only two
outputs, so they do not transfer directly. `probe_circulant_fourier.py`
instead measures the effect of ablating frequency components in the model's
first row-projection weights. This is explicitly an **adaptation**, not an
exact replication of the paper's progress measures.

An operational grokking criterion for this first run is: train mean RMSE
below `0.05` while test mean RMSE remains above `0.4`, followed later by test
mean RMSE below `0.1`. Failure to reach the latter threshold in 40,000
steps means this run did not exhibit that delayed-generalization pattern;
it does not prove that no architecture or regularization setting can grok.
