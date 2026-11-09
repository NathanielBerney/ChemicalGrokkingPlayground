# Nanda et al. original-task reproduction

This directory reproduces the main **modular addition** experiment in
[Nanda et al., *Progress measures for grokking via mechanistic
interpretability*](https://arxiv.org/abs/2301.05217). It is separate from the
circulant HOMO/LUMO experiment in the parent directory.

The authors' [reference code](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper)
uses a one-layer, four-head causal transformer with a 128-wide residual
stream and 512-wide ReLU MLP, no LayerNorm, `p=113`, a 30% train split, and
full-batch AdamW (`lr=0.001`, `weight_decay=1.0`, betas `0.9,0.98`) with a
ten-step warmup. `train_modular.py` implements that architecture and update
rule locally without the reference code's Weights & Biases service. It
evaluates on held-out pairs every 100 steps rather than every step; this
does not alter training gradients.

Each seed has a `run_seedN/` directory containing `config.json`,
`metrics.csv`, checkpoints, and timing. The runs use at most two H200s:

- GPU 1: tmux session `nanda_mod113_seed0` for seed 0, followed by seeds 2
  and 4 in `nanda_mod113_seeds2_4`.
- GPU 0: tmux session `nanda_mod113_seeds1_3` for seeds 1 and 3, run
  sequentially.

Attach with `tmux attach -t SESSION_NAME`. Stop a run using Ctrl-C there or
`tmux kill-session -t SESSION_NAME`. The latest checkpoint is saved every
2,000 steps. To resume a seed, use `--resume run_seedN/latest.pt` with the
same `--seed N` and `--outdir run_seedN`.

After the runs, `python analyze_runs.py` creates `reproduction_curves.png`
and `reproduction_summary.json` with the first step reaching 99% train and
test accuracy. Rising held-out accuracy long after training reaches 99%
reproduces the paper's basic grokking result. The two Fourier Gini curves
track the concentration of learned embedding and neuron-logit weights.

`python ablate_fourier.py --gpu 1` scores the seed 0 final checkpoint after
keeping or removing the five strongest output frequencies of the learned
neuron-logit map. The corresponding `run_seedN/fourier_ablation.json` files
show that all five final models retain 100% held-out accuracy with their five
strongest frequencies, while removing them leaves 0.0–3.8% accuracy. This
checks that the final models rely on small Fourier subspaces.

## Results on this H200 host

| Seed | First 99% train step | First 99% test step | Seconds to 99% test | Full 40,000-step run |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 200 | 12,700 | 39.8 | 140.2 s |
| 1 | 200 | 13,900 | 40.3 | 107.6 s |
| 2 | 200 | 4,700 | 17.2 | 117.0 s |
| 3 | 200 | 7,400 | 14.3 | 109.6 s |
| 4 | 200 | 9,300 | 17.2 | 90.6 s |

All five ended at 100% test accuracy. The median generalization onset was
step 9,300; the median full-run time was 109.6 seconds. These times include
the 100-step evaluation cadence and concurrent GPU activity on GPUs 0 and 1.

This is a **faithful local reimplementation**, not a byte-for-byte execution
of the authors' notebook. The paper's more detailed restricted/excluded
losses and manual circuit dissection are separate analyses; the curves here
benchmark the original task and Fourier sparsity trend.

An additional [PCA analysis](pca_analysis/README.md) projects the learned
integer embeddings and full checkpoint weight trajectory. Its first two
embedding PCs reveal a modular-frequency ring near generalization onset.

The [training animations](animations/README.md) show the saved seed-0 logit
map, token embedding PCA plane, Fourier Gini curves, and Fourier frequency
spectra evolving from initialization through step 40,000. Recreate them with
`python animate_training.py` from this directory.
