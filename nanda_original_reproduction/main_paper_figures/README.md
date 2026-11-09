# Nanda et al. main-body figure reproduction

These figures use the locally reproduced mod-113 transformer in
`../run_seed0/last.pt`, the five-seed metrics in `../run_seed*/metrics.csv`,
and 41 seed-0 snapshots in `../figure_run_seed0/figure_snapshots/`.
They follow Figures 1–7 in [the paper](https://arxiv.org/html/2301.05217v3)
and the definitions in the authors' [analysis code](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper).
The entire set is in `main_figures.pdf`; each figure is also a PNG.

| Paper figure | Local artifact | Quantity shown |
| --- | --- | --- |
| Extra | `figure00_fourier_basis.png` | Orthonormal real Fourier basis and one learned sine/cosine pair |
| 1 | `figure01_fourier_algorithm.png` | Rotation-algorithm schematic with actual model logits for a=17, b=29 |
| 2 | `figure02_learning_curves.png` | Train/test accuracy and loss, five seeds |
| 3 | `figure03_weight_fourier_norms.png` | Fourier norms of embedding W_E and neuron-logit map W_L |
| 4 | `figure04_periodicity_and_logit_fft.png` | Attention weight, MLP neuron 0, and the 2D input Fourier norm of logits |
| 5 | `figure05_neuron_frequency_logit_map.png` | Single-frequency neuron fits and Fourier-space W_L for k=14 neurons |
| 6 | `figure06_frequency_ablation.png` | Held-out loss after removing each input-frequency component |
| 7 | `figure07_progress_measures.png` | Excluded/restricted loss, Fourier Gini, and weight norm across training |
| Table 1 | `table01_trig_readout.csv` | Cosine/sine readout directions and their variance explained |

## Results and differences from the pictured run

The local seed-0 model has **four dominant Fourier frequencies: 14, 41, 42,
and 52**. The paper's pictured checkpoint has five: 14, 35, 41, 42, and 52.
The fifth-largest frequency in our neuron-logit map is over 600 times weaker
in squared amplitude than the four active frequencies, so the plots use the
four modes actually learned. The five-seed accuracy reproduction is in the
parent directory. Figure 1 is a newly drawn explanatory schematic; the
remaining figures are computed from local model weights, activations, logits,
and saved training checkpoints.

For Figure 5, 402 of 512 neurons have over 85% of their variance explained
by the degree-2 Fourier terms of one active frequency. The eight Table 1
readout directions have 96.8–99.0% variance explained by a single
cos(2πk(a+b)/113) or sin(2πk(a+b)/113) wave.

For Figure 6, original held-out cross-entropy is 2.14e-7. Keeping only the
four active input Fourier modes **improves** it to 5.03e-8 at 100% accuracy.
Removing any one of these modes raises loss to 0.022–0.604; removing any
other mode leaves it below 2.16e-7. Figure 7 records the exact underlying
values in `figure07_progress.csv` and the final summary in `manifest.json`.

## Reproduce locally

From the parent `nanda_original_reproduction` directory:

```bash
CUDA_VISIBLE_DEVICES=1 /home/nate/anaconda3/envs/grokking/bin/python reproduce_main_figures.py --gpu 0
```

The snapshot run is controlled by tmux session `nanda_main_figures_seed0`;
`run_figure_checkpoints_tmux.sh` starts the 40,000-step run with one archived
checkpoint every 1,000 steps. The figure analysis was also run in tmux as
`nanda_main_figures_analysis`. Both have finished and are idle.

The real Fourier basis has orthogonality error below 6e-15. The efficient
sum-residue projection used for Figures 6–7 matches an explicit 2D cosine/
sine projection to below 1e-15 on a random matrix. The activation-capture
forward pass exactly matches the model's ordinary forward pass, and the
snapshot rerun's final parameters exactly match `run_seed0/last.pt`.
