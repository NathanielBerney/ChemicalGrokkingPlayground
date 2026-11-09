# First experiment: 40,000 steps

The mode-ordered control learned a well-generalizing, frequency-specific map.
The main energy-sorted task did **not** show comparable spectral generalization
or a clear grokking transition in these runs.

Start with the [mode-control causal animation](animations/mode_seed0/causal_ablations.gif),
then compare [mode-control frequencies](animations/mode_seed0/frequencies.gif)
against [sorted-target frequencies](animations/sorted_seed0_wd0p1/frequencies.gif).
The [full gallery](animations/index.html) includes every run and embedding view.

| Target | Seed | Weight decay | Final train RMSE | Final test RMSE |
|---|---:|---:|---:|---:|
| Fourier-mode order | 0 | 1.0 | 0.05076 | 0.05166 |
| Fourier-mode order | 1 | 1.0 | 0.11971 | 0.11989 |
| Sorted energy | 0 | 1.0 | 0.41361 | 0.65547 |
| Sorted energy | 1 | 1.0 | 0.63442 | 0.64047 |
| Sorted energy, follow-up | 0 | 0.1 | 0.10831 | 0.68133 |
| Sorted energy, follow-up | 1 | 0.1 | 0.05661 | 0.68079 |

All errors use the full split and all 42 eigenvalues, in `|beta|` units.
These are the final checkpoints, without choosing a best test checkpoint.
Mini-batch optimization produces visible fluctuations, especially in the
mode-ordered control. The learned linear control achieves approximately
`1e-14` test RMSE for mode targets and `0.724–0.728` for sorted targets.

## Evidence from the control

Removing a real site-frequency subspace before the first attention block
concentrated **89.65%** (seed 0) and **88.94%** (seed 1) of the resulting
prediction-change energy in the corresponding output pair `k,42-k`, averaged
over removed frequencies. At initialization, those values were **5.27%** and
**4.63%**. Interventions before the second block gave similar final selectivity
(89.62% and 89.03%). The Nyquist frequency uses its single output.

This is visible as two diagonals in the ablation heatmaps. It supports the
model's causal use of frequency-specific information. It does not establish
a unique internal implementation of the DFT, and this control explicitly
aligns the targets by Fourier mode.

The exact cosine coupling-sensitivity pattern becomes recognizable in the
learned response. Relative finite-difference errors remain approximately
0.290 and 0.329 on the four fixed sensitivity probes, so the continuous local
map is approximate. The ordinary held-out task uses binary coupling patterns;
the sensitivity test probes nearby continuous couplings.

This control generalized early: train and test first crossed RMSE 0.1 at the
same logged step (800 for seed 0; 400 for seed 1). Those are descriptive first
crossings, not sustained convergence thresholds. The histories do not show
a long memorization phase followed by delayed test improvement.

## What the main task shows

With weight decay 1, sorted-spectrum learning was weak and seed dependent.
Reducing weight decay to 0.1 allowed a much closer training fit, while test
RMSE stayed near 0.68. Both lower-decay runs crossed train RMSE 0.1 around
12,000–12,800 steps, and neither crossed test RMSE 0.1 within 40,000 steps.
The stronger regularization therefore contributed to underfitting, but easing
it did not produce the desired generalization.

One useful caution appears in `sorted_seed1`: its final aggregate non-DC
Fourier Gini is about 0.952, while only about 2.4% of input-weight power remains
outside DC and predictions are poor. A large Gini alone is not evidence of
learning the spectral rule. The animations display both quantities.

These results support using the mode-aligned task to validate the mechanism
probes. They leave discovery of the full spectrum-plus-ordering rule unresolved
for this transformer and training setup. Forty thousand steps and two seeds
do not rule out a later transition or success with a different architecture,
sample count, or optimizer regime.

See [methods and reproduction instructions](README.md), [raw results](results.json),
and each run's `analysis/diagnostics.npz` for the checkpoint data.
