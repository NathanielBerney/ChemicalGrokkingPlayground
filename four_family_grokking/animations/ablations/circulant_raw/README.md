# Raw circulant: embedding, readout, and output-frequency ablations

This report evaluates the archived step-40,000 checkpoint used by
[circulant_raw_frequency.gif](../../gifs/circulant_raw_frequency.gif).
The script is [ablate_circulant_raw.py](../../../ablate_circulant_raw.py);
all numerical results are in [summary.json](summary.json). The checkpoint
and original animation are unchanged.

## What the Gini measures

The GIF's Gini is computed on nonconstant DCT power of the two model outputs
after sorting all 10,000 examples by molecule size. This is a spectrum across
different molecule sizes, not sparsity of learned embedding weights or
physical orbital frequencies. The output Gini is **0.991 at initialization**
and **0.943 at step 40,000**; the exact target's Gini is **0.892**.
The final output's strongest 32 nonconstant modes carry **22.8%** of its
nonconstant power. Its strongest 1% of modes carry **45.3%**, and its
strongest 10% carry **94.9%**. The inverse-participation effective mode count
is about **313**. The spectrum is concentrated, but it is not a 32-mode
solution.

The final checkpoint has **25.2% train** and **23.9% held-out** strict
relative accuracy (both HOMO and LUMO within 5%), with held-out normalized
MSE **0.460**. This is the useful baseline here; the original absolute
0.05-energy-unit accuracy is nearly saturated by predicting zero.

## Weight interventions

The [weight ablation plot](weight_ablation.png) removes 10% or 25% of 128
channels. For the shared bit embeddings, selected channels of the 0 and 1
vectors are set to their midpoint, removing bit-value information in those
channels while preserving their common component. For position vectors,
selected channels at the 16 bit positions are zeroed; the CLS position stays
fixed. For the final readout, selected weight columns are zeroed while the
bias stays fixed. Channels are selected by largest magnitude, smallest
magnitude, or one of five random equal-size controls.

| 10% channels removed | Largest-magnitude test MSE / baseline | Random mean | Smallest-magnitude |
| --- | ---: | ---: | ---: |
| Shared bit-embedding difference | 2.97 | 1.83 | 1.07 |
| Position vectors | 1.84 | 1.11 | 1.02 |
| Final readout weights | 0.95 | 0.96 | 1.00 |

The largest embedding and position channels matter more than matched
controls. The readout intervention's small MSE improvement is not a strict
accuracy gain: removing its largest 10% of channels drops held-out strict
accuracy from **23.9% to 2.6%**. Separate Ginis of absolute weight magnitudes
are 0.380 for the bit-embedding difference, 0.324 for position-vector norms,
and 0.054 for readout-column norms. These weight Ginis describe different
objects from the output-power Gini and should not be compared as if they were
the same statistic.

## Output-frequency interventions

The two outputs are continuous HOMO/LUMO **regression predictions**, rather
than classifier logits. The [32-mode ablation plot](output_mode_ablation.png)
filters their ordered-size DCT after inference. Keeping only the 32 lowest
nonconstant modes raises held-out MSE from **0.460 to 1.119** and drops strict
accuracy from **23.9% to 0.9%**. Keeping the 32 strongest modes gives MSE
**1.124** and strict accuracy **1.3%**. Removing the strongest 32 also hurts
(MSE **0.667**, strict accuracy **0.6%**). Those modes contribute, but they
do not form a sufficient solution.

A broader [low-pass sweep](lowpass_sweep.png) gives a limited useful result.
The 7,000 original held-out examples are sorted by size and alternated into
3,500 calibration and 3,500 audit examples. The cutoff **2,000** is chosen
by calibration-half normalized MSE, then scored on the untouched audit half:

| Audit half | Unfiltered | Retain 2,000 modes |
| --- | ---: | ---: |
| Normalized MSE | 0.256 | 0.113 |
| Strict 5% relative accuracy | 24.3% | 22.7% |

The filter reduces squared error without improving the strict relative
metric. Filtering uses model
predictions for the entire size grid but no audit labels; it is transductive
post-processing, not a learned weight circuit.

## Readout probes

The [readout probe plot](readout_probe.png) freezes the model at steps 0,
1,000, and 40,000. A ridge-regularized linear head is fitted to the
128-dimensional CLS states using training targets only. Its penalty is chosen
on an 80/20 split of the training indices and then refitted on all training
examples. At the final checkpoint, strict held-out accuracy rises from
**23.9% to 31.3%**, while held-out MSE moves from **0.460 to 0.454**.
An analogous two-output calibration reaches **28.4%** strict accuracy;
a linear probe on the original 16 input bits reaches **1.0%**. At initialization
the hidden-state probe reaches **2.7%**, and at step 1,000 **14.2%**.
The hidden states therefore contain some learned signal that the original
readout does not fully use, but the result is far from a grokked solution.

For comparison, the already archived log-target circulant rerun reaches
**66.0% held-out** strict relative accuracy at step 40,000, with output
Gini **0.999**. That is a different training objective and does not arise
from these checkpoint ablations.

## Limits

These are interventions on one archived run and one train/test split.
Zeroing weight channels can move activations off their training distribution.
The random controls have five draws; they are not a multi-seed retraining
study. The low-pass sweep chooses a cutoff on half of the held-out examples
and reports the disjoint audit half. A high spectral Gini alone is not evidence
for a causal Fourier circuit or for grokking.
