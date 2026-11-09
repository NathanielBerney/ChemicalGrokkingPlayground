# Four-family training animations

The gallery follows the original circulant, Möbius, and linear frontier-energy
models with raw targets, their separate log-magnitude target controls, and the
pericyclic thermal-label model. These are **archived reruns**: the earlier
experiments saved only final/latest model checkpoints. The reruns use the same
seed, data split, architecture, optimizer, and objective as each corresponding
run, and write model-only snapshots every 1,000 energy steps or 2,000
pericyclic steps. Small numerical differences from the earlier GPU runs are
possible. Every animation uses its own rerun's checkpoints and metrics.

| Dataset | Raw energy embeddings | Raw energy frequency | Log energy embeddings | Log energy frequency |
| --- | --- | --- | --- | --- |
| Circulant | [GIF](gifs/circulant_raw_embeddings.gif) | [GIF](gifs/circulant_raw_frequency.gif) | [GIF](gifs/circulant_log_embeddings.gif) | [GIF](gifs/circulant_log_frequency.gif) |
| Möbius | [GIF](gifs/mobius_raw_embeddings.gif) | [GIF](gifs/mobius_raw_frequency.gif) | [GIF](gifs/mobius_log_embeddings.gif) | [GIF](gifs/mobius_log_frequency.gif) |
| Linear | [GIF](gifs/linear_raw_embeddings.gif) | [GIF](gifs/linear_raw_frequency.gif) | [GIF](gifs/linear_log_embeddings.gif) | [GIF](gifs/linear_log_frequency.gif) |

The pericyclic classifier has [embedding](gifs/pericyclic_embeddings.gif) and
[cyclic-reference frequency](gifs/pericyclic_frequency.gif) animations.

## What each axis means

**Energy embeddings.** The three energy transformers receive 16 binary
digits of the molecule size `n`. The left panel shows the 32 effective
0/1-position token vectors: shared learned bit embeddings plus learned
position vectors. Paired tokens are joined. The right panel shows the
128-dimensional CLS readout state for 2,200 fixed examples, colored by
`log₁₀(n)`. Each PCA plane is fitted to the *final* checkpoint and held fixed
through time. Within each frame, coordinates are centered and divided by
their 2D RMS radius; the raw radius is reported. Lower panels show strict
5% relative train/test accuracy and normalized training-objective MSE. The
original 0.05 absolute energy cutoff is nearly saturated by a zero-energy
baseline, so the stricter metric is more informative here.

**Energy frequencies.** For all 10,000 distinct molecule sizes, we sort by
`n` and take an orthonormal DCT-II of the two normalized model outputs and
the 128 readout features. The top panels show output power per mode versus
the exact target, and the hidden-state power share in size-frequency bands.
Gini numbers describe concentration of *nonconstant* power. The lower-left
panel splits full-grid normalized MSE into DCT modes 0–32 and 33 onward;
those two terms sum exactly to the full-grid MSE. The lower-right panel shows
strict relative train/test accuracy. The DCT runs along **molecule size**,
not around physical orbital sites. The displayed output vertical axis adapts
per frame; time-series axes stay fixed. Step 0 is a separate marker because
no intermediate model states were saved before step 1,000. The existing [orbital Fourier
bases](../fourier_bases/) are separate physical reference plots.

**Pericyclic embeddings.** The upper panels track the eight matrix-column
projection vectors and eight site-position vectors. The lower-left panel
projects CLS readout states onto the final checkpoint's top principal
component and final readout direction; this exposes the thermal-label
signal that the first two variance PCs can miss. All axes remain fixed to
the final checkpoint, with per-frame centering and RMS scaling as above.
The lower-right panel shows full train/test accuracy.

**Pericyclic frequencies.** We evaluate 256 fixed examples from each label.
For each transition-state matrix, its signed active-site loop defines a
gauged periodic or anti-periodic reference Fourier basis. We project the
post-attention, post-MLP site-token vectors onto that basis, average each
example's power share in four relative-wavenumber bands, and track the upper
two bands through training. Raw site-token RMS is reported separately.
Because these matrices have varying couplings and site energies, the
reference modes are **not** exact eigenvectors of each matrix. The plot
describes learned site-token structure, not proof of an orbital mechanism.
Its step-0 point is separate from the 2,000-step checkpoint curve.

## What to look for

For the log-target energy runs, output-frequency Gini begins around `0.991`
and ends around `0.999` in all three families. Spectral concentration is
already high before accurate predictions; it is not by itself a grokking
signal. The lower-left frequency panel is more diagnostic: it shows which
parts of the normalized prediction error fall with training. The raw-target
runs retain much more high-mode error and correspondingly low strict
relative accuracy.

In the pericyclic site spectrum, allowed and forbidden examples already
have different high-mode shares at step 0. Those initial differences reflect
the input matrices and reference bases as well as the random model. A later
shift should be read alongside the classification curve, not as independent
evidence that a Fourier circuit was learned.

The compressed arrays in `analysis/*.npz` contain frame data, numerical
metrics, and spectra. `energy_summary.json` and `pericyclic_summary.json`
give final measurements.

The [raw circulant ablation report](ablations/circulant_raw/README.md)
tests embedding, position, readout, and output-frequency interventions on
the checkpoint behind the raw circulant frequency GIF.

Regenerate the animations after reruns finish:

```bash
bash run_animation_reruns.sh
/home/nate/anaconda3/envs/grokking/bin/python animate_energy.py
/home/nate/anaconda3/envs/grokking/bin/python animate_pericyclic.py
```
