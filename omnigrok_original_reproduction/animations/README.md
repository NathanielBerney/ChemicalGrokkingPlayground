# QM9 training animations

These animations use four **archived reruns** of the original QM9 grokking
conditions: first `N=100` or `N=400` QM9 molecules, with a 50/50 train/test
split and initialization scale `α=1` or `α=3`. The original runs saved only
their final weights; the reruns save model-only snapshots every 1,000 optimizer
steps. Their train/test split and seed match the original runs, while GPU
training can lead to small numerical differences. Every plot uses metrics
computed from the same archived rerun as its model states.

| View | First 100 molecules: 50 train/test | First 400 molecules: 200 train/test |
| --- | --- | --- |
| Predictions and losses | [GIF](predictions_total100.gif) | [GIF](predictions_total400.gif) |
| Predictions and losses, linear step axis | [GIF](predictions_total100_linear_x.gif) | [GIF](predictions_total400_linear_x.gif) |
| Graph embedding space | [GIF](embeddings_total100.gif) | [GIF](embeddings_total400.gif) |
| Full weight trajectory | [GIF](weights_total100.gif) | [GIF](weights_total400.gif) |
| Test-error graph spectrum | [GIF](residual_spectrum_total100.gif) | [GIF](residual_spectrum_total400.gif) |
| Test-error graph spectrum, linear step axis | [GIF](residual_spectrum_total100_linear_x.gif) | [GIF](residual_spectrum_total400_linear_x.gif) |
| Atom-activation graph spectrum | [GIF](node_spectrum_total100.gif) | [GIF](node_spectrum_total400.gif) |

Each GIF compares the `α=1` control (left) with `α=3` (right) at the same
optimizer step. The predictions view plots every molecule's predicted versus
true isotropic polarizability. Blue points are training molecules and orange
points are test molecules. The dashed line is perfect prediction. Its lower
panels reveal **true full-subset MSE** through the current step. The prediction
vertical axes adjust per frame to accommodate the very large initial errors;
the loss axes stay fixed.

The linear-step variants use evenly spaced optimizer steps on the lower x axes,
including step zero. The lower MSE y axes and the spectrum bar y axes remain
logarithmic. The original log-step GIFs are retained for comparison.

The [N=400 channel ablation report](ablations/total400/README.md) checks which
pooled graph and readout channels matter for the saved predictions.
The [chemistry concept probe](chemistry_probes/total400/README.md) tests whether
the frozen pooled embedding distinguishes rings and functional groups after
controlling for element counts.
The [N=400 structure map](structure_map/total400/README.md) tests whether the
model actually uses the top PCA directions, validates a chemistry-linked
direction on additional QM9 molecules, and shows the learned weight,
activation, and scalar-output matrices.
The [N=400 non-frequency checks](nonfrequency_checks/total400/README.md)
probe each graph layer for QM9 properties and closed-walk counts, test
larger graphs, and verify atom-relabeling invariance.

The embedding view plots the 16-dimensional graph representation after the
second graph convolution and global sum pooling. Each model has its own PCA
directions fitted on its final representation. Those directions stay fixed
over time. Coordinates are centered and divided by their frame's RMS radius
to show geometry across the large change in raw scale; each panel reports the
raw radius. Color indicates the true target. Circles are training molecules;
triangles are test molecules. The two panels' PCA coordinates describe
structure within each model and are not a shared basis.

The weight view applies PCA to each model's saved full 30,321-parameter
vectors. Its axes remain fixed throughout the animation. The path shows
training to date, with initial and current positions marked.

The **test-error spectrum** is a graph Fourier decomposition of the scalar
prediction residual on *test molecules only*. We construct one fixed graph
from the test molecules' RDKit Morgan fingerprints (radius 2, 2048 bits),
using Tanimoto similarity to connect each molecule to its eight closest
neighbors. If this graph is disconnected, the strongest available
between-component edges join it. The eigenvectors of its symmetric normalized
Laplacian give modes from smooth variation across similar molecules to rapid
variation. The ten bars group equal numbers of sorted eigenmodes. Each bar is
its exact contribution to test MSE, so all ten sum to test MSE. The first bar
includes the constant mode. The lower panels track all-mode, low-three-decile,
and high-three-decile error through training. The graph and eigenbasis remain
fixed for every frame and both initialization scales. The upper spectral
vertical axes adjust per frame, while the lower time-series axes stay fixed.
A few QM9 SMILES fail
modern RDKit valence sanitization; for those, fingerprint generation uses
their parseable connectivity without strict sanitization. The count is in
`spectral_summary.json`.

The **atom-activation spectrum** uses each molecule's own unweighted bond graph
and its symmetric normalized Laplacian. It projects the 32-channel `conv1`
and 16-channel `conv2` atom activations onto that molecule's eigenvectors.
The four bars show the mean per-molecule share of *nonconstant* power in the
eigenvalue intervals `0–0.5`, `0.5–1`, `1–1.5`, and `1.5–2`. The degree-weighted
constant mode is removed before these shares are calculated. Titles separately
report the nonconstant share of **total** activation power and its RMS level,
so a changing relative spectrum is not mistaken for changing amplitude.

These are exploratory graph-based frequency definitions. Unlike modular
addition, QM9 has no canonical shared Fourier axis: the test-error spectrum
depends on the chosen molecular similarity graph, and each atom spectrum uses
a different molecule's bond graph. Low eigenvalues mean smoothness on the
specified graph; they do not identify a common chemical mechanism by
themselves. The method follows the graph Fourier interpretation in
[Shuman et al. (2013)](https://arxiv.org/abs/1211.0053); the fingerprint and
similarity choices use [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html).

The compressed `analysis_total100.npz` and `analysis_total400.npz` files
contain the plotted predictions, embedding coordinates, weight PCA scores,
targets, split masks, and steps. The summary is in `summary.json`.
The `graph_spectra_total100.npz` and `graph_spectra_total400.npz` files contain
the per-frame spectral data and the similarity-graph eigenvalues; their summary
is in `spectral_summary.json`.

Regenerate from the parent directory after the archived reruns finish:

```bash
bash run_grokking_animations.sh
.venv/bin/python animate_qm9.py
.venv/bin/python animate_spectra.py
.venv/bin/python animate_qm9.py --linear-x-only
.venv/bin/python animate_spectra.py --linear-x-only
.venv/bin/python ablate_qm9.py
.venv/bin/python probe_chemistry_qm9.py
.venv/bin/python map_omnigrok_structure.py
```
