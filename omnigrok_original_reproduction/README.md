# Omnigrok QM9 reproduction

This directory reproduces the QM9 experiment in Figure 5 of
[Liu, Michaud, and Tegmark, *Omnigrok: Grokking Beyond Algorithmic Data*](https://arxiv.org/abs/2210.01117)
using the authors' [public QM9 notebooks](https://github.com/KindXiaoming/Omnigrok/tree/main/qm9)
at commit `3a0a57346230a7255e5390fcd565812d00f1a281`.

The target is QM9 isotropic polarizability (`y[:, 1]`, in Bohr radii cubed).
The processed PyTorch Geometric dataset here has 130,831 molecules. Each run
uses the **first N dataset entries** and a seeded 50/50 random split, exactly
as in the notebooks. The model is the authors' two-layer `NNConv` network:
11 input node features → 32 → 16, global sum pooling, then 16 → 32 → 1.
Training uses batch size 32 and learning rate 0.001.

## Figure 5(d): delayed generalization

The paper caption says **200 training molecules**, which means `N=400` under
its stated 50/50 split. The released notebook's example instead sets `N=100`,
meaning 50 training molecules, and its plotted file names also say 100. Both
interpretations are run here at initialization scales `α=1` and `α=3`, with
zero weight decay. Epoch counts follow the notebook's
`int(100 * 50000 / N)` formula, giving about 100,000 optimizer steps. Seeds,
the test-batch sampling after each update, and the 20-update plot smoothing
follow the released code.

The notebook appends `MSE_loss.item() / batch.num_graphs` after each update,
although `MSE_loss` already averages over graphs. Files named
`batch20_metrics.csv` retain that plotted quantity for comparison. Files named
`full_metrics.csv` contain actual **per-molecule MSE** on the full train and
test subsets at regular checkpoints. The plots and summary label them
separately. The `config.json` for each run stores the split indices.

## Figure 5(a–c): fixed weight norm landscape

The landscape grid contains the notebook's 22 data sizes and 21 initial
weight scales. After every Adam update, all model parameters are rescaled to
their initial L2 norm. Each point trains for `int(100000 / N)` epochs. The
saved JSON records the minimum train and test losses, in both the notebook's
plotted units and true per-molecule MSE. An additional 21-point `N=3000`
slice supports Figure 5(c); the `N=100` slice is part of the grid.

## Files and rerunning

- `reproduce_qm9.py`: model, data split, training, fixed-norm grid, and
  resumable grokking checkpoints.
- `plot_results.py`: learning curves, landscape heatmaps, Figure 5(c) slices,
  and `figures/summary.json`.
- `figures/landscape_results.csv`: one row per constrained-norm run, including
  the exact `N`, weight scale, true MSE, and notebook-plotted loss.
- `runs/`: raw metrics, checkpoints, exact split indices, and grid results.
- `logs/`: worker logs.
- `data/qm9/`: downloaded and processed QM9 data.
- `data_manifest.json`: source URLs and raw-file checksums.

This host's `.venv` inherits PyTorch 2.10.0 from the existing `grokking`
environment and locally installs PyTorch Geometric 2.8.1 and RDKit 2026.9.1.
To rerun on this host from this directory:

```bash
bash run_grokking_all.sh
bash run_landscape_all.sh
.venv/bin/python plot_results.py
```

The launch scripts use four GPUs for grokking and split the landscape over
four CPU and four GPU workers. `run_landscape_all.sh cpu` and
`run_landscape_all.sh gpu` run those halves separately. The optional
`run_landscape_cpu_support.sh` works backward through the GPU half using
spare CPUs. Landscape computations skip completed points; grokking runs
resume from their latest epoch-boundary checkpoints.

## Results

The four Figure 5(d) comparison runs are complete. These are full-subset,
unscaled test MSE values read back from the saved final models:

| First N QM9 entries | Train/test count | α=1 final test MSE | α=3 final test MSE |
| ---: | ---: | ---: | ---: |
| 400 | 200/200 | 0.790 | 1.003 |
| 100 | 50/50 | 1.952 | 3.221 |

For `N=400, α=3`, test MSE drops from 96.3 near step 9,800 to 1.70 near
step 20,300, while `α=1` is already around 0.96 at step 9,800. For the
released notebook's `N=100` setup, the `α=3` test MSE is about 101 at step
10,000, 11.1 at step 20,000, and 3.45 at step 40,000; `α=1` stays around
1.6–1.8 during this interval. Thus both settings show delayed improvement
with large initialization, and the smaller notebook setup has the more
distinct train/test lag. The large-initialization model remains worse at the
end of both runs. The complete curves are in `figures/`.

All **462** main landscape grid points and **21** extra `N=3000` points are
complete. At weight scale 1.585, the best notebook-plotted train/test losses
are 0.000503 / 0.0525 for `N=100` (about 105× apart), versus 0.0727 / 0.0798
for `N=3000` (about 1.10× apart). This reproduces the small-data landscape
mismatch and the much closer large-data curves in Figure 5(c). At very large
initialization scales, both losses rise, as in the authors' plot. The largest
relative fixed-norm drift across all runs is below `2.1e-7`.

| Paper panel | Notebook-scale plot | True MSE plot |
| --- | --- | --- |
| Figure 5(a–b): landscapes | [heatmaps](figures/figure05ab_landscape_author.png) | [heatmaps](figures/figure05ab_landscape_true.png) |
| Figure 5(c): N=100 and N=3000 slices | [loss slices](figures/figure05c_landscape_cuts_author.png) | [loss slices](figures/figure05c_landscape_cuts_true.png) |
| Figure 5(d): 200 train, 200 test | [grokking curves](figures/figure05d_notebook_metric_total400.png) | [grokking curves](figures/figure05d_true_mse_total400.png) |
| Released notebook's 50 train, 50 test | [grokking curves](figures/figure05d_notebook_metric_total100.png) | [grokking curves](figures/figure05d_true_mse_total100.png) |

The qualitative results reproduce the paper's delayed improvement and
small-data loss mismatch. Exact curve values can differ because this run
uses a newer PyTorch Geometric and RDKit stack than the 2023 notebooks.

## Training animations

The [animation gallery](animations/README.md) compares the `α=1` and `α=3`
models through training for both dataset sizes. It shows prediction versus
target, pooled graph embeddings, and a PCA of the full model weight
trajectory. It also shows graph Fourier spectra for test prediction error
across similar molecules and atom activations across each molecule's bonds.
`run_grokking_animations.sh` repeats the four conditions with archived
weights every 1,000 steps; `animate_qm9.py` and `animate_spectra.py` render
the GIFs.
