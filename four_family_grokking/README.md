# Four-family grokking probes

This directory tests the three graph-distinct Hückel datasets and the
pericyclic transition-state matrix dataset in `OPSandbox`. Each contains
10,000 samples. The initial four runs use seed 0, a random 30% training
split, one-layer/four-head/d=128/ReLU-MLP=512 transformer with no LayerNorm,
full-batch AdamW (learning rate 0.001, weight decay 1), 40,000 steps, and
evaluation every 200 steps. One tmux session and one H200 were used per
family, on GPUs 0–3.

## Inputs and accuracy

The circulant, Möbius, and linear datasets have one distinct graph per size
`n`, with fixed within-family coupling and graph rule. The largest matrices
would be 20,000–40,000 square, so these experiments feed the **16-bit value
of n** to the transformer. This is a lossless description of the matrix
*within a family*, but it does not test a model reading a dense Hückel
matrix or discovering its Fourier eigenvectors from orbital entries. The
target is the HOMO and LUMO energy pair. The pericyclic model receives its
actual padded 8×8 transition-state matrix and active-site mask, and predicts
the binary thermal-allowed label.

For energy regression, `train_accuracy` and `test_accuracy` mean **both**
frontier energies have absolute error below `0.05 |β|`, as requested. The
curves also show the stricter condition that **both** have relative error
below 5%, along with RMSE. This distinction matters: most frontier energies
approach zero as n grows, so the predictor `(HOMO,LUMO)=(0,0)` already gets
about **99.7%** on the absolute accuracy metric while getting **0%** relative
accuracy. The plots show that baseline explicitly.

`runs/` contains the raw-energy target runs. `runs_log/` contains three
additional energy controls with the same architecture and split, trained on
the log magnitudes of HOMO and LUMO. That loss treats small and large
molecules more evenly; predicted signs are fixed by the all-carbon
closed-shell model. The controls are separate runs so the original results
remain reviewable.

## Results at step 40,000

| Dataset | Test accuracy, abs 0.05 | Zero baseline | Test accuracy, rel 5% | Test RMSE | Zero RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circulant | 99.76% | 99.67% | 24.10% | 0.00895 | 0.01541 |
| Möbius | 99.97% | 99.70% | 29.27% | 0.00175 | 0.00851 |
| Linear | 99.89% | 99.69% | 17.89% | 0.00318 | 0.01118 |
| Pericyclic | 96.30% | 50.33% majority | — | binary task | — |

The raw-energy runs reached 99% absolute train **and** test accuracy at
step 1 because of the vanishing-target baseline. Their RMSE does improve
over the zero predictor, so they learned some size dependence. Their 5%
relative train and test accuracy stayed low and moved together; no delayed
grokking transition appeared. The pericyclic model reached 99% train accuracy at step 400, but
test accuracy rose early to about 96% and then plateaued. The exact
Hückel/Möbius loop-sign rule scores 100% on its held-out set.

The log-magnitude controls reached 99% relative train/test accuracy at the
same recorded step for circulant (2,400) and Möbius (2,000). Linear reached
99% on train at step 3,000 and test at 8,000. All three train and test curves
mostly track one another, with substantial later optimization oscillations;
the final relative test accuracies are 66.0%, 92.4%, and 94.0%, respectively.
The comparison plot keeps the full trajectories visible rather than selecting
the best test checkpoint.

### Stricter frontier-energy accuracy

The original `0.05 |β|` cutoff is permissive for large rings and chains.
`strict_frontier_final.py` therefore re-scores the final checkpoints using
the maximum relative error across HOMO and LUMO; **both** energies must be
within 1% or 5%. At the final checkpoint, held-out accuracy is:

| Family | Raw target, 1% | Log target, 1% | Raw target, 5% | Log target, 5% |
| --- | ---: | ---: | ---: | ---: |
| Circulant | 4.67% | 0.36% | 24.10% | 66.03% |
| Möbius | 6.33% | 4.07% | 29.27% | 92.43% |
| Linear | 3.20% | 5.13% | 17.89% | 94.04% |

The log-target runs oscillate, so these numbers describe their **final**
checkpoints rather than their best transient values. The exact figures and
small-size breakdowns are in `strict_frontier_final.json`; the original
step-by-step CSV files recorded 5% relative accuracy but not 1%.

At step 40,000, pericyclic error was concentrated in the `[4+4]`
cycloaddition template (80.0% held-out accuracy) and Alder–ene (92.6%).

### Pericyclic continuation to step 120,000

The pericyclic run continued from its 40,000-step checkpoint in the tmux
session `grok_pericyclic_extend_gpu3`. The 40,000-step checkpoint, metrics,
summary, and plot are retained with `_40000` suffixes. At step 120,000,
training accuracy was 100% and test accuracy was **96.04%**. The best test
accuracy over all 120,000 steps remained **96.47%**, first reached before
step 40,000. There was no late test-accuracy jump. The final test loss was
0.216, compared with 0.128 at step 40,000.

On the final checkpoint, `[4+4]` held-out accuracy was 78.18% and Alder–ene
was 92.21%; the other reaction families were at least 98.4%. Re-gauging and
cyclically reordering the held-out matrices gave 96.16% and 96.22% mean
accuracy respectively, but changed about 6.0% and 5.4% of individual
predictions. The learned rule remains approximate and lacks exact orbital
gauge invariance.

## Figures and reproducibility

- `plots/accuracy_dashboard.png`: all four train/test accuracy curves,
  including the pericyclic continuation.
- `plots/loss_dashboard.png`: all four train/test loss curves.
- `plots/<family>_learning_curves.png`: separate curves, including relative
  accuracy and RMSE for the energy tasks.
- `plots/energy_objective_comparison.png`: raw versus log target, with
  train/test 5% relative accuracy.
- `plots/strict_frontier_accuracy.png`: final-checkpoint 1% and 5% relative
  held-out accuracy for all six energy models.
- `plots/energy_size_diagnostics.png`: predicted frontier energies and
  relative error versus n for the raw-energy models.
- `plots/energy_size_diagnostics_log.png`: the same size diagnostics for the
  log-magnitude controls.
- `plots/pericyclic_generalization.png`: accuracy by reaction family and on
  symmetry-equivalent matrix descriptions.
- `fourier_bases/*.png`: periodic circulant, anti-periodic Möbius, open-chain
  sine, and pericyclic cyclic-reference bases. `basis_validation.json` records
  numerical checks.
- `summary.json`, `summary_log.json`, `diagnostics.json`: machine-readable
  final results. `summary_40000.json` and
  `runs/pericyclic/metrics_40000.csv` preserve the original endpoint. Each
  run also has `metrics.csv`, `config.json`, and checkpoints.

The first three bases exactly diagonalize the uniform Hamiltonians: the
largest off-diagonal entry of Q†HQ is below `1.3e-14`, and the analytic
eigenvalues match `eigvalsh` to below `5e-15`. The pericyclic matrices have
heterogeneous couplings and site energies; their gauge-aligned periodic or
anti-periodic modes are **reference bases**, not exact eigenvectors.

Start the original four sessions with `bash launch_all_tmux.sh`, and the
energy controls with `bash launch_log_tmux.sh`. Sessions are named
`grok_<family>_gpu<index>` and `grok_<family>_log_gpu<index>`. Attach with
`tmux attach -t SESSION_NAME`, or stop a session with
`tmux kill-session -t SESSION_NAME`. Checkpoints are saved every 2,000
steps; `train_family.py --resume PATH` supports continuation with the same
family, seed, and output directory. All original runs, energy controls, and
the pericyclic continuation have finished.

Refresh the figures with:

```bash
/home/nate/anaconda3/envs/grokking/bin/python analyze_runs.py
/home/nate/anaconda3/envs/grokking/bin/python analyze_runs.py --run-root runs_log --plots plots_log --families circulant mobius linear
/home/nate/anaconda3/envs/grokking/bin/python compare_energy_objectives.py
/home/nate/anaconda3/envs/grokking/bin/python plot_fourier_bases.py
CUDA_VISIBLE_DEVICES=0 /home/nate/anaconda3/envs/grokking/bin/python diagnose_final.py
```

The original n=42 circulant *matrix-input* experiment is in the parent
directory. This directory uses the newer unrestricted-size, graph-distinct
families. These are synthetic Hückel and formal pericyclic rule benchmarks;
the results do not establish chemical reaction feasibility or stable
three-dimensional molecular geometry.

## Circulant gap follow-up

The stricter HOMO–LUMO gap study, its seven neural runs, training-only
log-linear baseline, and Nanda-versus-gap architecture diagram are in
[`circulant_gap_probe/`](circulant_gap_probe/README.md). The directory is
also indexed from the project root in
[`CIRCULANT_GAP_RUNS_README.md`](../CIRCULANT_GAP_RUNS_README.md).

Final-checkpoint PCA views across all four chemical families are in
[`pca_final_families/`](pca_final_families/README.md). The pericyclic
classifier's label direction is largely in PC7, outside its top-variance
PCA plane.

The 10,000-molecule dense-Hückel gap comparison across matrix attention,
π-graph and full-atom GNNs, and SMILES transformers is in
[`representation_gap_probe/`](representation_gap_probe/README.md). It includes
strict train/test accuracy curves, two initializations, PCA, and a
site-renumbering invariance probe.

## Training animations

The [animation gallery](animations/README.md) follows embedding geometry and
frequency power through archived reruns of all four families. It includes
both raw- and log-target energy controls. For the first three families, the
frequency axis is an orthonormal cosine transform across ordered molecule
sizes, applied to outputs and readout states; this is distinct from the
physical ring-site Hückel basis. For pericyclic matrices, site-token activity
is projected onto the existing gauged cyclic reference bases. The gallery
explains both constructions and links to the GIFs and numerical arrays.
