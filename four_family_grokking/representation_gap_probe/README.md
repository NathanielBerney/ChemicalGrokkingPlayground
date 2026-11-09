# Molecular representation gap probe

This directory compares how models learn a Hückel HOMO–LUMO gap from the **same 10,000 distinct molecules** when given a padded Hamiltonian, its π graph, a full atom graph, or canonical SMILES. It is the dense-matrix companion to [`../circulant_gap_probe/`](../circulant_gap_probe/README.md), whose 10,000 very large circulants were encoded by size rather than materialized as dense matrices.

For the actual code and attribution record behind these encoders, see [`PROVENANCE.md`](PROVENANCE.md). The graph and SMILES models here are local implementations; the repository does not record a paper or pretrained model as the source of either exact architecture.

## Dataset and split

- Source: [`../../heteroatom_huckel_dataset/`](../../heteroatom_huckel_dataset/), with 10,000 unique canonical SMILES, SDF atom/bond graphs, 5–40 π sites, and 40×40 padded Hückel matrices. Heteroatom parameters are illustrative, in units of |β_CC|; these are **model-computed gaps**, not experimental gaps.
- [`prepare.py`](prepare.py) joins SDF atom indices to the π-site ordering, tokenizes the existing canonical SMILES, and writes [`features.npz`](features.npz).
- One fixed 7,000/3,000 train/test split is grouped by rounded Hückel eigenvalue spectrum **and** π-electron count. Its 8,317 groups keep the 1,683 duplicate π spectra out of opposite sides of the split. The split is in [`split.json`](split.json).
- All models use standardized log-gap MSE, AdamW, batch 128, 20,000 gradient steps, cosine learning-rate decay, and the same seed-specific initialization schedule. We report the fixed final checkpoint. A prediction is accurate at **1%** when `abs(predicted − true) / true < 0.01`; 5% is a secondary view. All train/test curves also include MAE in |β_CC|.

## Inputs and architectures

| Run name | Input | Architecture | Trainable parameters |
|---|---|---|---:|
| `matrix_rows` | 40×40 H, per-site π electrons, total π electrons and site count | One-layer row-attention transformer adapted from the earlier matrix model | 159,617 |
| `pi_graph` | **Exactly the H and occupancy information of `matrix_rows`** as a weighted π-site graph | Six-layer permutation-invariant message-passing network | 154,305 |
| `molecular_graph` | Full SDF atom/bond graph with π-site classes and H edge couplings | Four-layer atom-graph message-passing network | 228,897 |
| `smiles` | Canonical SMILES tokens plus total π electrons and site count | Two-layer token transformer | 188,001 |
| `molecular_graph_raw` | SDF atom elements, formal charge, bond order, degree, atom count | Same atom-graph network, with H/π annotations removed | 228,897 |
| `smiles_raw` | Canonical SMILES tokens only | Same token transformer, with π metadata removed | 188,001 |

### Precomputed-eigenvalue controls

| Run name | What the neural network receives | Interpretation |
|---|---|---|
| `matrix_spectral` | The **precomputed full eigenvalue spectrum** plus HOMO/LUMO position markers | Tests a flat MLP on an already diagonalized H; it does not learn diagonalization. |
| `matrix_frontier` | The **precomputed HOMO and LUMO eigenvalues**, selected before the neural network | A near-oracle readout: the target is their difference. Its 99.8% score is not evidence that a model learned matrix spectral structure. |

Both spectral controls diagonalize H outside the network, using the stored eigenvalues derived from H. The exact eigensolver control reaches 100% 1% accuracy and ~2×10⁻⁸|β| MAE ([`baselines.json`](baselines.json)). The fair architecture comparison is `matrix_rows` versus `pi_graph`: nearly matched parameter counts and the same Hamiltonian/occupancy information.

The frontier control's steep 1% accuracy curve is **not a delayed grokking transition**. Its train and test curves cross 50% at the same step (2,000), 90% at the same step (3,500), 95% at the same step (4,750), and 99% at the same step (7,750). The largest train/test 1% accuracy separation in the logged run is 4.4 percentage points. Continuous MAE falls throughout this interval; the strict 1% cutoff makes the improvement look more sudden as predictions cross that threshold. The network is trained on standardized `log(gap)`, so its remaining learned task is a nonlinear readout of two supplied orbital energies, rather than literal subtraction in raw energy space.

## Main results

Two initializations, same grouped split, final checkpoint at 20,000 steps:

| Model | Test 1% accuracy, seed 0 / 1 | Test 5% accuracy, seed 0 / 1 | Test MAE, seed 0 / 1 (|β|) |
|---|---:|---:|---:|
| Matrix row attention | 10.6% / 11.4% | 41.5% / 43.4% | 0.0896 / 0.0884 |
| π-graph GNN | 27.3% / 34.3% | 79.7% / 80.4% | 0.0295 / 0.0271 |
| Full molecular graph GNN + π annotations | 17.1% / 18.4% | 64.3% / 66.8% | 0.0462 / 0.0431 |
| SMILES transformer + π metadata | 8.4% / 8.3% | 37.4% / 38.7% | 0.0970 / 0.0955 |
| Atom graph only | 13.8% / 15.2% | 59.1% / 57.8% | 0.0534 / 0.0534 |
| SMILES only | 8.4% / 8.0% | 37.4% / 36.9% | 0.0982 / 0.0999 |

Separate precomputed-eigenvalue controls: flat spectral MLP 7.3% test 1% and 0.0958|β| MAE; HOMO/LUMO-energy readout 99.8% and 0.00047|β| MAE. The constant log-gap control is 2.9% and 0.282|β| MAE. See [`spectral_controls_20k_seed0.json`](spectral_controls_20k_seed0.json), [`summary_20k_seed0.json`](summary_20k_seed0.json), and [`seed_comparison.json`](seed_comparison.json) for unrounded values.

The graph models learn useful held-out structure early. At 1,500 steps, the π-graph passes 50% test 5% accuracy; the full atom graph does so at 2,000 steps. The row-attention and SMILES models keep lowering training error while their held-out accuracy plateaus or worsens. There is no delayed test-accuracy jump in 20,000 steps like the modular-arithmetic grokking transition; longer runs could behave differently.

The structure-only atom graph still learns from element and bond connectivity, but supplying the π/Hückel annotations improves its two-seed mean test 1% accuracy from **14.5% to 17.8%** and mean test MAE from **0.0534 to 0.0447|β|**. The corresponding SMILES metadata changes test 1% accuracy by only **0.2 percentage points** (8.2% to 8.3%). See [`structure_only_controls.json`](structure_only_controls.json) for all values.

### Representation and symmetry probes

- [`plots_20k_seed0/training_dynamics.png`](plots_20k_seed0/training_dynamics.png): strict 1% and 5% train/test accuracy plus MAE across the four primary representation models.
- [`plots_20k_seed0/precomputed_eigenvalue_controls.png`](plots_20k_seed0/precomputed_eigenvalue_controls.png): the two already-diagonalized orbital-energy controls, kept separate from the input-representation comparison.
- [`plots_20k_seed0/heldout_predictions.png`](plots_20k_seed0/heldout_predictions.png): final held-out predictions against the Hückel labels.
- [`plots_20k_seed0/hidden_pca.png`](plots_20k_seed0/hidden_pca.png): hidden-state PCA, colored by gap. A fixed ridge probe on the learned hidden vectors predicts held-out log gap with R² 0.83 for row attention, 0.98 for the π graph, 0.96 for the full atom graph, and 0.83 for SMILES. PCA is descriptive; it does not prove a particular mechanism.
- [`permutation_probe_20k_seed0.json`](permutation_probe_20k_seed0.json): on 1,000 test molecules, three fresh site/atom renumberings each changed the row-attention gap prediction by a **median 0.220|β|** and dropped its 1% accuracy from 10.0% to 2.1%. The two GNNs changed by at most 4.8×10⁻⁷|β|, as expected from permutation-invariant graph pooling. This is a concrete architectural failure of the row-input encoding.
- [`plots_20k_seed0/seed_comparison.png`](plots_20k_seed0/seed_comparison.png): two-initialization learning curves, with the range shaded.
- [`plots_20k_seed0/structure_only_controls.png`](plots_20k_seed0/structure_only_controls.png): graph and SMILES learning dynamics with and without supplied π/Hückel annotations.

Unlike the original circulant benchmark, heterogeneous molecules have no shared cyclic site index, so a Fourier transform of row-weight columns would depend on arbitrary atom numbering. We use the Hamiltonian's **eigenbasis** as the physically relevant spectral representation and use PCA to inspect hidden states. Earlier Fourier/Gini analysis on the true circulant gap model is in [`../circulant_gap_probe/fourier_progress/`](../circulant_gap_probe/fourier_progress/README.md).

## Reproduce and stop runs

```bash
cd /mnt/das_storage/nates_workspace/OPGrokkingDatasets/OPSandbox/four_family_grokking/representation_gap_probe
/home/nate/anaconda3/envs/grokking/bin/python prepare.py
tmux new-session -d -s rep_gap20k_pi_graph_gpu1 'bash run_one_tmux.sh pi_graph 1 20000 runs_20k_seed0 0'
tmux kill-session -t rep_gap20k_pi_graph_gpu1  # stop an active run
OPENBLAS_NUM_THREADS=1 /home/nate/anaconda3/envs/grokking/bin/python analyze.py
/home/nate/anaconda3/envs/grokking/bin/python permutation_probe.py
/home/nate/anaconda3/envs/grokking/bin/python compare_seeds.py
/home/nate/anaconda3/envs/grokking/bin/python compare_raw_controls.py
```

The complete primary runs were launched in separate named tmux sessions on GPUs 0–3. Their final models, logs, checkpoints, per-molecule predictions, and hidden vectors are in [`runs_20k_seed0/`](runs_20k_seed0/) and [`runs_20k_seed1/`](runs_20k_seed1/). The initial 2,000-step smoke/pilot runs are in [`runs_seed0/`](runs_seed0/). Sessions close automatically after training finishes. The generated molecules were sanitized by RDKit during dataset creation, but the 2D structures do not establish 3D planarity, stability, or synthetic accessibility.

The follow-up testing whether 20,000 gradient steps were enough, including 100,000-step runs, row-depth and site-permutation controls, is in [`long_100k_seed0/`](long_100k_seed0/README.md).
