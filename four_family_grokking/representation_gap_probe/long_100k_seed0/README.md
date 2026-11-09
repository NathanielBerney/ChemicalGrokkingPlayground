# Do the matrix and molecular models need more gradient steps?

These runs use the same 7,000/3,000 molecule split and models as the [20,000-step comparison](../README.md). They train from the same seed for **100,000 steps**. The learning rate follows the original 20,000-step cosine decay and then stays at its 0.1× floor. Thus the first 20,000 steps reproduce the earlier trajectories, and the next 80,000 isolate additional training at that rate. Every full run used a separate tmux session and at most one of the four H200 GPUs.

## Main result

| Model | Train 1% accuracy, 20k → 100k | Test 1% accuracy, 20k → 100k | Test MAE, 20k → 100k (|β|) |
|---|---:|---:|---:|
| One-layer matrix rows | 29.1% → **89.0%** | 10.6% → **9.9%** | 0.0896 → **0.0988** |
| π-graph GNN | 56.7% → **99.2%** | 27.3% → **28.9%** | 0.0295 → **0.0292** |
| Full molecular graph GNN | 90.4% → **100.0%** | 17.1% → **15.9%** | 0.0462 → **0.0465** |
| SMILES transformer | 72.7% → **98.9%** | 8.4% → **7.8%** | 0.0970 → **0.1000** |

Extra steps plainly mattered for **training fit**. They did not produce delayed held-out improvement in the first 100,000 steps. The atom graph reached nearly perfect train accuracy around 45,000 steps and remained near 16% test accuracy through 100,000. The π graph has the best held-out score, but its extra 80,000 steps changed test MAE by less than 0.001|β|. The row model's test MAE worsened by 0.0093|β|; a paired bootstrap over held-out spectral groups gives a 95% interval of **+0.0068 to +0.0118|β|** ([`paired_changes.json`](paired_changes.json)). These results do not rule out a transition beyond 100,000 steps, especially for models that only reach near-perfect train accuracy at the end.

## What entering H as rows changes

The [architecture diagram](plots/row_vs_graph_architecture.png) shows the two input paths. A matrix row is a length-40 vector whose coordinates are **fixed atom indices**. A learned row position is also added. Relabeling the same molecular sites permutes both H axes but changes the row-attention model's output. Its single attention layer lets the CLS token pool input row embeddings; the token cannot read row-to-row contextualized embeddings until another attention layer is added. A Hückel frontier gap is a global spectral property, so this is a restrictive input path.

The π-graph uses the same H and occupancy information as weighted edges and site features. Six message-passing rounds and invariant pooling respect site renumbering. In the [20,000-step permutation probe](../permutation_probe_20k_seed0.json), equivalent renumberings changed row-model predictions by median 0.220|β| and π-graph predictions by less than 5×10⁻⁷|β|.

### Row controls

| Control | Final train 1% | Final test 1% | Test MAE (|β|) | Interpretation |
|---|---:|---:|---:|---|
| One layer, original LR decay | 89.0% | 9.9% | 0.0988 | Still fitting train at 100k. |
| One layer, LR decay stretched to 100k | 99.6% | 9.5% | 0.0862 | Reaches train saturation without test breakthrough. |
| Two row-attention layers, similar parameter count | 99.7% | 15.4% | 0.0742 | Fits train rapidly; modest test gain. |
| One layer, random site relabeling during training | 11.3% | 7.2% | 0.0967 | Cannot fit the augmented task under original schedule. |
| Random site relabeling, slow LR decay | 19.2% | 9.1% | 0.1022 | Higher LR helps training somewhat, still underfits. |

The two-layer model tests whether an extra row-contextualization round changes the result. It reaches 94.8% train accuracy by 14,000 steps, compared with 89.0% after 100,000 steps for the original one-layer schedule. Yet its final test accuracy is 15.4%, below the π graph's 28.9%. Its predictions still change by a median **0.174|β|** under equivalent site relabelings, and its 1% accuracy falls from 15.0% to 3.1% on the 1,000-molecule permutation sample. More depth helps but does not make row input invariant.

Random site relabeling during training reduces the one-layer model's median prediction change from **0.252|β|** to 0.010|β| (0.0065|β| with slower LR decay). Those relabeling-trained models are **underfit** at 11–19% train accuracy, so their test scores do not establish what a sufficiently expressive permutation-trained row model could achieve. The π graph remains invariant to numerical precision. See [`permutation_probe.json`](permutation_probe.json) and [`plots/site_permutation_sensitivity.png`](plots/site_permutation_sensitivity.png). The plotted row comparisons and final values come from [`../compare_row_controls.py`](../compare_row_controls.py).

## Files and reproducibility

- [`plots/extended_training_dynamics.png`](plots/extended_training_dynamics.png): 1% and 5% train/test accuracy and MAE through 100,000 steps.
- [`plots/row_controls.png`](plots/row_controls.png): row-depth, permutation augmentation, and learning-rate schedule controls.
- [`plots/site_permutation_sensitivity.png`](plots/site_permutation_sensitivity.png): relabeling probe after long training.
- [`summary.json`](summary.json): milestone scores and changes during the last 20,000 steps.
- [`paired_changes.json`](paired_changes.json): paired test changes from 20k to 100k with spectral-group bootstrap intervals.
- Run subdirectories: model configurations, full histories, fixed-step checkpoints, and per-molecule final predictions.

Regenerate the figures from the parent directory with `python analyze_long.py`, `python compare_row_controls.py`, and `python permutation_probe_long.py` using `/home/nate/anaconda3/envs/grokking/bin/python`. The runs were exploratory comparisons on one fixed grouped split; the test set was viewed during analysis, so these are not untouched confirmatory estimates for selecting a final architecture.
