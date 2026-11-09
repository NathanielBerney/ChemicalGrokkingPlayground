# Omnigrok: non-frequency structure checks

The pasted Hückel-transformer discussion suggested layerwise probes, graph
closed-walk moments, node-relabeling checks, and structured size tests. This
report adapts those ideas to the archived Omnigrok QM9 reproduction. That
model is a **two-layer graph network trained to predict one scalar,
isotropic polarizability**. It does not ingest Hückel matrices or have
attention heads, token embeddings, or classification logits.

## Data and measurements

The original run uses the first 400 QM9 molecules, split into 200 training
and 200 held-out molecules. This analysis fits every ridge probe and its
five-fold penalty choice on those 200 training molecules only. The next
1,000 QM9 molecules, indices 400–1399, are used for outside evaluation.
Frozen α=1 and α=3 checkpoints are examined at steps 0, 20k, 40k, and
87.5k. No model was retrained.

For each molecule, the probes read:

1. The sum of 11 input atom features.
2. The sum of 32 atom channels after graph convolution one.
3. The first 16 training-fitted PCs of that 32-channel sum, as a
   dimension-matched control.
4. The sum of 16 atom channels after graph convolution two.
5. The 32-channel final ReLU readout.

Each probe also receives H/C/N/O/F counts. Its outside score is
**1 − outside MSE / MSE of the original training-set mean**. Thus 0 means
no improvement over that mean. The [layerwise figure](layerwise_property_probes.png)
uses linear axes. Its four targets are actual QM9 polarizability, HOMO,
LUMO, and HOMO–LUMO gap values, not eigenvalues of a Hückel input matrix.
[PyG's QM9 target list](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.datasets.QM9.html)
gives their names and units.

## Result: selective access between graph layers

At α=3, step 40k:

| Target | Conv1, 32 channels | Conv1, first 16 PCs | Conv2, 16 channels | Final readout |
| --- | ---: | ---: | ---: | ---: |
| Polarizability | 0.984 | 0.980 | 0.982 | 0.980 |
| HOMO energy | 0.600 | 0.560 | 0.415 | 0.349 |
| LUMO energy | 0.748 | 0.730 | 0.314 | 0.366 |
| HOMO–LUMO gap | 0.521 | 0.584 | 0.345 | 0.482 |

The LUMO score advantage of Conv1-PCA16 over Conv2 is **0.416**, with a
paired outside-molecule bootstrap 95% interval **0.354–0.481**. Matching the
channel count therefore does not explain the drop by itself. The α=1
step-40k control is less severe: LUMO scores are 0.678 for Conv1-PCA16
and 0.615 for Conv2. At α=3 initialization, Conv2's LUMO score was 0.575.
The second convolution becomes more specialized during polarizability
training in this saved run.
The [checkpoint progression](probe_trajectory_and_prediction_error.png)
shows the LUMO probe decline at Conv2 while the original held-out
polarizability MSE falls from 5.53 at 20k to 0.786 at 40k. Both axes are
linear; the very large step-zero prediction MSE is omitted from the
right panel only.

This is a **linear-accessibility** result. It does not prove that Conv2
contains no nonlinear information about LUMO, or that the model's
polarizability prediction causally uses LUMO energy.

## Graph-walk moments

For each molecule, A is the unweighted heavy-atom bond adjacency, with
bond orders ignored. We probe tr(A²), tr(A³), and tr(A⁴): closed walks of
length 2, 3, and 4. In a simple undirected graph, tr(A²) is twice the
number of heavy-atom bonds and tr(A³) is six times the number of
triangles. These are graph-topology diagnostics, not molecular orbital
energies. The [graph-walk figure](graph_walk_moment_probes.png) uses linear
axes.

At α=3 step 40k, tr(A³) scores 0.467 from Conv1-PCA16 and 0.195 from
Conv2. Their score difference is 0.272 (paired outside-molecule bootstrap
95% interval 0.218–0.328). tr(A⁴) likewise falls from 0.917 to 0.689,
while polarizability stays near 0.98. The original graph network evidently
retains less *linearly readable* general topology after the second layer,
while retaining its target signal.

## Size generalization and permutation

Every original training molecule has at most six heavy atoms. In the
outside slice, 383 molecules have six heavy atoms, **616 have seven**, and
one has eight. The [size figure](size_generalization_mse.png) compares
the saved model's raw polarizability MSE with a ridge baseline fitted to
element counts on the original training molecules:

| Evaluation group | Molecules | α=1 step 40k MSE | α=3 step 40k MSE | Element-count baseline MSE |
| --- | ---: | ---: | ---: | ---: |
| Original held-out, ≤6 heavy atoms | 200 | 0.792 | 0.786 | 1.508 |
| Outside, 6 heavy atoms | 383 | 1.395 | 2.493 | 7.084 |
| Outside, 7 heavy atoms | 616 | 3.028 | 2.968 | 5.693 |

Both saved models predict the unseen seven-heavy-atom group better than
this element-count baseline, although their error is higher than on the
original held-out group. This is evidence of some size transfer, but the
outside slice is ordered and was not a controlled size-holdout training
experiment. At the final α=3 checkpoint, outside six-heavy-atom MSE rises
to 6.969, so checkpoint choice matters.

For 50 outside molecules per step-40k model, randomly permuting atom
indices and updating bond indices changes predictions by at most
**0.000012** for α=1 and **0.000092** for α=3. The small differences are
floating-point summation effects. Permutation invariance is expected from
the graph architecture; this is a sanity check, not evidence of learned
chemistry.

## Which suggestions fit this model?

- **Already done before this report:** pooled-channel and PCA subspace
  ablations, chemistry-concept probes, other-QM9-target probes, effective
  rank, and full-weight trajectory PCA. See the
  [structure map](../../structure_map/total400/README.md),
  [channel ablations](../../ablations/total400/README.md), and
  [chemistry probes](../../chemistry_probes/total400/README.md).
- **New here:** probes at both graph layers, dimension-matched Conv1
  control, graph closed-walk probes, a true larger-graph evaluation group,
  and atom-index permutation checks.
- **Not directly applicable:** Hückel eigenvalue perturbation derivatives,
  exact ring/chain cosine sweeps, and attention-head analysis. This model
  receives atom/bond graph features and predicts QM9 polarizability; it
  receives no Hückel matrix and has no attention heads.

The layer-probe approach follows the question in
[Akhondzadeh et al., *Probing Graph Representations*](https://proceedings.mlr.press/v206/akhondzadeh23a.html).
The probes have only 200 fitting molecules and one archived seed. Their
bootstrap intervals resample the outside molecules while holding model
weights and fitted probes fixed. The unweighted walk moments ignore
element identity and bond order. These checks locate accessible signals;
causal claims would require targeted interventions and independent runs.

Regenerate from the project root with:

    .venv/bin/python probe_omnigrok_layers.py

All numbers, checkpoints, and probe penalties are in [summary.json](summary.json).
