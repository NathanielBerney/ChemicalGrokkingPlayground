# Omnigrok N=400: what the low-dimensional structure contains

This analysis replays archived QM9 checkpoints for α=1 and α=3 at steps 0,
10k, 20k, 40k, and 87.5k. PCA is fit on the original **200 training
molecules** at each checkpoint. The directions are then applied to the
archived 200 test molecules and to 1,000 additional QM9 molecules (indices
400–1399) outside this run's train/test universe.

## Analysis path

~~~mermaid
flowchart LR
  G["QM9 molecular graph"] --> M["Frozen model checkpoint"]
  M --> Z["16 pooled channels per molecule"]
  T["Original 200 training molecules"] --> P["Fit PCA axes and probe readouts"]
  Z --> P
  P --> I["Keep PC1+PC2 or remove PC3"]
  I --> H["Original frozen prediction head"]
  H --> E["Compare MSE on held-out and outside molecules"]
  Z --> C["Chemical labels and 19 QM9 properties"]
  C --> Q["Train-only concept/property probes"]
  Q --> V["Validate on 1,000 outside molecules"]
~~~

The PCA interventions ask which directions the existing prediction uses.
The probes ask what can be decoded from the frozen embedding. Both are
needed before assigning a possible chemical interpretation to a direction.

## Two different PCAs

| PCA | Rows being compared | α=1, N=400 | α=3, N=400 |
| --- | --- | ---: | ---: |
| Final pooled embedding | 400 molecules × 16 channels | 84.3% in PC1+PC2 | 96.1% in PC1+PC2 |
| Full weight trajectory | 89 checkpoints × 30,321 parameters | 93.6% in PC1+PC2 | 88.8% in PC1+PC2 |

The first describes molecular variation at the final checkpoint. The
second describes how model weights move over time. The
[weight-trajectory figure](weight_trajectory_structure.png) shows that over
98% of trajectory variance lies in the two graph-convolution blocks. Weight
movement by itself does not identify two chemical variables. Those blocks
also contain 29,744 of 30,321 parameters (98.1%), so their large total
variance share partly reflects their size.

## Does the model actually use only two directions?

At step 40k, training-only PCA gives:

| Model | PC1 variance | PC1+PC2 variance | Original test MSE | Keep only PC1+PC2, same head | Keep PC1+PC2+PC3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| α=1 | 75.0% | 86.1% | 0.792 | 2.744 | 1.781 |
| α=3 | 96.1% | 98.9% | 0.786 | 2.883 | 0.788 |

The [variance versus fidelity figure](pca_variance_vs_function.png) uses
linear axes. For α=3, **PC3 carries only 0.99% of training-set embedding
variance**, yet removing PC3 alone from the actual embedding raises held-out
MSE from 0.786 to 2.862. The paired MSE increase is 2.076 (molecule
bootstrap 95% interval 1.576–2.646). Keeping only the first two PCs is
similarly damaging; adding PC3 restores accuracy. The two-PC reconstruction
has no negative pooled entries on held-out molecules, so the result is not
caused by breaking the pooled ReLU nonnegativity constraint. Reconstruction
may still move activations off their usual manifold.

α=3 has five pooled channels active on any archived molecule at step 40k,
compared with eight for α=1. Its participation-ratio effective rank is
1.08, versus 1.72 for α=1. This is strong compression, while the function
still depends on a small-variance correction.

The smaller [N=100 comparison](n100_pca_fidelity_comparison.png) behaves
differently. At its final checkpoint, α=3 puts essentially all pooled
variance in two PCs; keeping those two gives test MSE 3.14 versus the
unmodified 3.23. Thus the two-PC picture is nearly sufficient for N=100,
while the more accurate N=400 model needs the small third direction.

## What do the axes correspond to?

The [embedding panels](embedding_chemistry_axes.png) color the train-fitted
PC axes by true polarizability, total atom count, and nitrile presence. At
α=3 step 40k, PC1 has held-out Pearson correlation −0.980 with
polarizability and −0.888 with atom count. The dominant axis blends the
target and a strong size correlate.

We oriented each PC using the original training molecules only, then tested
it on 1,000 additional QM9 molecules. [Single-PC concept AUCs](external_pc_chemical_concepts.png)
compare molecules with identical H/C/N/O/F counts. α=3 PC3 distinguishes
external nitriles with ordinary AUC **0.973** and composition-matched AUC
**0.987** (30 formula groups, 1,378 positive-negative pairs; group-mean AUC
0.979). PC1 also has matched nitrile AUC 0.987, so PC3 is neither unique nor
a pure nitrile unit.

The [stricter matching check](pc3_nitrile_matching_controls.png) also matches
ring, carbonyl, hydroxyl, and amine presence. PC3 retains pair-weighted
nitrile AUC 0.958 across 168 pairs from 18 matching groups (group-mean AUC
0.903). This reduces several obvious structural confounds without proving
that nitrile itself causes the activation.

The [training-stage control](nitrile_pc_axes_over_training.png) repeats the
external nitrile test for independently refitted PC1–PC3 at each checkpoint.
For α=3, the third-ranked PC has matched nitrile AUC 0.710 at initialization
and 0.987 at 40k. The earlier [full-embedding probe](../../chemistry_probes/total400/README.md)
already reads nitriles well at initialization, so this change is best viewed
as a reorganization or concentration of accessible information, not proof
that the network discovered nitrile chemistry from scratch.

The [external PC3 intervention](pc3_nitrile_external_validation.png)
tests output relevance. Removing PC3 raises polarizability MSE on the 111
external nitrile molecules from **1.56 to 23.85** (paired increase 22.29;
molecule bootstrap 95% interval 13.99–32.99). For the other 889 molecules,
MSE rises from 2.94 to 5.60. PC3 is particularly important for nitriles
and also helps other molecules. The model's overall MSE on these extra
molecules is 2.79, higher than its 0.786 MSE on the archived held-out 200.

The [matched-molecule gallery](composition_matched_nitrile_examples.png)
shows three illustrative nitrile/non-nitrile pairs with the same elemental
formula and nearly equal true polarizability. The selection rule picks the
smallest target difference in each of three distinct formulas; the gallery
is not used to estimate AUC or model accuracy.

At step 40k, α=1 and α=3 pooled representations have linear CKA 0.924 on
archived test molecules. After removing each representation's
training-fitted linear dependence on H/C/N/O/F counts, CKA is 0.218.
Their geometric similarity is therefore largely associated with
composition. CKA follows [Kornblith et al. (2019)](https://proceedings.mlr.press/v97/kornblith19a).

## Does the representation retain other QM9 properties?

The [property-transfer figure](external_qm9_property_transfer.png) shows
seven selected rows from a screen of **all 19 QM9 targets**; every row is in
[summary.json](summary.json). Ridge probes use the original 200 training
molecules and five-fold training-only penalty selection. They are evaluated
on the next 1,000 QM9 molecules. The score is
1 − external MSE / MSE of the original training-set mean, so 0 means no
improvement over that mean. The figure uses a linear color scale.

| Property | Element counts only | α=3 initial pooled + counts | α=3 step-40k pooled + counts |
| --- | ---: | ---: | ---: |
| Polarizability | 0.961 | 0.971 | 0.982 |
| HOMO energy | 0.195 | 0.656 | 0.415 |
| LUMO energy | 0.241 | 0.575 | 0.314 |
| HOMO–LUMO gap | 0.304 | 0.554 | 0.345 |
| Dipole moment | 0.106 | 0.218 | 0.096 |

Training for polarizability makes the α=3 pooled embedding slightly better
for that target while making several other properties less accessible to a
small linear readout. This supports **selective compression** more directly
than the PCA percentage alone. The initial graph representation already
encodes useful chemistry, and a lower linear score does not prove that a
nonlinear readout could not recover a property. Some rotational-constant
scores are strongly negative on this ordered external slice; those remain
visible in the complete JSON. For HOMO, the α=3 score change is −0.241
(paired molecule bootstrap 95% interval −0.291 to −0.194); for
polarizability it is +0.011 (interval +0.005 to +0.018). These intervals
condition on the one saved training run and the fitted probes.

## Which matrices are in the model?

This network has no learned atom-embedding lookup table and no
classification logits. It uses 11 atom features, edge-conditioned graph
convolutions, global sum pooling into 16 channels, a 32-channel ReLU
readout, and one scalar polarizability output.

| Figure | Rows × columns | Meaning |
| --- | --- | --- |
| [Learned parameter matrices](learned_weight_matrices_step40000.png) | Conv1 root 32×11; Conv2 root 16×32; pooled-to-hidden 32×16; output row 1×32 | Signed learned weights |
| [α=1 bond filters](edge_filters_alpha1_step40000.png), [α=3 bond filters](edge_filters_alpha3_step40000.png) | Per-bond Conv1 32×11 and Conv2 16×32 | Filters generated for single, double, and triple bonds |
| [Molecule activation matrices](activation_and_scalar_output_matrices_step40000.png) | Pooled 400×16; hidden 400×32; local sensitivity 400×16 | Activations and local output derivatives |
| [Scalar-output matrix](scalar_output_not_logits_step40000.png) | Output weights 1×32; true/α=1/α=3 values 400×3; residuals 400×2 | Actual regression values and errors in place of a “logit matrix” |

Molecule rows are sorted by **true polarizability**, so bands may reflect
target magnitude. Channel indices do not identify atom types or concepts
on their own. The sensitivity matrix is the local derivative of output with
respect to pooled channels, not an input-atom attribution.

## Interpretation and limits

There is quantifiable structure: α=3 compresses most variation into one
large target/size axis and uses a smaller third direction that is associated
with nitriles on unseen QM9 molecules and materially affects predictions.
This does **not** establish that PC3 represents only nitriles, that any
weight encodes an orbital, or that the model has learned an electronic
mechanism. The network receives graph features but no 3D coordinates or
electric-field input. The measurements use one training seed and one
ordered external QM9 slice. The [chemical probes](../../chemistry_probes/total400/README.md)
and [channel ablations](../../ablations/total400/README.md) give other views.

Regenerate from the project root with: .venv/bin/python map_omnigrok_structure.py

All measurements are in [summary.json](summary.json).
