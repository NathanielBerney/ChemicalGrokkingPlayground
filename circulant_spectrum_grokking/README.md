# Circulant matrix → full eigenvalue spectrum

This experiment asks whether a transformer trained on raw matrices develops
Fourier representations and uses them to generalize to unseen matrices.
It extends the earlier HOMO/LUMO circulant experiment to all 42 eigenvalues.

Open **[the animation gallery](animations/index.html)** for the learning curves,
embedding animations, frequency diagnostics, and causal interventions.
[The Markdown gallery](animations/README.md) links individual GIFs.
Machine-readable results are in `results.json` and each run's
`analysis/summary.json`.
Read [the first-run findings](RESULTS.md) for the quantitative interpretation.

## Tasks and controls

Both tasks receive exactly the same full `42 × 42` matrix, represented as
42 row tokens. No FFT, eigenvectors, spectral features, or trigonometric
positional encodings enter the model.

- **Main task (`sorted`)**: predict all eigenvalues in ascending energy order.
  Training labels come from the stored eigensolver results.
- **Control (`mode`)**: predict eigenvalues in Fourier-mode order, `k=0,...,41`.
  Labels are the real DFT of the first row. This supplies a consistent mode
  alignment in the supervision, making the underlying map linear.
- **Learned linear controls**: least-squares regression on the raw first row
  plus an intercept, fitted separately to each target on the training split.
  The first row is sufficient information for these canonical circulants;
  using it avoids the redundant columns in a flattened full matrix.
- **Exact mathematical audit**: compare the sorted DFT eigenvalues against
  the stored eigensolver labels. This verifies the task, and is not a trained
  competitor or a feature supplied to the transformer.

For first row `c`, the mode-indexed eigenvalues satisfy

```
lambda[k] = sum_r c[r] exp(-2 pi i k r / n).
```

For a real symmetric circulant, conjugate terms cancel and eigenvalues are
real. A uniform nearest-neighbor ring gives
`lambda[k] = alpha + 2 beta cos(2 pi k/n)`.
For this even-sized dataset, the opposite-site offset `r=21` occurs once;
its contribution is `c[21] (-1)^k`, without a factor of two.

Mode-ordered regression is linear, so it can generalize without a delayed
grokking transition. Successful mode prediction does not establish learning
a general eigensolver. The sorted task additionally requires energy ordering.

## Data and split

- Source: `../dataset/circulant_n42.npz`, unchanged.
- 10,000 distinct symmetric circulant matrices, `n=42`, diagonal zero,
  and off-diagonal values in `{0,-1}`.
- The dataset contains **8,509 distinct spectra** at seven-decimal precision.
  Distinct matrices can share a spectrum. These are grouped before splitting,
  so a spectrum group appears entirely in train or test.
- Approximately 30% of matrices train, 70% test. Seeds 0 and 1 define independent
  group splits. Each seed uses the same split and initialization across targets.
- One global target mean and standard deviation are computed from training
  eigenvalues and shared across target orderings. Reported RMSE is in `|beta|`
  units and includes every output and every matrix in the relevant split.
- The dataset hash, exact indices, and group assignments are saved with each run.

These are mathematical offset-pattern circulants. They do not enforce carbon
valence or graph planarity. Held-out performance tests new coupling patterns at
fixed size and canonical site order. It does not establish transfer to other
sizes, arbitrary atom relabelings, or noncirculant molecules.

## Model and training

- Learned row projection: 42 → 128; learned CLS token and absolute positions.
- Two transformer blocks, four attention heads each, width 128, ReLU MLP 512.
- Residual attention and MLP connections; no LayerNorm and no dropout.
- CLS readout: 128 → 42 continuous eigenvalues.
- AdamW, learning rate `0.001`, batch size 256 sampled with replacement,
  40,000 optimizer steps, float32 with highest matmul precision.
- Initial suite: both tasks × two seeds, weight decay `1.0`.
- Follow-up: sorted targets × two seeds, weight decay `0.1`, added after the
  initial sorted runs showed poor training fit. These are explicitly follow-up
  runs, not additional independent confirmations chosen without observing data.
- Full train/test metrics every 200 steps. No test-based early stopping or
  checkpoint selection; reported final results use step 40,000.
- 87 archived checkpoints: initialization, steps 1/10/25/50/100/200, and every
  500 steps through 40,000. `latest.pt` also stores optimizer and sampling RNG
  state for continuation.

The model is inspired by the small interpretable transformers used in modular
arithmetic work, but this is continuous regression with matrix tokens and
mini-batches. It is not an exact Nanda reproduction.

## Reading the animations

Each run has three GIFs and a high-resolution PNG of each final frame:

1. **`representations.gif`**: input-column weight embeddings, final row-token
   representations of a fixed held-out matrix, CLS embeddings of 128 fixed
   held-out matrices, and predicted versus true spectra for three examples.
   PCA axes are fitted jointly across checkpoints and then fixed. This is a
   post-hoc visualization; axis rotation between frames cannot create motion.
   Explained variance is shown, and two PCs need not expose all Fourier modes.
2. **`frequencies.gif`**: full-set errors; Fourier power of input weights and
   hidden row activations across the cyclic **site** axis; aggregate and
   per-channel Gini; non-DC weight power fraction; and exact versus learned
   sensitivity to changes in symmetric coupling coefficients.
   Frequency powers include the multiplicity of conjugate pairs, consistent
   with Parseval's identity. DC is excluded from Gini and channel selectivity.
   Channels are displayed in a fixed ordering based on final dominant mode.
3. **`causal_ablations.gif`**: remove a real Fourier site subspace before each
   attention block, preserve CLS, and re-evaluate the same 128 held-out examples.
   Frequencies 1–20 remove a sine/cosine pair (rank two); frequency 21 removes
   the Nyquist direction (rank one). Random controls use three fixed
   DC-orthogonal subspaces per rank. They match rank, not removed activation
   energy. The heatmaps show which predicted eigenvalues change.

The final row activations are displayed but not ablated after the final block:
at that point they no longer feed the CLS readout. Interventions occur before
attention, where row information can influence the output.

Coupling sensitivities use central finite differences with epsilon `0.001`
on four fixed test matrices. Perturbations preserve symmetry and circulant
structure. This tests local continuous behavior adjacent to binary training
couplings. For sorted spectra, central differences can average slopes at
eigenvalue crossings; the exact comparison uses the same perturbations and
sorting, rather than assigning an arbitrary derivative at a crossing.

For mode-ordered runs, the summary also reports the mean fraction of ablation
prediction-change energy in the expected output pair `k, n-k`, separately for
each intervention stage. This is averaged over removed frequencies, including
the single Nyquist output. Compare it at initialization and the final checkpoint.

Fourier structure exists in the inputs before training. Spectral plots alone
do not prove a learned mechanism. Gini can rise while non-DC weights vanish;
the power fraction is displayed to reveal this possibility. Full-spectrum
prediction may use many modes, so global sparsity is not a necessary success
criterion. Causal selectivity supports use of frequency-specific information,
but does not by itself identify a unique complete internal algorithm.

## Reproduce or continue

The existing environment is `/home/nate/anaconda3/envs/grokking/bin/python`.
Dependencies are NumPy, PyTorch, Matplotlib, and Pillow. Run from this directory.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
PYTHON=/home/nate/anaconda3/envs/grokking/bin/python

# Validate labels, split isolation, Fourier projectors, the learned linear
# solution, and model/intervention forward-backward behavior.
"$PYTHON" validate.py

# First suite: four simultaneous GPUs, then diagnostics and animations.
# Existing run directories are protected against accidental overwrite.
bash run_suite.sh all

# Optional lower-decay follow-up: two GPUs (0 and 2).
bash run_lower_decay.sh
CUDA_VISIBLE_DEVICES=0 "$PYTHON" analyze.py --run runs/sorted_seed0_wd0p1
CUDA_VISIBLE_DEVICES=2 "$PYTHON" analyze.py --run runs/sorted_seed1_wd0p1
"$PYTHON" gallery.py
"$PYTHON" verify_artifacts.py

# Regenerate diagnostics/animations for the initial four completed runs.
bash run_suite.sh analyze

# Continue one existing run with matching original settings.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" train.py --target sorted --seed 0 \
  --outdir runs/sorted_seed0 --steps 80000 --resume
```

Check `logs/` for each job's progress. A training run is finished when its
`complete.json` exists and its step matches the current config. Use `analyze.py`
after a run has finished; `--no-render` saves only numerical diagnostics.
`gallery.py` includes every analyzed directory under `runs/`.
The separate 500-step `pilot_sorted_seed0/` directory is a smoke test and is
excluded from the main results gallery.

`verification.json` records checks of all six final checkpoints, full-test
metrics, diagnostic arrays, gallery links, and all 18 GIFs (87 frames each).
A separate interrupted/resumed training check matched uninterrupted training
exactly for both model parameters and the minibatch sampling RNG.

Files: `experiment.py` contains data/model utilities; `train.py` trains;
`analyze.py` extracts checkpoint diagnostics and baselines; `render.py` makes
GIFs; `gallery.py` creates the comparison and local HTML gallery.

## References

- [Nanda et al., Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217).
- [Charton, Linear algebra with transformers](https://arxiv.org/abs/2112.01898).
- [Gray, Toeplitz and Circulant Matrices: A Review, Chapter 3](https://ee.stanford.edu/~gray/toeplitz.pdf).
