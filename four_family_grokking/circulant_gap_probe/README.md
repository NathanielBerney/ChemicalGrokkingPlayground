# Circulant HOMO–LUMO gap probe

This directory contains the new circulant gap runs. The root-level
[`CIRCULANT_GAP_RUNS_README.md`](../../CIRCULANT_GAP_RUNS_README.md) points
here. The source data are
[`constrained_huckel_datasets/circulant/dataset.npz`](../../constrained_huckel_datasets/circulant/dataset.npz):
10,000 distinct-size, valence-constrained carbon rings with `n=6,10,…,40002`.
The exact uniform-coupling gap is `4 sin(pi/n)` in units of `|β|`.

## Directory map

```text
four_family_grokking/circulant_gap_probe/
├── README.md                       This index and result notes
├── runs/                           Constant-learning-rate runs, 40k steps
│   ├── raw_bits/                    16-bit transformer → raw gap
│   ├── log_bits/                    16-bit transformer → log gap
│   ├── log_flatbits/                16-feature MLP → log gap
│   └── log_scalar/                  log(n) MLP → log gap
├── runs_decay/                     Cosine-learning-rate controls, 40k steps
│   ├── log_bits/                    Shared-bit transformer → log gap
│   ├── log_flatbits/                16-feature MLP → log gap
│   └── log_posbits/                 Position-aware transformer → log gap
├── fourier_progress/               Snapshot runs and Nanda-inspired probes
├── train_gap.py                     Trainer and strict metrics
├── analyze_gap.py                   Main plots, size and extrapolation checks
├── compare_decay.py                 Optimizer and architecture comparisons
├── plot_architectures.py            Nanda-versus-gap architecture diagram
├── summary.json                    Main-run numerical results
├── decay_summary.json              Optimizer-control numerical results
└── architecture_scores.json        Diagram scores, including larger rings
```

Each run directory has `config.json`, `metrics.csv`, `latest.pt`, and
`last.pt`. The full console log is beside each run directory, for example
`runs/log_bits.log` or `runs_decay/log_bits.log`.

The tmux sessions are `grok_gap_raw_bits_gpu0`, `grok_gap_log_bits_gpu1`,
`grok_gap_log_scalar_gpu2`, `grok_gap_log_flatbits_gpu3`,
`grok_gap_log_bits_decay_gpu2`, `grok_gap_log_flatbits_decay_gpu3`, and
`grok_gap_log_posbits_decay_gpu3`. A completed session stays open at a Bash
prompt. Attach with `tmux attach -t SESSION_NAME`; end it with
`tmux kill-session -t SESSION_NAME`.

## What is being tested

The fixed-family sparse Hamiltonian is fully determined by `n`. Dense
40,000×40,000 inputs are impractical, so these models receive either the
16 binary digits of `n` or `log(n)`. They test **size-to-gap rule learning**;
they do not test reading matrix entries or learning Fourier eigenvectors
from matrices. The original n=42 matrix-input experiment is documented in
[`CIRCULANT_GROKKING_README.md`](../../CIRCULANT_GROKKING_README.md).

All neural runs use the same seed-0 random split: 3,000 training sizes and
7,000 held-out sizes. The primary accuracy is **gap relative error below
1%**; 5% accuracy, median relative error, log RMSE, and error by size are
also recorded. A separate extrapolation check evaluates `n=40006…60002`,
which is outside the original training-size range and was not used to tune
the models.

The raw gap varies by about 3.8 orders of magnitude. Only 24 samples
(`n≤100`, 0.24% of the set) contribute 95.5% of the squared error made by
a zero-gap predictor. Thus raw MSE mostly trains on tiny rings. Log-gap
training spreads attention across sizes; cosine learning-rate decay tests
whether the constant-rate models' oscillations explain poor final accuracy.

## Results at step 40,000

The table uses each **final** checkpoint, the 1% relative-gap threshold,
and the same random held-out set. The last column tests a separate range
of larger rings without retraining.

| Run directory | Input and model | Target / learning rate | Held-out | Larger rings |
| --- | --- | --- | ---: | ---: |
| `runs/raw_bits` | bit transformer | raw / constant | 6.54% | 1.96% |
| `runs/log_bits` | bit transformer | log / constant | 24.43% | 13.62% |
| `runs/log_flatbits` | flat-bit MLP | log / constant | 60.16% | 4.78% |
| `runs/log_scalar` | log(n) MLP | log / constant | 99.99% | 3.64% |
| `runs_decay/log_bits` | bit transformer | log / cosine | 96.11% | 35.58% |
| `runs_decay/log_flatbits` | flat-bit MLP | log / cosine | 96.90% | 4.78% |
| `runs_decay/log_posbits` | position-aware transformer | log / cosine | 98.49% | 4.78% |
| training-only baseline | log-linear fit | log / two parameters | 99.99% | 100.00% |

The constant-rate log-bit transformer reached 78.07% held-out 1% accuracy
at its best recorded step, but finished at 24.43%; the flat-bit MLP similarly
peaked at 93.33% and finished at 60.16%. Cosine decay made both stable and
raised their final scores to 96.11% and 96.90%. Position-specific token
embeddings raised the transformer's random-holdout score to 98.49%.

All neural models' train and test curves moved together, so these runs do
not show Nanda-style delayed grokking. The gap itself is learnable: the
two-parameter fit `log(gap) = a + b log(n)` estimated `b = -0.999945` from
the 3,000 training sizes and scored 100% on the larger-ring range. The
neural models' high random-holdout scores mostly reflect interpolation;
their larger-ring scores show that they have not reliably recovered the
inverse-size law. This is an inference from the split and held-out results,
not a claim that transformers cannot represent the law.

## Figures

- `gap_accuracy.png`: train/test 1% and 5% relative-gap accuracy for the four
  constant-rate runs, with constant and training-only log-linear baselines.
- `gap_by_size.png`: exact and predicted gaps, plus held-out relative error
  by ring size.
- `decay_comparison.png`: constant versus cosine learning rate for shared-bit
  transformer and flat-bit MLP.
- `architecture_comparison.png`: train/test curves for the three cosine
  bit-input architectures.
- `architecture_map.svg` and `architecture_map.png`: Nanda's modular-addition
  transformer and our gap-model pathways, with held-out and larger-ring
  accuracy. Nanda's classification accuracy and these regression accuracies
  are different tasks and should not be compared numerically.

## Reproduce the figures

```bash
cd /mnt/das_storage/nates_workspace/OPGrokkingDatasets/OPSandbox/four_family_grokking/circulant_gap_probe
/home/nate/anaconda3/envs/grokking/bin/python analyze_gap.py
/home/nate/anaconda3/envs/grokking/bin/python compare_decay.py
/home/nate/anaconda3/envs/grokking/bin/python plot_architectures.py
```

The original four runs can be restarted in separate tmux sessions with
`bash launch_tmux.sh`. The cosine controls used `run_decay_tmux.sh`. The
training scripts specify GPUs through `CUDA_VISIBLE_DEVICES`, and the run
configs record all model and optimizer settings.

Nanda's model diagram follows the [paper](https://arxiv.org/abs/2301.05217)
and this workspace's faithful
[`train_modular.py`](../../nanda_original_reproduction/train_modular.py)
reimplementation: three integer tokens `[a,b,=]`, learned token and
position embeddings, one causal four-head attention layer, ReLU MLP, and a
last-token readout over 114 logits. Our gap transformer instead uses
16 bit tokens, bidirectional attention, a CLS readout, and scalar regression.

## Fourier progress follow-up

The checkpointed Gini, Fourier-spectrum, and output-mode-filtering study is
in [`fourier_progress/`](fourier_progress/README.md). Its three retrained
models and all analysis figures are indexed there.
