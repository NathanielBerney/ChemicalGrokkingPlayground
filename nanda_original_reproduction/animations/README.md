# Training animations

These four looping GIFs use the 41 archived seed-0 model snapshots, from step
0 to 40,000 at 1,000-step intervals. Their titles show held-out accuracy from
`../figure_run_seed0/metrics.csv` at each saved step.

| Animation | What changes |
| --- | --- |
| [Logit space](logit_space.gif) | For fixed `b=29`, the 113 × 113 map of centered output logits over input `a` and output class `c`. Black dots mark the correct class. The right panel shows `a=17`. Each frame uses its labeled logit scale to reveal the pattern as logit magnitude grows. |
| [Weight embedding space](embedding_space.gif) | Integer-token embeddings projected into one **fixed** PCA plane fitted on the final snapshot. Color is the phase of the final plane's dominant modular frequency, `k=41`. The axes stay fixed across frames. |
| [Fourier Gini](fourier_gini.gif) | Gini coefficients of the embedding and neuron-logit Fourier power, alongside train and test accuracy. Gini uses nonzero frequencies `k=1..56`; higher values mean power is concentrated in fewer frequencies. Metrics between saved model frames come from the 100-step training log. |
| [Frequency space](frequency_space.gif) | The fraction of total nonzero Fourier power at each frequency for the input embedding `W_E` and neuron-logit map `W_L = W_U[:, :113].T @ W_out`. Axes remain fixed, and orange bands mark the four final active frequencies. |

Generate the GIFs again from this directory's parent with:

```bash
/home/nate/anaconda3/envs/grokking/bin/python animate_training.py
```

Use `--run-dir`, `--output-dir`, `--fps`, `--device`, `--fixed-a`, or
`--fixed-b` to change the source or rendering. The script reads saved weights
and metrics; it does not train a model.
