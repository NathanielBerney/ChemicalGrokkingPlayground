# Three cosine modes for an N = 3 ring

The three separate plots show

$$
f_k(j)=\alpha+2\beta\cos\left(\frac{2\pi k j}{3}\right),
\qquad k=0,1,2.
$$

The plots use $\alpha=0$ and $\beta=-1$. Position $j$ runs continuously around
one ring for the line plots, while dots mark the three sites $j=0,1,2$.
At $j=1$, the plotted function gives the requested Hückel energy
$E_k=\alpha+2\beta\cos(2\pi k/3)$: $E_0=-2$ and $E_1=E_2=+1$.

- [Cosine mode k = 0](cosine_k0.png)
- [Cosine mode k = 1](cosine_k1.png)
- [Cosine mode k = 2](cosine_k2.png)
- [Mode-frequency spectrum](frequency_spectrum.png)

The frequency chart shows the one-sided Fourier amplitude of each continuous
cosine curve at $k/3$ cycles per site. Its $k=0$ bar is the constant (DC)
component. With only three sampled sites, the $k=1$ and $k=2$ cosine values
coincide; the continuous curves show their different frequencies.

Regenerate the four PNG plots by running python plot_cosine_modes.py from
this directory.
