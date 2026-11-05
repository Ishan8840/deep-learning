# Denoising Diffusion Probabilistic Models (DDPM)

[`ddpm.py`](ddpm.py) implements the forward-noise and reverse-denoising equations. [`train.py`](train.py) trains on MNIST; [`sampling.py`](sampling.py) and [`generate_grid.py`](generate_grid.py) produce examples.

| File |
| --- |
| [plot_schedules.py](plot_schedules.py) |
| [sampling.py](sampling.py) |
| [generate_grid.py](generate_grid.py) |
| [ddpm.py](ddpm.py) |
| [denoise_example.py](denoise_example.py) |
| [train.py](train.py) |

Run from the repository root:

```bash
uv run python 02-generative-models/02-ddpm/train.py --download --epochs 1
```

[Read the visual notes](https://ishan8840.github.io/deep-learning-notes/#/02-generative-models/03-ddpm).
