# Diffusion Policy

[`model.py`](model.py) implements a transformer-based action denoiser conditioned on PushT observations. [`train.py`](train.py) trains it and [`eval.py`](eval.py) runs closed-loop rollouts. The [example rollout](success_rollout.mp4) shows one successful episode. See DiT architecture notes for the image-generation model that motivated the backbone.

| File |
| --- |
| [model.py](model.py) |
| [eval.py](eval.py) |
| [train.py](train.py) |

[Read the visual notes](https://ishan8840.github.io/deep-learning-notes/#/04-robot-policies/02-diffusion-policy).
