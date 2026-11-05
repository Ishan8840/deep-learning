# Proximal policy optimization

[`model.py`](model.py) adapts the PPO script from my `pendulum-V1` repo. It trains a continuous-action Gaussian actor and value critic on Pendulum-v1, using GAE and PPO clipping. It reports the environment's raw reward; it does not use the source script's reward shaping. The repository does not include a trained checkpoint or benchmark result.

| File |
| --- |
| [model.py](model.py) |

Run from the repository root:

```bash
uv run --extra rl python 03-reinforcement-learning/05-ppo/model.py --episodes 1000
```

[Read the visual notes](https://ishan8840.github.io/deep-learning-notes/#/03-reinforcement-learning/06-ppo).
