# One-step actor-critic on Pendulum

[`pendulum_td.py`](pendulum_td.py) is adapted from the `actorcritic.py` script in my `pendulum-V1` repo. It is distinct from [REINFORCE](../03-policy-gradients/reinforce.py), which waits until the end of an episode to update, and from [PPO](../05-ppo/model.py), which reuses a rollout for clipped minibatch updates.

| File |
| --- |
| [pendulum_td.py](pendulum_td.py) |

Run from the repository root:

```bash
uv run --extra rl python 03-reinforcement-learning/04-actor-critic/pendulum_td.py --episodes 1000
```

[Explore the notes](https://ishan8840.github.io/deep-learning-notes/#/).
