# Policy gradients

[`reinforce.py`](reinforce.py) adapts the REINFORCE experiment from my `cartpole-rl` repo. A categorical policy samples actions, and a value network supplies a baseline. After each episode, the policy update weights action log-probabilities by discounted return minus the estimated value.

| File |
| --- |
| [reinforce.py](reinforce.py) |

Run from the repository root:

```bash
uv run --extra rl python 03-reinforcement-learning/03-policy-gradients/reinforce.py --episodes 600
```

[Explore the notes](https://ishan8840.github.io/deep-learning-notes/#/).
