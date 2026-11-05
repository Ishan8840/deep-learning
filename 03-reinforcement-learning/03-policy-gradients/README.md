# Policy gradients

[`reinforce.py`](reinforce.py) adapts the REINFORCE experiment from my `cartpole-rl` repo. A categorical policy samples actions, and a value network supplies a baseline. After each episode, the policy update weights action log-probabilities by discounted return minus the estimated value.

```bash
uv run --extra rl python 03-reinforcement-learning/03-policy-gradients/reinforce.py --episodes 600
```

Unlike [DQN](../02-deep-q-learning/README.md), this method directly changes the policy rather than learning action values for greedy selection. It evaluates on separate seeds without opening a window unless `--render` is passed.
