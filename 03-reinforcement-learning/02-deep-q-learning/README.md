# Deep Q-learning

[`dqn.py`](dqn.py) adapts the CartPole experiment from my `cartpole-rl` repo. It uses a replay buffer and an online network for action selection. A separate target network estimates the selected next action's value, giving a Double DQN target:

| File |
| --- |
| [dqn.py](dqn.py) |

Run from the repository root:

```bash
uv run --extra rl python 03-reinforcement-learning/02-deep-q-learning/dqn.py --episodes 650
```

[Explore the notes](https://ishan8840.github.io/deep-learning-notes/#/).
