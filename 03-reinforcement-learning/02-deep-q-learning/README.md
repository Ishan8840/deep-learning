# Deep Q-learning

[`dqn.py`](dqn.py) adapts the CartPole experiment from my `cartpole-rl` repo. It uses a replay buffer and an online network for action selection. A separate target network estimates the selected next action's value, giving a Double DQN target:

$$y=r+\gamma(1-d)Q_{\text{target}}(s',\arg\max_a Q_{\text{online}}(s',a)).$$

```bash
uv run --extra rl python 03-reinforcement-learning/02-deep-q-learning/dqn.py --episodes 650
```

Evaluation runs without a window by default; pass `--render` to watch it. The script reports episode returns and the mean return on separate evaluation seeds.
