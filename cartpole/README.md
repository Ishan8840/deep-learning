# CartPole control

These two implementations are adapted from my `cartpole-rl` repo. They use the same environment but learn in different ways:

| File | Learning rule |
| --- | --- |
| [`dqn.py`](dqn.py) | Double DQN: replay buffer, online action selection, target-network value estimate |
| [`reinforce.py`](reinforce.py) | REINFORCE: sampled categorical policy, discounted returns, learned value baseline |

The source scripts opened a render window automatically after training; these versions evaluate without rendering unless `--render` is passed. They also update the observation during evaluation and close the environments. A short run checks the code path, not learning quality.

```bash
uv run --extra rl python cartpole/dqn.py --episodes 650
uv run --extra rl python cartpole/reinforce.py --episodes 600
```

Both use the project's optional `rl` dependencies. The DQN target is $r+\gamma(1-d)Q_{\mathrm{target}}(s',\arg\max_a Q_{\mathrm{online}}(s',a))$. The REINFORCE policy update weights each action log-probability by its return minus the value baseline.
