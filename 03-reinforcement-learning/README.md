# 03 · Reinforcement learning

The sequence runs from tabular planning and prediction to neural value and policy methods. Environment names appear inside each example; the folder names follow the ideas.

| Topic | What to look for |
| --- | --- |
| [01 · Markov decision processes](01-markov-decision-process/README.md) | States, actions, rewards, and tabular methods code |
| [02 · Deep Q-learning](02-deep-q-learning/README.md) | Replay and target networks |
| [03 · Policy gradients](03-policy-gradients/README.md) | REINFORCE with a value baseline |
| [04 · Actor-critic](04-actor-critic/README.md) | One-step TD updates on continuous actions |
| [05 · Generalized advantage estimation](05-generalized-advantage-estimation/README.md) | The bias–variance tradeoff in advantage estimates |
| [06 · PPO](06-ppo/README.md) | Clipped updates with GAE and Pendulum code |
| [07 · Soft Actor-Critic](07-soft-actor-critic/README.md) | Off-policy learning with an entropy objective |

Install the optional environments from the repository root with `uv sync --extra rl`. The implementation folders include run commands. These are learning examples; short runs do not establish trained performance.
