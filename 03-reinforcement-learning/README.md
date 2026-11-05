# 03 · Reinforcement learning

This chapter moves from small state tables to neural policies. The folder names describe the learning rule; the environment is only an example.

1. [Tabular methods](01-tabular-methods/README.md): planning, Monte Carlo and TD prediction, Q-learning, and SARSA.
2. [Deep Q-learning](02-deep-q-learning/README.md): learn action values with replay and a target network.
3. [Policy gradients](03-policy-gradients/README.md): optimize a sampled policy using returns and a value baseline.
4. [Actor-critic](04-actor-critic/README.md): update a continuous-action policy from one-step TD errors.
5. [PPO](05-ppo/README.md): collect a rollout, estimate advantages, and make clipped policy updates.

Install the optional environments from the repository root with `uv sync --extra rl`. Each topic README has a run command. These examples demonstrate the algorithms; short runs do not establish trained performance.
