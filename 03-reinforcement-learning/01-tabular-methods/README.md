# Tabular reinforcement learning

These examples are adapted from my `rl-basics` repo. They keep one implementation for each distinct idea:

The [Markov Decision Process](Markov%20Decision%20Process.md) note defines the state, action, transition, reward, and discounting concepts used by these examples.

| File | Idea |
| --- | --- |
| [`value_iteration.py`](value_iteration.py) | Dynamic programming when the transition and reward rules are known |
| [`random_walk.py`](random_walk.py) | Monte Carlo, TD(0), and TD(λ) **value prediction** from sampled episodes |
| [`monte_carlo_control.py`](monte_carlo_control.py) | Every-visit Monte Carlo **control** from complete sampled returns |
| [`frozen_lake.py`](frozen_lake.py) | Q-learning and SARSA **control** on one shared FrozenLake task |

The key distinction in `frozen_lake.py` is the bootstrap target: Q-learning uses the best estimated next action, while SARSA uses the next action sampled by the current policy. Both stop bootstrapping when an episode ends. The evaluation uses held-out random seeds and a greedy policy.

```bash
uv run --extra rl python 03-reinforcement-learning/01-tabular-methods/value_iteration.py
uv run --extra rl python 03-reinforcement-learning/01-tabular-methods/random_walk.py --episodes 10000
uv run --extra rl python 03-reinforcement-learning/01-tabular-methods/monte_carlo_control.py --episodes 10000
uv run --extra rl python 03-reinforcement-learning/01-tabular-methods/frozen_lake.py --episodes 10000
```

The FrozenLake example uses the project's optional `rl` dependencies. I left out the separate gridworld, cliff-walking, and FrozenLake copies of Q-learning and SARSA because they repeat the same update rules. The one-hot linear Q-learning example is also equivalent to a tabular Q-table here.
