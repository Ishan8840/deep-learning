# Markov decision processes and tabular methods

| File | Idea |
| --- | --- |
| [`value_iteration.py`](value_iteration.py) | Dynamic programming when the transition and reward rules are known |
| [`random_walk.py`](random_walk.py) | Monte Carlo, TD(0), and TD(λ) **value prediction** from sampled episodes |
| [`monte_carlo_control.py`](monte_carlo_control.py) | Every-visit Monte Carlo **control** from complete sampled returns |
| [`frozen_lake.py`](frozen_lake.py) | Q-learning and SARSA **control** on one shared FrozenLake task |

| File |
| --- |
| [value_iteration.py](value_iteration.py) |
| [monte_carlo_control.py](monte_carlo_control.py) |
| [random_walk.py](random_walk.py) |
| [frozen_lake.py](frozen_lake.py) |

Run from the repository root:

```bash
uv run --extra rl python 03-reinforcement-learning/01-tabular-methods/value_iteration.py
uv run --extra rl python 03-reinforcement-learning/01-tabular-methods/random_walk.py --episodes 10000
uv run --extra rl python 03-reinforcement-learning/01-tabular-methods/monte_carlo_control.py --episodes 10000
uv run --extra rl python 03-reinforcement-learning/01-tabular-methods/frozen_lake.py --episodes 10000
```

[Read the visual notes](https://ishan8840.github.io/deep-learning-notes/#/03-reinforcement-learning/01-markov-decision-process).
