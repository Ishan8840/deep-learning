"""Monte Carlo, TD(0), and TD(lambda) prediction on a seven-state random walk."""

import argparse
import random

STATES = "ABCDEFG"
TERMINALS = {"A", "G"}
NONTERMINALS = STATES[1:-1]


def step(state, rng):
    next_state = STATES[STATES.index(state) + rng.choice((-1, 1))]
    return next_state, float(next_state == "G")


def monte_carlo(episodes, seed=0):
    rng = random.Random(seed)
    values = {state: 0.5 for state in NONTERMINALS}
    counts = {state: 0 for state in NONTERMINALS}
    for _ in range(episodes):
        state = "D"
        trajectory = []
        while state not in TERMINALS:
            next_state, reward = step(state, rng)
            trajectory.append((state, reward))
            state = next_state
        ret = 0.0
        for state, reward in reversed(trajectory):
            ret += reward
            counts[state] += 1
            values[state] += (ret - values[state]) / counts[state]
    return values


def td_zero(episodes, alpha=0.01, seed=0):
    rng = random.Random(seed)
    values = {state: 0.5 for state in NONTERMINALS}
    for _ in range(episodes):
        state = "D"
        while state not in TERMINALS:
            next_state, reward = step(state, rng)
            target = reward + (values[next_state] if next_state not in TERMINALS else 0.0)
            values[state] += alpha * (target - values[state])
            state = next_state
    return values


def td_lambda(episodes, alpha=0.01, lam=0.7, seed=0):
    rng = random.Random(seed)
    values = {state: 0.5 for state in NONTERMINALS}
    for _ in range(episodes):
        traces = {state: 0.0 for state in NONTERMINALS}
        state = "D"
        while state not in TERMINALS:
            next_state, reward = step(state, rng)
            next_value = values[next_state] if next_state not in TERMINALS else 0.0
            error = reward + next_value - values[state]
            traces[state] += 1.0
            for visited in NONTERMINALS:
                values[visited] += alpha * error * traces[visited]
                traces[visited] *= lam
            state = next_state
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()
    for name, method in (("MC", monte_carlo), ("TD(0)", td_zero), ("TD(lambda)", td_lambda)):
        print(name, method(args.episodes))
