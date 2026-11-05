"""Every-visit Monte Carlo control with a decaying epsilon-greedy policy."""

import argparse

import numpy as np

GOAL = (3, 3)
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def step(state, action):
    dr, dc = ACTIONS[action]
    next_state = (min(3, max(0, state[0] + dr)), min(3, max(0, state[1] + dc)))
    return next_state, 10.0 if next_state == GOAL else -1.0


def train(episodes=10000, max_steps=100, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros((4, 4, len(ACTIONS)))
    visits = np.zeros_like(q)
    for episode in range(episodes):
        epsilon = 1.0 / (episode + 1)
        state = (0, 0)
        trajectory = []
        for _ in range(max_steps):
            action = int(rng.integers(len(ACTIONS))) if rng.random() < epsilon else int(np.argmax(q[state]))
            next_state, reward = step(state, action)
            trajectory.append((state, action, reward))
            state = next_state
            if state == GOAL:
                break
        ret = 0.0
        for state, action, reward in reversed(trajectory):
            ret = reward + ret
            index = (*state, action)
            visits[index] += 1
            q[index] += (ret - q[index]) / visits[index]
    return q


def greedy_path(q, max_steps=100):
    state, path = (0, 0), [(0, 0)]
    for _ in range(max_steps):
        state, _ = step(state, int(np.argmax(q[state])))
        path.append(state)
        if state == GOAL:
            break
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("greedy path:", greedy_path(train(args.episodes, seed=args.seed)))
