"""Compare tabular Q-learning and SARSA on slippery FrozenLake."""

import argparse

import gymnasium as gym
import numpy as np


def choose_action(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    return int(np.argmax(q[state]))


def train(method, episodes=10000, alpha=0.1, gamma=0.99, seed=0):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            epsilon = max(0.1, 0.995 ** episode)
            action = choose_action(q, state, epsilon, rng)
            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if method == "q_learning":
                    next_value = 0.0 if done else np.max(q[next_state])
                    next_action = choose_action(q, next_state, epsilon, rng) if not done else 0
                elif method == "sarsa":
                    next_action = choose_action(q, next_state, epsilon, rng) if not done else 0
                    next_value = 0.0 if done else q[next_state, next_action]
                else:
                    raise ValueError(f"Unknown method: {method}")
                q[state, action] += alpha * (reward + gamma * next_value - q[state, action])
                state, action = next_state, next_action
    finally:
        env.close()
    return q


def evaluate(q, episodes=100, seed=100000):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    successes = 0
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            done = False
            while not done:
                state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
                done = terminated or truncated
            successes += int(reward == 1)
    finally:
        env.close()
    return successes / episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for method in ("q_learning", "sarsa"):
        q = train(method, args.episodes, seed=args.seed)
        print(f"{method}: {evaluate(q, args.eval_episodes):.1%} success")
