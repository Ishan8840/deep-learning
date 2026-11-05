"""Double DQN with replay and a target network for CartPole-v1."""

import argparse
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, actions))

    def forward(self, states):
        return self.net(states)


def evaluate(policy, episodes=10, seed=10000, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    returns = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            done, total = False, 0.0
            while not done:
                with torch.no_grad():
                    action = int(policy(torch.as_tensor(state, dtype=torch.float32)).argmax())
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            returns.append(total)
    finally:
        env.close()
    return float(np.mean(returns))


def train(episodes=650, seed=0, eval_episodes=10, render=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    online = QNetwork(env.observation_space.shape[0], env.action_space.n)
    target = QNetwork(env.observation_space.shape[0], env.action_space.n)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    replay = deque(maxlen=50000)
    steps = 0
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            epsilon = max(0.01, 0.995 ** episode)
            done, total = False, 0.0
            while not done:
                steps += 1
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = int(online(torch.as_tensor(state, dtype=torch.float32)).argmax())
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay.append((state, action, reward, next_state, done))
                state = next_state
                total += reward
                if len(replay) >= 64:
                    batch = random.sample(replay, 64)
                    states = torch.as_tensor(np.array([row[0] for row in batch]), dtype=torch.float32)
                    actions = torch.as_tensor([row[1] for row in batch], dtype=torch.long)
                    rewards = torch.as_tensor([row[2] for row in batch], dtype=torch.float32)
                    next_states = torch.as_tensor(np.array([row[3] for row in batch]), dtype=torch.float32)
                    dones = torch.as_tensor([row[4] for row in batch], dtype=torch.float32)
                    with torch.no_grad():
                        next_actions = online(next_states).argmax(dim=1, keepdim=True)
                        next_q = target(next_states).gather(1, next_actions).squeeze(1)
                        targets = rewards + 0.99 * (1 - dones) * next_q
                    predictions = online(states).gather(1, actions[:, None]).squeeze(1)
                    loss = nn.functional.mse_loss(predictions, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if steps % 200 == 0:
                    target.load_state_dict(online.state_dict())
            if episode % 25 == 0 or episode == episodes - 1:
                print(f"episode {episode:4d} | return {total:.0f} | epsilon {epsilon:.3f}")
    finally:
        env.close()
    if eval_episodes:
        print(f"mean evaluation return: {evaluate(online, eval_episodes, render=render):.1f}")
    return online


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=650)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(args.episodes, args.seed, args.eval_episodes, args.render)
