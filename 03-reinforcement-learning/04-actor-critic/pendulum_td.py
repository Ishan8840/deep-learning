"""One-step actor-critic for continuous Pendulum-v1 actions."""

import argparse

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        return Normal(self.net(state), self.log_std.clamp(-5, 2).exp())


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


def evaluate(actor, episodes=5, seed=10000, render=False):
    env = gym.make("Pendulum-v1", render_mode="human" if render else None)
    totals = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            done, total = False, 0.0
            while not done:
                with torch.no_grad():
                    mean = actor(torch.as_tensor(state, dtype=torch.float32)).mean
                    action = mean.tanh().numpy() * env.action_space.high
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            totals.append(total)
    finally:
        env.close()
    return float(np.mean(totals))


def train(episodes=1000, seed=0, eval_episodes=5, render=False):
    torch.manual_seed(seed)
    env = gym.make("Pendulum-v1")
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    critic = Critic(env.observation_space.shape[0])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    scale = torch.as_tensor(env.action_space.high, dtype=torch.float32)
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            done, total = False, 0.0
            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                distribution = actor(state_tensor)
                raw_action = distribution.sample()
                squashed = raw_action.tanh()
                log_prob = (distribution.log_prob(raw_action)
                            - torch.log(1 - squashed.square() + 1e-6)).sum()
                next_state, reward, terminated, truncated, _ = env.step((squashed * scale).detach().numpy())
                done = terminated or truncated
                value = critic(state_tensor)
                with torch.no_grad():
                    next_value = 0.0 if done else critic(torch.as_tensor(next_state, dtype=torch.float32))
                    target = reward + 0.99 * next_value
                delta = target - value
                critic_optimizer.zero_grad()
                delta.square().backward()
                critic_optimizer.step()
                actor_optimizer.zero_grad()
                (-log_prob * delta.detach()).backward()
                actor_optimizer.step()
                total += reward
                state = next_state
            if episode % 25 == 0 or episode == episodes - 1:
                print(f"episode {episode:4d} | environment return {total:.1f}")
    finally:
        env.close()
    if eval_episodes:
        print(f"mean evaluation return: {evaluate(actor, eval_episodes, render=render):.1f}")
    return actor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(args.episodes, args.seed, args.eval_episodes, args.render)
