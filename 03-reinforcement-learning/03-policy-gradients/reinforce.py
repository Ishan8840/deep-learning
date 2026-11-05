"""REINFORCE with a learned value baseline on CartPole-v1."""

import argparse

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, state_dim, actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, actions))

    def forward(self, state):
        return self.net(state)


class Value(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


def discounted_returns(rewards, gamma=0.99):
    returns = []
    future = 0.0
    for reward in reversed(rewards):
        future = reward + gamma * future
        returns.append(future)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)


def evaluate(policy, episodes=10, seed=10000, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    totals = []
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
            totals.append(total)
    finally:
        env.close()
    return float(np.mean(totals))


def train(episodes=600, seed=0, eval_episodes=10, render=False):
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    value = Value(env.observation_space.shape[0])
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            states, log_probs, rewards = [], [], []
            done = False
            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                distribution = Categorical(logits=policy(state_tensor))
                action = distribution.sample()
                next_state, reward, terminated, truncated, _ = env.step(int(action))
                states.append(state_tensor)
                log_probs.append(distribution.log_prob(action))
                rewards.append(reward)
                state = next_state
                done = terminated or truncated
            returns = discounted_returns(rewards)
            predictions = value(torch.stack(states))
            advantages = returns - predictions.detach()
            policy_loss = -(torch.stack(log_probs) * advantages).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            value_loss = nn.functional.mse_loss(predictions, returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            if episode % 25 == 0 or episode == episodes - 1:
                print(f"episode {episode:4d} | return {sum(rewards):.0f}")
    finally:
        env.close()
    if eval_episodes:
        print(f"mean evaluation return: {evaluate(policy, eval_episodes, render=render):.1f}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(args.episodes, args.seed, args.eval_episodes, args.render)
