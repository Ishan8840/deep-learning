"""Continuous-action PPO-Clip with GAE on Pendulum-v1."""

import argparse

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh())
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states):
        features = self.body(states)
        return Normal(self.mean(features), self.log_std.clamp(-5, 2).exp())


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, states):
        return self.net(states).squeeze(-1)


def squashed_log_prob(distribution, raw_actions):
    squashed = raw_actions.tanh()
    return (distribution.log_prob(raw_actions)
            - torch.log(1 - squashed.square() + 1e-6)).sum(dim=-1)


def advantages_and_returns(rewards, values, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return advantages, returns


def evaluate(actor, episodes=5, seed=10000, render=False):
    env = gym.make("Pendulum-v1", render_mode="human" if render else None)
    totals = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            done, total = False, 0.0
            while not done:
                with torch.no_grad():
                    distribution = actor(torch.as_tensor(state, dtype=torch.float32))
                    action = distribution.mean.tanh().numpy() * env.action_space.high
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
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    action_scale = torch.as_tensor(env.action_space.high, dtype=torch.float32)
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            states, raw_actions, old_log_probs, rewards, values = [], [], [], [], []
            done = False
            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    distribution = actor(state_tensor)
                    raw_action = distribution.sample()
                    log_prob = squashed_log_prob(distribution, raw_action)
                    value = critic(state_tensor)
                    action = (raw_action.tanh() * action_scale).numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                states.append(state_tensor)
                raw_actions.append(raw_action)
                old_log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                state = next_state
                done = terminated or truncated

            states = torch.stack(states)
            raw_actions = torch.stack(raw_actions)
            old_log_probs = torch.stack(old_log_probs)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            values = torch.stack(values)
            advantages, returns = advantages_and_returns(rewards, values)
            for _ in range(10):
                for batch in torch.randperm(len(states)).split(64):
                    distribution = actor(states[batch])
                    log_probs = squashed_log_prob(distribution, raw_actions[batch])
                    ratio = (log_probs - old_log_probs[batch]).exp()
                    objective = ratio * advantages[batch]
                    clipped = ratio.clamp(0.8, 1.2) * advantages[batch]
                    actor_loss = -torch.minimum(objective, clipped).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    critic_loss = nn.functional.mse_loss(critic(states[batch]), returns[batch])
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()
            if episode % 25 == 0 or episode == episodes - 1:
                print(f"episode {episode:4d} | environment return {rewards.sum().item():.1f}")
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
