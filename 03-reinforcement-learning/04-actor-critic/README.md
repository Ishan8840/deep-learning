# One-step actor-critic on Pendulum

[`pendulum_td.py`](pendulum_td.py) is adapted from the `actorcritic.py` script in my `pendulum-V1` repo. It is distinct from [REINFORCE](../03-policy-gradients/reinforce.py), which waits until the end of an episode to update, and from [PPO](../05-ppo/model.py), which reuses a rollout for clipped minibatch updates.

At each step, the critic learns from the one-step TD target $r_t+\gamma V(s_{t+1})$. The actor adjusts the log-probability of its sampled action using the detached TD error as an advantage estimate. The action is sampled from a Gaussian, passed through tanh, then scaled to Pendulum's action range. The script logs the environment's raw reward.

```bash
uv run --extra rl python 03-reinforcement-learning/04-actor-critic/pendulum_td.py --episodes 1000
```

Evaluation runs without rendering unless `--render` is passed. This is an educational implementation; no benchmark result or trained weights are checked in.

Related note: [Soft Actor-Critic](Soft%20Actor-Critic.md) extends the actor-critic idea with off-policy replay and an entropy objective. SAC is explained here, but not implemented in this folder.
