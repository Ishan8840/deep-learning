"""Sample actions and evaluate checkpoints. Run `python inference.py --help`."""

import argparse
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def sample_actions(model, qpos, images, num_steps=20, generator=None):
    """Euler integration from noise (t=0) to normalized actions (t=1)."""
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    was_training = model.training
    model.eval()
    try:
        actions = torch.randn(
            qpos.shape[0], model.chunk_size, model.action_dim,
            device=qpos.device, dtype=qpos.dtype, generator=generator,
        )
        memory = model.encode_observations(qpos, images)
        step_size = 1.0 / num_steps
        for step in range(num_steps):
            time = torch.full(
                (qpos.shape[0],), step / num_steps,
                device=actions.device, dtype=actions.dtype,
            )
            velocity = model.predict_velocity(memory, actions, time)
            actions = actions + step_size * velocity
        return actions
    finally:
        model.train(was_training)


def load_policy(checkpoint_path, device="cpu", weights="auto"):
    from model import FM

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FM(**checkpoint["config"]).to(device)
    if weights not in ("auto", "raw", "ema"):
        raise ValueError("weights must be auto, raw or ema")
    ema_weights = checkpoint.get("ema_model_state_dict")
    if weights == "ema" and ema_weights is None:
        raise ValueError("Checkpoint has no EMA weights")
    use_ema = weights == "ema" or (weights == "auto" and ema_weights is not None)
    model.load_state_dict(ema_weights if use_ema else checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_actions(model, qpos, images, stats, num_steps=20):
    """Raw positions + RGB float images in [0,1] -> actions in pixel units."""
    stats = {key: value.to(qpos.device) for key, value in stats.items()}
    qpos = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
    actions = sample_actions(model, qpos, images, num_steps)
    return actions * stats["action_std"] + stats["action_mean"]


def observation_tensors(history, device):
    qpos = np.stack([observation["agent_pos"] for observation in history])
    pixels = np.stack([observation["pixels"] for observation in history])
    qpos = torch.as_tensor(qpos, dtype=torch.float32, device=device)[None]
    images = torch.as_tensor(pixels, device=device)
    images = images.permute(0, 3, 1, 2)[None, :, None].float() / 255
    if len(history) == 1:
        return qpos[:, 0], images[:, 0]
    return qpos, images


def run_episode(env, model, stats, seed, num_steps, execute_steps, max_steps):
    device = next(model.parameters()).device
    observation, _ = env.reset(seed=seed)
    history = deque([observation] * model.obs_steps, maxlen=model.obs_steps)
    result = dict(seed=seed, success=False, max_reward=0.0, reward_sum=0.0, steps=0)

    while result["steps"] < max_steps:
        qpos, images = observation_tensors(history, device)
        actions = predict_actions(model, qpos, images, stats, num_steps)[0]
        for action in actions[:execute_steps].cpu().numpy():
            if not np.isfinite(action).all():
                raise RuntimeError("Policy generated nonfinite actions")
            action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, info = env.step(action)
            history.append(observation)
            result["steps"] += 1
            result["max_reward"] = max(result["max_reward"], float(reward))
            result["reward_sum"] += float(reward)
            result["success"] |= bool(info.get("is_success", False))
            if terminated or truncated or result["steps"] >= max_steps:
                return result
    return result


@torch.no_grad()
def rollout(model, stats, episodes, num_steps, execute_steps, max_steps, seed):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import gymnasium as gym
    import gym_pusht  # Registers PushT.

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array", max_episode_steps=max_steps)
    results = []
    try:
        for episode in range(episodes):
            row = run_episode(env, model, stats, seed + episode, num_steps, execute_steps, max_steps)
            results.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    return dict(success_rate=float(np.mean([r["success"] for r in results])),
                mean_max_reward=float(np.mean([r["max_reward"] for r in results])), episodes=results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["offline", "rollout", "both"], default="both")
    parser.add_argument("--weights", choices=["auto", "raw", "ema"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pusht"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=20, help="Euler integration steps")
    parser.add_argument("--execute-steps", type=int, default=8, help="Actions before replanning")
    parser.add_argument("--rollout-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.batch_size, args.num_steps, args.execute_steps, args.rollout_episodes,
           args.max_steps, args.threads) < 1 or args.max_batches < 0:
        parser.error("Counts must be positive; max-batches must be nonnegative")
    return args


def evaluate_offline(model, checkpoint, args):
    from torch.utils.data import DataLoader
    from train import REVISION, PushTDataset, load_episodes, validate

    if not checkpoint.get("val_episodes"):
        raise ValueError("Checkpoint has no held-out split; use --mode rollout")
    if checkpoint["dataset"]["revision"] != REVISION:
        raise ValueError("Checkpoint dataset revision differs from loader")
    ids = checkpoint["val_episodes"]
    episodes = load_episodes(args.data_dir, ids)
    dataset = PushTDataset(
        [episodes[i] for i in ids], model.chunk_size, checkpoint["stats"], model.obs_steps,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    return validate(model, loader, args.device, args.max_batches, args.num_steps, checkpoint["stats"])


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model, checkpoint = load_policy(args.checkpoint, args.device, args.weights)
    if args.execute_steps > model.chunk_size:
        raise ValueError("execute-steps cannot exceed the checkpoint's chunk size")
    report = dict(checkpoint=str(args.checkpoint), epoch=checkpoint.get("epoch"),
                  step=checkpoint.get("global_step"),
                  weights="ema" if args.weights != "raw" and checkpoint.get("ema_model_state_dict") is not None else "raw",
                  num_steps=args.num_steps, execute_steps=args.execute_steps,
                  max_steps=args.max_steps, seed=args.seed, max_batches=args.max_batches)
    if args.mode in ("offline", "both"):
        report["offline"] = evaluate_offline(model, checkpoint, args)
        print(json.dumps(report["offline"]), flush=True)
    if args.mode in ("rollout", "both"):
        report["rollout"] = rollout(model, checkpoint["stats"], args.rollout_episodes,
                                     args.num_steps, args.execute_steps, args.max_steps, args.seed)
    output = args.output or args.checkpoint.parent / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
