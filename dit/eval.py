"""Roll out a trained DiT or ACT in Gym PushT and save videos and metrics."""

import argparse
from collections import deque
from contextlib import nullcontext
import importlib.metadata
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import time

# These must be set before importing pygame/gym_pusht on a headless VM.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import av
import cv2
from diffusers import DDIMScheduler, DDPMScheduler
import gymnasium as gym
import gym_pusht  # noqa: F401 -- registers gym_pusht/PushT-v0
import numpy as np
import torch

from model import DiT


class DiffusionPolicy:
    def __init__(self, checkpoint_path, device="cuda", sampler="ddim", denoise_steps=10, amp="none"):
        self.device = torch.device(device)
        self.amp = amp
        # Only load trusted checkpoints (training saves Python/NumPy RNG states).
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = DiT(**checkpoint["model_kwargs"]).to(self.device).eval()
        self.model.load_state_dict(checkpoint["model"])
        self.checkpoint_step = checkpoint["step"]
        self.horizon = checkpoint["model_kwargs"]["horizon"]
        self.action_dim = checkpoint["model_kwargs"]["action_dim"]
        self.n_obs_steps = checkpoint["model_kwargs"].get("n_obs_steps", 1)
        self.stats = {key: {name: value.to(self.device) for name, value in stats.items()}
                      for key, stats in checkpoint["normalization"].items()}
        self.fps = checkpoint["preprocessing"]["fps"]
        if checkpoint["preprocessing"]["image_normalization"] != "2*x-1":
            raise ValueError("Unsupported checkpoint image preprocessing")
        schedule = checkpoint["betas"].cpu().numpy()
        if not 1 <= denoise_steps <= len(schedule):
            raise ValueError("denoise_steps must be within the trained diffusion schedule")
        scheduler_class = DDPMScheduler if sampler == "ddpm" else DDIMScheduler
        self.scheduler = scheduler_class(
            num_train_timesteps=len(schedule), trained_betas=schedule,
            prediction_type="epsilon", clip_sample=True, clip_sample_range=1.0,
            timestep_spacing="trailing" if sampler == "ddim" else "linspace",
        )
        self.scheduler.set_timesteps(denoise_steps)
        # Python integer timesteps avoid a GPU-to-CPU synchronization every step.
        self.timesteps = self.scheduler.timesteps.tolist()
        self.sampler = sampler
        self.reset()

    def reset(self):
        self.observation_history = deque(maxlen=self.n_obs_steps)

    def observe(self, observation):
        """Record every control step, including steps within an action chunk."""
        snapshot = {"agent_pos": np.asarray(observation["agent_pos"]).copy(),
                    "pixels": np.asarray(observation["pixels"]).copy()}
        if not self.observation_history:
            self.observation_history.extend([snapshot] * self.n_obs_steps)
        else:
            self.observation_history.append(snapshot)

    def autocast(self):
        if self.amp == "none":
            return nullcontext()
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    @torch.inference_mode()
    def sample(self, observation, generator):
        if (not self.observation_history
                or not np.array_equal(self.observation_history[-1]["agent_pos"], observation["agent_pos"])
                or not np.array_equal(self.observation_history[-1]["pixels"], observation["pixels"])):
            self.observe(observation)
        state = torch.as_tensor(np.stack([item["agent_pos"] for item in self.observation_history]), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        state_stats = self.stats["observation.state"]
        state = 2 * (state - state_stats["min"]) / (state_stats["max"] - state_stats["min"]).clamp_min(1e-6) - 1
        pixels = observation["pixels"]
        if pixels.shape != (96, 96, 3) or pixels.dtype != np.uint8:
            raise ValueError("Expected 96x96 RGB uint8 observations, matching training")
        image = torch.as_tensor(np.stack([item["pixels"] for item in self.observation_history]),
                                device=self.device).permute(0, 3, 1, 2).unsqueeze(0).float()
        if self.n_obs_steps == 1:
            state, image = state[:, 0], image[:, 0]
        image = image.div(255).mul(2).sub(1)
        with self.autocast():
            state_emb, image_emb = self.model.encode_observation(state, image)
        actions = torch.randn((1, self.horizon, self.action_dim), device=self.device, generator=generator)
        timestep = torch.empty((1,), device=self.device, dtype=torch.long)
        for t in self.timesteps:
            timestep.fill_(t)
            with self.autocast():
                noise = self.model.forward_with_observation(actions, timestep, state_emb, image_emb)
            kwargs = {"eta": 0.0} if self.sampler == "ddim" else {"generator": generator}
            actions = self.scheduler.step(noise.float(), t, actions, **kwargs).prev_sample
        stats = self.stats["action"]
        actions = (actions.clamp(-1, 1) + 1) * 0.5 * (stats["max"] - stats["min"]).clamp_min(1e-6) + stats["min"]
        actions = actions.squeeze(0).cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("Non-finite sampled action")
        return actions


class ACTPolicy:
    def __init__(self, checkpoint_path, act_dir, device="cuda", amp="none", temporal_ensemble=False):
        self.device = torch.device(device)
        self.amp = amp
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = act_dir / "act.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != checkpoint["act_source_sha256"]:
            raise ValueError("ACT source differs from the source used to train this checkpoint")
        spec = importlib.util.spec_from_file_location("act_rollout_model", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.ACT(**checkpoint["config"]).to(self.device).eval()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.checkpoint_step = checkpoint["step"]
        self.horizon = checkpoint["config"]["chunk_size"]
        self.fps = checkpoint["preprocessing"]["fps"]
        if checkpoint["preprocessing"]["image_normalization"] != "x/255":
            raise ValueError("Unsupported ACT image preprocessing")
        self.stats = {key: {name: value.to(self.device) for name, value in stats.items()}
                      for key, stats in checkpoint["normalization"].items()}
        self.ensemble = module.TemporalEnsemble(decay=0.01) if temporal_ensemble else None

    def reset(self):
        if self.ensemble is not None:
            self.ensemble.reset()

    @torch.inference_mode()
    def sample(self, observation, generator):
        state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        stats = self.stats["observation.state"]
        state = 2 * (state - stats["min"]) / (stats["max"] - stats["min"]).clamp_min(1e-6) - 1
        pixels = observation["pixels"]
        if pixels.shape != (96, 96, 3) or pixels.dtype != np.uint8:
            raise ValueError("Expected 96x96 RGB uint8 observations")
        image = torch.as_tensor(pixels.copy(), device=self.device).permute(2, 0, 1).float().div(255)
        # ACT has an explicit camera axis and was trained with images in [0,1].
        image = image.unsqueeze(0).unsqueeze(0)
        context = nullcontext() if self.amp == "none" else torch.autocast("cuda", dtype=torch.bfloat16)
        with context:
            predicted, _, _ = self.model(state, image)  # no targets -> zero latent
        stats = self.stats["action"]
        actions = (predicted.float().squeeze(0) + 1) * 0.5 * (stats["max"] - stats["min"]).clamp_min(1e-6) + stats["min"]
        if self.ensemble is not None:
            actions = self.ensemble(actions).unsqueeze(0)
        actions = actions.cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("Non-finite ACT action")
        return actions


class VideoWriter:
    def __init__(self, path, fps):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=int(fps))
        self.stream.width = self.stream.height = 512
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "20", "preset": "fast"}

    def write(self, image, text):
        image = image.copy()
        cv2.rectangle(image, (0, 0), (512, 28), (255, 255, 255), -1)
        cv2.putText(image, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=("dit", "act"), default="dit")
    parser.add_argument("--act-dir", type=Path, default=Path(__file__).resolve().parent.parent / "act")
    parser.add_argument("--temporal-ensemble", action="store_true",
                        help="ACT only: combine overlapping chunks; requires --execute-steps 1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pusht-eval"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--execute-steps", type=int, default=8)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.execute_steps, args.denoise_steps) < 1:
        parser.error("episodes, max-steps, execute-steps, and denoise-steps must be positive")
    if args.temporal_ensemble and (args.policy != "act" or args.execute_steps != 1):
        parser.error("--temporal-ensemble requires --policy act --execute-steps 1")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or install CUDA PyTorch")
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 evaluation requires a supported CUDA GPU")
    if (args.output_dir / "episodes.jsonl").exists():
        raise FileExistsError("Choose a new output-dir to avoid overwriting evaluation results")
    if args.policy == "act":
        policy = ACTPolicy(args.checkpoint, args.act_dir, args.device, args.amp, args.temporal_ensemble)
        args.sampler = args.denoise_steps = None
    else:
        policy = DiffusionPolicy(args.checkpoint, args.device, args.sampler, args.denoise_steps, args.amp)
    if args.execute_steps > policy.horizon:
        raise ValueError("execute-steps must not exceed the model horizon")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(checkpoint_step=policy.checkpoint_step,
                  n_obs_steps=getattr(policy, "n_obs_steps", 1),
                  gym_pusht_version=importlib.metadata.version("gym-pusht"),
                  diffusers_version=importlib.metadata.version("diffusers"))
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                   observation_width=96, observation_height=96,
                   visualization_width=512, visualization_height=512, max_episode_steps=args.max_steps)
    if env.unwrapped.metadata["render_fps"] != policy.fps:
        raise ValueError("Environment control rate differs from training")
    results = []
    print(json.dumps({"event": "start", **config}), flush=True)
    try:
        with (args.output_dir / "episodes.jsonl").open("w") as log:
            for episode in range(args.episodes):
                if hasattr(policy, "reset"):
                    policy.reset()
                seed = args.seed + episode
                observation, info = env.reset(seed=seed)
                if hasattr(policy, "observe"):
                    policy.observe(observation)
                generator = torch.Generator(device=device).manual_seed(args.policy_seed + episode)
                rewards, actions, positions, coverages, plans = [], [], [], [], []
                success = False
                started = time.monotonic()
                video_path = args.output_dir / f"episode_{episode:03d}_seed_{seed}.mp4"
                writer = VideoWriter(video_path, policy.fps)
                writer.write(env.render(), f"Seed {seed} | step 0 | checkpoint {policy.checkpoint_step}")
                try:
                    while len(rewards) < args.max_steps:
                        chunk = policy.sample(observation, generator)
                        plans.append(chunk)
                        for action in chunk[:args.execute_steps]:
                            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
                            actions.append(action)
                            positions.append(observation["agent_pos"].copy())
                            observation, reward, terminated, truncated, info = env.step(action)
                            if hasattr(policy, "observe"):
                                policy.observe(observation)
                            rewards.append(float(reward))
                            coverages.append(float(info["coverage"]))
                            success = success or bool(info["is_success"])
                            writer.write(env.render(), f"Seed {seed} | step {len(rewards)} | reward {reward:.3f} | success {success}")
                            if terminated or truncated:
                                break
                        if terminated or truncated:
                            break
                        if len(plans) % 10 == 0:
                            print(json.dumps({"event": "progress", "episode": episode, "steps": len(rewards),
                                              "max_reward": max(rewards), "elapsed_s": time.monotonic() - started}), flush=True)
                finally:
                    writer.close()
                np.savez_compressed(args.output_dir / f"episode_{episode:03d}_trajectory.npz",
                                    actions=np.asarray(actions), positions=np.asarray(positions),
                                    rewards=np.asarray(rewards), coverage=np.asarray(coverages), plans=np.asarray(plans))
                result = dict(episode=episode, seed=seed, success=success, steps=len(rewards),
                              sum_reward=sum(rewards), max_reward=max(rewards), final_reward=rewards[-1],
                              max_coverage=max(coverages), final_coverage=coverages[-1],
                              wall_seconds=time.monotonic() - started, video=str(video_path))
                results.append(result)
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(json.dumps({"event": "episode", **result}), flush=True)
    finally:
        env.close()
    summary = dict(episodes=len(results), successes=sum(row["success"] for row in results),
                   success_rate=float(np.mean([row["success"] for row in results])),
                   mean_max_reward=float(np.mean([row["max_reward"] for row in results])),
                   mean_sum_reward=float(np.mean([row["sum_reward"] for row in results])),
                   checkpoint_step=policy.checkpoint_step, sampler=args.sampler,
                   denoise_steps=args.denoise_steps, execute_steps=args.execute_steps,
                   policy=args.policy, temporal_ensemble=args.temporal_ensemble,
                   n_obs_steps=getattr(policy, "n_obs_steps", 1),
                   episode_results=results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "finished", **summary}), flush=True)


if __name__ == "__main__":
    main()
