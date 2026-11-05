"""Train the conditional DiT on lerobot/pusht. Downloads begin only in main()."""

# Runtime: Python 3.11, torch 2.7.1, torchvision 0.22.1, lerobot 0.4.3.
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import DiT, betas, noise_scheduler, q_sample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="For training without validation: train.py --until-plateau --help")
    parser.add_argument("--repo-id", default="lerobot/pusht")
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pusht"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-images", action="store_true",
                        help="Decode PushT video once into CPU RAM (use --workers 0)")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-batches", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=4,
                        help="Stop after this many validations without meaningful improvement; 0 disables")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for key in ("steps", "batch_size", "horizon", "n_obs_steps", "hidden_dim", "num_heads",
                "val_batches", "eval_every", "save_every", "log_every"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or not 0 <= args.warmup_steps < args.steps:
        parser.error("workers must be nonnegative and 0 <= warmup-steps < steps")
    if not 0 < args.val_fraction < 1 or args.hidden_dim % args.num_heads:
        parser.error("val-fraction must be in (0, 1); hidden-dim must divide evenly by num-heads")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("lr and grad-clip must be positive; weight-decay must be nonnegative")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0:
        parser.error("early stopping patience and min-delta must be nonnegative")
    if args.cache_images and args.workers != 0:
        parser.error("--cache-images requires --workers 0")
    return args


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, args, shuffle, device):
    kwargs = {}
    if args.workers:
        # Spawn avoids forking an initialized CUDA runtime or video decoder.
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + (0 if shuffle else 1)),
        **kwargs,
    )


def fit_normalization(dataset):
    """Read only training parquet columns, without decoding any images."""
    stats = {}
    for key in ("action", "observation.state"):
        values = torch.stack(list(dataset.hf_dataset[key])).float()
        stats[key] = {"min": values.amin(0), "max": values.amax(0)}
    return stats


def normalize(value, stats):
    return 2 * (value - stats["min"]) / (stats["max"] - stats["min"]).clamp_min(1e-6) - 1


def prepare_batch(batch, stats, device):
    actions = normalize(batch["action"].to(device, dtype=torch.float32, non_blocking=True), stats["action"])
    state = normalize(batch["observation.state"].to(device, dtype=torch.float32, non_blocking=True), stats["observation.state"])
    # LeRobot's decoder already returns CHW float images in [0, 1].
    image = batch["observation.image"].to(device, dtype=torch.float32, non_blocking=True)
    image = (image - 0.5) / 0.5
    valid = ~batch["action_is_pad"].to(device, non_blocking=True)
    return actions, state, image, valid


def noise_loss(model, batch, stats, device, amp):
    actions, state, image, valid = prepare_batch(batch, stats, device)
    t = torch.randint(len(betas), (actions.shape[0],), device=device)
    noisy, noise = q_sample(actions, t)
    context = nullcontext() if amp == "none" else torch.autocast(
        device_type=device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16
    )
    with context:
        prediction = model(noisy, state, t, image)
    squared_error = F.mse_loss(prediction.float(), noise, reduction="none")
    # Exclude repeated endpoint actions used by LeRobot to pad each episode.
    numerator = (squared_error * valid.unsqueeze(-1)).sum()
    denominator = valid.sum() * actions.shape[-1]
    return numerator / denominator.clamp_min(1), numerator.detach(), denominator.detach()


@torch.no_grad()
def validate(model, loader, stats, device, args):
    model.eval()
    numerator = denominator = 0.0
    # Keep validation noise fixed across evaluations without changing training RNG.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed + 10_000)
        for i, batch in enumerate(loader):
            _, num, den = noise_loss(model, batch, stats, device, args.amp)
            numerator += num.item()
            denominator += den.item()
            if i + 1 >= args.val_batches:
                break
    model.train()
    return numerator / max(denominator, 1)


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA PyTorch or pass --device cpu.")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("AMP is supported here on CUDA only; use --amp none for CPU.")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use --amp fp16 or --amp none.")
    if args.resume is None and (args.output_dir / "last.pt").exists():
        raise FileExistsError("Output already has last.pt; pass --resume or choose another output-dir.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = None
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(diffusion_steps=len(betas), beta_schedule=noise_scheduler.config.beta_schedule)
    if args.resume:
        # Load only your own trusted checkpoints; these include Python/NumPy RNG state.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not torch.equal(checkpoint["betas"].cpu(), betas.cpu()):
            raise ValueError("Resume checkpoint has a different diffusion schedule; start a fresh run")
        if args.n_obs_steps != checkpoint["model_kwargs"].get("n_obs_steps", 1):
            raise ValueError("Resume requires the checkpoint's original --n-obs-steps")
        for key in ("repo_id", "revision", "seed", "horizon", "hidden_dim", "num_heads",
                    "batch_size", "lr", "weight_decay", "warmup_steps", "steps", "val_fraction", "amp", "grad_clip",
                    "early_stop_patience", "early_stop_min_delta"):
            if config[key] != checkpoint["config"][key]:
                raise ValueError(f"Resume requires the original --{key.replace('_', '-')}")

    # Import/instantiate the downloader only when explicitly launching this script.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    print(f"Loading {args.repo_id}@{args.revision}; uncached data will download now.", flush=True)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root, revision=args.revision)
    if metadata.total_episodes < 2:
        raise ValueError("At least two episodes are required for a held-out split")
    if tuple(metadata.features["action"]["shape"]) != (2,) or tuple(metadata.features["observation.state"]["shape"]) != (2,):
        raise ValueError("This setup expects PushT's 2D state and 2D action")
    episodes = np.random.default_rng(args.seed).permutation(metadata.total_episodes).tolist()
    count = min(metadata.total_episodes - 1, max(1, round(len(episodes) * args.val_fraction)))
    val_episodes, train_episodes = sorted(episodes[:count]), sorted(episodes[count:])
    if checkpoint and (train_episodes != checkpoint["train_episodes"] or val_episodes != checkpoint["val_episodes"]):
        raise ValueError("Dataset episode split has changed since the checkpoint")
    observation_offsets = [i / metadata.fps for i in range(1 - args.n_obs_steps, 1)]
    delta_timestamps = {"action": [i / metadata.fps for i in range(args.horizon)]}
    if args.n_obs_steps > 1:
        delta_timestamps.update({"observation.state": observation_offsets,
                                 "observation.image": observation_offsets})
    dataset_kwargs = dict(
        repo_id=args.repo_id, root=args.dataset_root, revision=args.revision,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    train_dataset = LeRobotDataset(episodes=train_episodes, **dataset_kwargs)
    val_dataset = LeRobotDataset(episodes=val_episodes, **dataset_kwargs)
    stats_cpu = checkpoint["normalization"] if checkpoint else fit_normalization(train_dataset)
    stats = {key: {name: value.to(device) for name, value in item.items()} for key, item in stats_cpu.items()}
    if args.cache_images:
        video_cache = {}
        train_dataset = CachedPushTDataset(train_dataset, args.horizon, video_cache, args.n_obs_steps)
        val_dataset = CachedPushTDataset(val_dataset, args.horizon, video_cache, args.n_obs_steps)
        del video_cache
    train_loader = make_loader(train_dataset, args, True, device)
    val_loader = make_loader(val_dataset, args, False, device)

    model_kwargs = dict(action_dim=2, hidden_dim=args.hidden_dim, horizon=args.horizon,
                        state_dim=2, num_heads=args.num_heads, n_obs_steps=args.n_obs_steps)
    model = DiT(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_multiplier(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / (args.steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    start_step, best_val = 0, float("inf")
    best_step, bad_validations, early_stop_reference = 0, 0, float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
        best_step = checkpoint.get("best_step", start_step)
        bad_validations = checkpoint.get("bad_validations", 0)
        early_stop_reference = checkpoint.get("early_stop_reference", best_val)
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if device.type == "cuda" and checkpoint["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        train_loader.generator.set_state(checkpoint["loader_rng"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Train: {len(train_episodes)} episodes / {len(train_dataset)} frames; "
          f"validation: {len(val_episodes)} episodes / {len(val_dataset)} frames", flush=True)
    model.train()
    iterator = iter(train_loader)
    running, running_count, started = 0.0, 0, time.monotonic()
    stopped_early, step = False, start_step
    with (args.output_dir / "metrics.jsonl").open("a") as metrics:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = noise_loss(model, batch, stats, device, args.amp)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            running += loss.item()
            running_count += 1
            record = {"step": step, "lr": optimizer.param_groups[0]["lr"]}
            if step % args.log_every == 0 or step == args.steps:
                record.update(train_loss=running / running_count, elapsed_s=time.monotonic() - started)
                running, running_count = 0.0, 0
            val_loss = None
            if step % args.eval_every == 0 or step == args.steps:
                val_loss = validate(model, val_loader, stats, device, args)
                if not math.isfinite(val_loss):
                    raise FloatingPointError(f"Non-finite validation loss at step {step}")
                record["val_loss"] = val_loss
            improved = val_loss is not None and val_loss < best_val
            if improved:
                best_val = val_loss
                best_step = step
            if val_loss is not None:
                if val_loss < early_stop_reference - args.early_stop_min_delta:
                    early_stop_reference, bad_validations = val_loss, 0
                else:
                    bad_validations += 1
                record.update(best_val=best_val, best_step=best_step, bad_validations=bad_validations)
                stopped_early = args.early_stop_patience > 0 and bad_validations >= args.early_stop_patience
            if len(record) > 2:
                print(json.dumps(record), flush=True)
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
            if improved or step % args.save_every == 0 or step == args.steps or stopped_early:
                payload = dict(
                    step=step, best_val=best_val, config=config, model_kwargs=model_kwargs,
                    best_step=best_step, bad_validations=bad_validations,
                    early_stop_reference=early_stop_reference,
                    model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                    normalization=stats_cpu, train_episodes=train_episodes, val_episodes=val_episodes,
                    betas=betas.clone(), loader_rng=train_loader.generator.get_state(),
                    preprocessing={"image_key": "observation.image", "image_normalization": "2*x-1", "fps": metadata.fps,
                                   "n_obs_steps": args.n_obs_steps, "observation_offsets_s": observation_offsets},
                    rng={"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                )
                save_checkpoint(args.output_dir / "last.pt", payload)
                if improved:
                    save_checkpoint(args.output_dir / "best.pt", payload)
            if stopped_early:
                print(f"Early stopping: {bad_validations} validations without improvement "
                      f"> {args.early_stop_min_delta}. Best validation {best_val:.6f} at step {best_step}.", flush=True)
                break
    summary = dict(status="early_stopped" if stopped_early else "completed",
                   step=step, best_step=best_step, best_val=best_val,
                   best_checkpoint=str(args.output_dir / "best.pt"))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Finished. Checkpoints: {args.output_dir.resolve()}", flush=True)



class CachedPushTDataset(Dataset):
    def __init__(self, source, horizon, video_cache, n_obs_steps=1):
        import av

        if n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive")
        self.n_obs_steps = n_obs_steps
        def column(key):
            return torch.stack(list(source.hf_dataset[key]))

        self.actions = column("action").float()
        self.states = column("observation.state").float()
        episodes = column("episode_index").flatten().long()
        timestamps = column("timestamp").flatten().double().numpy()
        self.ends = torch.empty(len(episodes), dtype=torch.long)
        self.starts = torch.empty(len(episodes), dtype=torch.long)
        self.offsets = torch.arange(horizon)
        self.observation_offsets = torch.arange(1 - n_obs_steps, 1)
        images = None
        for episode in episodes.unique().tolist():
            rows = (episodes == episode).nonzero().flatten()
            self.ends[rows] = rows[-1]
            self.starts[rows] = rows[0]
            path = source.root / source.meta.get_video_file_path(episode, "observation.image")
            if path not in video_cache:
                frames, times = [], []
                with av.open(str(path)) as container:
                    for frame in container.decode(video=0):
                        frames.append(frame.to_ndarray(format="rgb24"))
                        times.append(float(frame.pts * frame.time_base))
                video_cache[path] = np.asarray(times), np.stack(frames)
                print(f"Cached {len(frames)} RGB frames from {path.name}", flush=True)
            times, frames = video_cache[path]
            offset = source.meta.episodes[episode]["videos/observation.image/from_timestamp"]
            targets = timestamps[rows.numpy()] + offset
            right = np.searchsorted(times, targets).clip(0, len(times) - 1)
            left = (right - 1).clip(0, len(times) - 1)
            selected = np.where(abs(times[left] - targets) < abs(times[right] - targets), left, right)
            if np.max(abs(times[selected] - targets)) > 0.005:
                raise ValueError(f"Video timestamps do not align with episode {episode}")
            selected_frames = torch.from_numpy(frames[selected]).permute(0, 3, 1, 2)
            if images is None:
                images = torch.empty((len(episodes), *selected_frames.shape[1:]), dtype=torch.uint8)
            images[rows] = selected_frames
        self.images = images

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        indices = index + self.offsets
        valid_indices = indices.clamp(max=self.ends[index].item())
        observation_indices = (index + self.observation_offsets).clamp(min=self.starts[index].item())
        state = self.states[observation_indices]
        image = self.images[observation_indices].float().div(255)
        if self.n_obs_steps == 1:
            state, image = state[0], image[0]
        return {
            "action": self.actions[valid_indices],
            "action_is_pad": indices > self.ends[index],
            "observation.state": state,
            "observation.image": image,
        }


class TrainingPlateau:
    def __init__(self, patience=6, min_relative_improvement=0.01):
        if patience < 1 or not 0 <= min_relative_improvement < 1:
            raise ValueError("Invalid training plateau settings")
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.reference = float("inf")
        self.bad_windows = 0
        self.best = float("inf")

    def update(self, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Training window loss must be finite and nonnegative")
        self.best = min(self.best, value)
        if value < self.reference * (1 - self.min_relative_improvement):
            self.reference, self.bad_windows = value, 0
        else:
            self.bad_windows += 1
        return self.bad_windows >= self.patience


def parse_plateau_args(argv=None):
    parser = argparse.ArgumentParser(description="Continue a checkpoint until training loss plateaus, without validation.")
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lr", type=float, default=None, help="Default: keep source checkpoint's final LR constant")
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--max-additional-steps", type=int, default=50000,
                        help="Safety ceiling; reaching it is reported separately from a plateau")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if min(args.window, args.patience, args.max_additional_steps, args.log_every) < 1:
        parser.error("step counts must be positive")
    if not 0 <= args.min_relative_improvement < 1 or (args.lr is not None and args.lr <= 0):
        parser.error("Invalid relative improvement or learning rate")
    return args


def plateau_main(argv=None):
    args = parse_plateau_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not args.resume and (args.output_dir / "last.pt").exists():
        raise FileExistsError("Choose a new output-dir or use --resume")
    checkpoint = torch.load(args.resume or args.start_checkpoint, map_location="cpu", weights_only=False)
    if not torch.equal(checkpoint["betas"].cpu(), betas.cpu()):
        raise ValueError("Checkpoint and code diffusion schedules differ")
    config = dict(checkpoint["config"])
    amp = config["amp"]
    if amp != "none" and device.type != "cuda":
        raise ValueError("This checkpoint's AMP setting requires CUDA")
    source_step = checkpoint.get("source_step", checkpoint["step"])
    completed = checkpoint.get("additional_steps", 0)
    learning_rate = args.lr if args.lr is not None else checkpoint["optimizer"]["param_groups"][0]["lr"]
    if learning_rate <= 0:
        raise ValueError("Source LR is zero; supply a positive --lr to continue learning")
    plateau_config = dict(window=args.window, patience=args.patience,
                          min_relative_improvement=args.min_relative_improvement,
                          max_additional_steps=args.max_additional_steps, learning_rate=learning_rate)
    if args.resume and checkpoint["plateau_config"] != plateau_config:
        raise ValueError("Resume requires the original plateau settings and LR")
    tracker = TrainingPlateau(args.patience, args.min_relative_improvement)
    if args.resume:
        tracker.__dict__.update(checkpoint["plateau_state"])
    n_obs_steps = checkpoint["model_kwargs"].get("n_obs_steps", 1)
    fps = checkpoint["preprocessing"]["fps"]
    horizon = checkpoint["model_kwargs"]["horizon"]
    stats_cpu = checkpoint["normalization"]
    stats = {key: {name: value.to(device) for name, value in values.items()} for key, values in stats_cpu.items()}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    delta = {"action": [i / fps for i in range(horizon)]}
    if n_obs_steps > 1:
        offsets = [i / fps for i in range(1 - n_obs_steps, 1)]
        delta.update({"observation.state": offsets, "observation.image": offsets})
    # Only load the original training episodes. No validation loader or loss.
    source = LeRobotDataset(config["repo_id"], revision=config["revision"], root=config["dataset_root"],
                            episodes=checkpoint["train_episodes"], delta_timestamps=delta, video_backend="pyav")
    dataset = CachedPushTDataset(source, horizon, {}, n_obs_steps=n_obs_steps)
    loader_args = argparse.Namespace(batch_size=config["batch_size"], workers=0, seed=config["seed"])
    loader = make_loader(dataset, loader_args, True, device)
    model = DiT(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    if device.type == "cuda" and checkpoint["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    loader.generator.set_state(checkpoint["loader_rng"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(config, mode="training_plateau", source_step=source_step,
                      start_checkpoint=str(args.start_checkpoint), output_dir=str(args.output_dir),
                      lr=learning_rate, lr_schedule="constant", validation_used_for_stopping=False,
                      plateau_config=plateau_config)
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(json.dumps({"event": "start", "source_step": source_step, "additional_steps_completed": completed,
                      "training_frames": len(dataset), "n_obs_steps": n_obs_steps,
                      "validation_used": False, **plateau_config}), flush=True)
    train(model, optimizer, scaler, loader, stats, stats_cpu, tracker, checkpoint,
          config, run_config, plateau_config, source_step, completed, amp, device, args)


def train(model, optimizer, scaler, loader, stats, stats_cpu, tracker, source,
          config, run_config, plateau_config, source_step, completed, amp, device, args):
    model.train()
    iterator = iter(loader)
    running = count = window_numerator = window_denominator = 0
    last_window = None
    stopped = False
    best_step = source.get("best_train_step", source_step)
    started = time.monotonic()
    updates = completed
    with (args.output_dir / "metrics.jsonl").open("a") as log:
        for updates in range(completed + 1, args.max_additional_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss, numerator, denominator = noise_loss(model, batch, stats, device, amp)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite continuation loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            count += 1
            window_numerator += numerator.item()
            window_denominator += denominator.item()
            step = source_step + updates
            record = dict(step=step, additional_steps=updates, lr=optimizer.param_groups[0]["lr"])
            if updates % args.log_every == 0 or updates == args.max_additional_steps:
                record.update(train_loss=running / count, elapsed_s=time.monotonic() - started)
                running = count = 0
            check = updates % args.window == 0 or updates == args.max_additional_steps
            improved = False
            if check:
                last_window = window_numerator / max(window_denominator, 1)
                improved = last_window < tracker.best
                stopped = tracker.update(last_window)
                if improved:
                    best_step = step
                record.update(train_window_loss=last_window, best_train_window_loss=tracker.best,
                              plateau_reference=tracker.reference, bad_windows=tracker.bad_windows)
                window_numerator = window_denominator = 0
            if len(record) > 3:
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
            if check:
                payload = dict(
                    step=step, source_step=source_step, additional_steps=updates,
                    model=model.state_dict(), model_kwargs=source["model_kwargs"], optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), config=run_config, normalization=stats_cpu,
                    train_episodes=source["train_episodes"], val_episodes=source["val_episodes"],
                    betas=betas.clone(), preprocessing=source["preprocessing"],
                    train_window_loss=last_window, best_train_loss=tracker.best, best_train_step=best_step,
                    plateau_config=plateau_config, plateau_state=dict(tracker.__dict__),
                    validation_used_for_stopping=False, selection_metric="training_noise_mse_window",
                    loader_rng=loader.generator.get_state(),
                    rng={"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                )
                save_checkpoint(args.output_dir / "last.pt", payload)
                if step % 1000 == 0:
                    save_checkpoint(args.output_dir / f"step_{step:06d}.pt", payload)
                if improved:
                    save_checkpoint(args.output_dir / "best.pt", payload)
            if stopped:
                break
    summary = dict(status="training_plateau" if stopped else "max_additional_steps",
                   step=source_step + updates, source_step=source_step, additional_steps=updates,
                   final_train_window_mse=last_window, best_train_window_mse=tracker.best,
                   best_train_step=best_step, bad_windows=tracker.bad_windows,
                   validation_used_for_stopping=False, plateau_config=plateau_config,
                   evaluation_checkpoint=str(args.output_dir / "last.pt"))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "finished", **summary}), flush=True)



def main(argv=None):
    # Select continuation mode before parsing its independent CLI options.
    mode = argparse.ArgumentParser(add_help=False)
    mode.add_argument("--until-plateau", action="store_true")
    args, remaining = mode.parse_known_args(argv)
    if args.until_plateau:
        plateau_main(remaining)
    else:
        train_main(remaining)


if __name__ == "__main__":
    main()
