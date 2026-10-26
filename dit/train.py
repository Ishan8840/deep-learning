"""Train the conditional DiT on lerobot/pusht. Downloads begin only in main()."""

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
from torch.utils.data import DataLoader

from model import DiT, betas, q_sample


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lerobot/pusht")
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pusht"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
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
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for key in ("steps", "batch_size", "horizon", "hidden_dim", "num_heads",
                "val_batches", "eval_every", "save_every", "log_every"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or not 0 <= args.warmup_steps < args.steps:
        parser.error("workers must be nonnegative and 0 <= warmup-steps < steps")
    if not 0 < args.val_fraction < 1 or args.hidden_dim % args.num_heads:
        parser.error("val-fraction must be in (0, 1); hidden-dim must divide evenly by num-heads")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("lr and grad-clip must be positive; weight-decay must be nonnegative")
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


def main():
    args = parse_args()
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
    if args.resume:
        # Load only your own trusted checkpoints; these include Python/NumPy RNG state.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("repo_id", "revision", "seed", "horizon", "hidden_dim", "num_heads",
                    "batch_size", "lr", "weight_decay", "warmup_steps", "steps", "val_fraction", "amp", "grad_clip"):
            if config[key] != checkpoint["config"][key]:
                raise ValueError(f"Resume requires the original --{key.replace('_', '-')}")

    # Import/instantiate the downloader only when explicitly launching this script.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    print(f"Loading {args.repo_id}@{args.revision}; uncached data will download now.", flush=True)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root, revision=args.revision)
    if metadata.total_episodes < 2:
        raise ValueError("At least two episodes are required for a held-out split")
    if metadata.features["action"]["shape"] != [2] or metadata.features["observation.state"]["shape"] != [2]:
        raise ValueError("This setup expects PushT's 2D state and 2D action")
    episodes = np.random.default_rng(args.seed).permutation(metadata.total_episodes).tolist()
    count = min(metadata.total_episodes - 1, max(1, round(len(episodes) * args.val_fraction)))
    val_episodes, train_episodes = sorted(episodes[:count]), sorted(episodes[count:])
    if checkpoint and (train_episodes != checkpoint["train_episodes"] or val_episodes != checkpoint["val_episodes"]):
        raise ValueError("Dataset episode split has changed since the checkpoint")
    dataset_kwargs = dict(
        repo_id=args.repo_id, root=args.dataset_root, revision=args.revision,
        delta_timestamps={"action": [i / metadata.fps for i in range(args.horizon)]},
        video_backend="pyav",
    )
    train_dataset = LeRobotDataset(episodes=train_episodes, **dataset_kwargs)
    val_dataset = LeRobotDataset(episodes=val_episodes, **dataset_kwargs)
    stats_cpu = checkpoint["normalization"] if checkpoint else fit_normalization(train_dataset)
    stats = {key: {name: value.to(device) for name, value in item.items()} for key, item in stats_cpu.items()}
    train_loader = make_loader(train_dataset, args, True, device)
    val_loader = make_loader(val_dataset, args, False, device)

    model_kwargs = dict(action_dim=2, hidden_dim=args.hidden_dim, horizon=args.horizon,
                        state_dim=2, num_heads=args.num_heads)
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
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
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
            if len(record) > 2:
                print(json.dumps(record), flush=True)
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
            if improved or step % args.save_every == 0 or step == args.steps:
                payload = dict(
                    step=step, best_val=best_val, config=config, model_kwargs=model_kwargs,
                    model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                    normalization=stats_cpu, train_episodes=train_episodes, val_episodes=val_episodes,
                    betas=betas.clone(), loader_rng=train_loader.generator.get_state(),
                    preprocessing={"image_key": "observation.image", "image_normalization": "2*x-1", "fps": metadata.fps},
                    rng={"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                )
                save_checkpoint(args.output_dir / "last.pt", payload)
                if improved:
                    save_checkpoint(args.output_dir / "best.pt", payload)
    print(f"Finished. Checkpoints: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
