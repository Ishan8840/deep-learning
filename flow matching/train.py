"""Train on LeRobot PushT. Run `python train.py --help` for options.

Requires torch, torchvision, numpy, huggingface-hub, pyarrow, and av.
Simulator evaluation also requires gym-pusht==0.1.5 and pymunk<7.
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from model import FM


REPO_ID = "lerobot/pusht"
REVISION = "a16737ffb6c8dbe935291b9fd2f36d14c9bf0820"  # v2.0
TOTAL_EPISODES = 206


def load_episode(root, episode_id):
    """Download once, align video to timestamps, cache compact uint8 tensors."""
    import av
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    root = Path(root) / REVISION
    cached = root / f"episode_{episode_id:06d}.pt"
    if cached.exists():
        return torch.load(cached, map_location="cpu", weights_only=True)
    root.mkdir(parents=True, exist_ok=True)

    def fetch(filename):
        return hf_hub_download(REPO_ID, filename, repo_type="dataset",
                               revision=REVISION, cache_dir=root / "hub")

    name = f"episode_{episode_id:06d}"
    table = pq.read_table(fetch(f"data/chunk-000/{name}.parquet"))
    qpos = torch.tensor(table["observation.state"].to_pylist(), dtype=torch.float32)
    actions = torch.tensor(table["action"].to_pylist(), dtype=torch.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist())
    frames, frame_times = [], []
    with av.open(fetch(f"videos/chunk-000/observation.image/{name}.mp4")) as video:
        for frame in video.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            frame_times.append(float(frame.time))
    if not frames:
        raise ValueError(f"Episode {episode_id} has no video frames")
    distances = np.abs(timestamps[:, None] - np.asarray(frame_times)[None, :])
    indices = distances.argmin(axis=1)
    if distances[np.arange(len(timestamps)), indices].max() > 0.02:
        raise ValueError(f"Video/state timestamps are misaligned in episode {episode_id}")
    images = torch.from_numpy(np.stack(frames)[indices]).permute(0, 3, 1, 2).contiguous()
    if qpos.shape != actions.shape or qpos.shape != (len(images), 2):
        raise ValueError(f"Unexpected PushT shapes in episode {episode_id}")
    if not torch.isfinite(qpos).all() or not torch.isfinite(actions).all():
        raise ValueError(f"Nonfinite positions in episode {episode_id}")
    episode = dict(qpos=qpos, images=images, actions=actions)
    temporary = cached.with_suffix(".tmp")
    torch.save(episode, temporary)
    temporary.replace(cached)
    return episode


def load_episodes(root, ids):
    ids = list(ids)
    if not ids or any(i < 0 or i >= TOTAL_EPISODES for i in ids):
        raise ValueError("Episode IDs must be in [0, 205]")
    print(f"Loading {len(ids)} PushT episodes into RAM (RGB stored as uint8)", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        episodes = list(pool.map(lambda i: load_episode(root, i), ids))
    return dict(zip(ids, episodes))


def split_episodes(count=TOTAL_EPISODES, val_fraction=0.1, seed=0):
    if not 2 <= count <= TOTAL_EPISODES or not 0 < val_fraction < 1:
        raise ValueError("Need 2..206 episodes and 0 < val_fraction < 1")
    ids = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    nval = min(count - 1, max(1, round(count * val_fraction)))
    return sorted(ids[nval:]), sorted(ids[:nval])


def fit_stats(episodes):
    """Fit on training episodes only; count each frame once."""
    stats = {}
    for source, prefix in (("qpos", "qpos"), ("actions", "action")):
        values = torch.cat([episode[source] for episode in episodes])
        stats[f"{prefix}_mean"] = values.mean(0)
        stats[f"{prefix}_std"] = values.std(0, unbiased=False).clamp_min(1e-6)
    return stats


class PushTDataset(Dataset):
    def __init__(self, episodes, chunk_size, stats, obs_steps=1):
        if chunk_size < 1 or obs_steps < 1 or not episodes:
            raise ValueError("Need episodes and positive chunk_size/obs_steps")
        self.episodes = list(episodes)
        self.chunk_size = chunk_size
        self.stats = stats
        self.obs_steps = obs_steps
        self.index = [(i, t) for i, ep in enumerate(self.episodes) for t in range(len(ep["qpos"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode_id, frame = self.index[index]
        episode = self.episodes[episode_id]
        indices = torch.arange(frame, frame + self.chunk_size)
        is_pad = indices >= len(episode["actions"])
        actions = episode["actions"][indices.clamp_max(len(episode["actions"]) - 1)]
        if self.obs_steps == 1:
            qpos = episode["qpos"][frame]
            images = episode["images"][frame].unsqueeze(0)
        else:
            history = torch.arange(frame - self.obs_steps + 1, frame + 1).clamp_min(0)
            qpos = episode["qpos"][history]
            images = episode["images"][history].unsqueeze(1)
        return {
            "qpos": (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"],
            "images": images.float().div(255),
            "actions": (actions - self.stats["action_mean"]) / self.stats["action_std"],
            "is_pad": is_pad,
        }


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(model, loader, device, max_batches=0, num_steps=0, stats=None):
    """Fixed-noise validation; optional sampled-action MSE in raw pixel units."""
    from inference import sample_actions
    from model import masked_mse

    was_training = model.training
    model.eval()
    total, count, action_total = 0.0, 0, 0.0
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(12345)
            sample_generator = torch.Generator(device=device).manual_seed(12346)
            for index, batch in enumerate(loader):
                if max_batches and index >= max_batches:
                    break
                batch = to_device(batch, device)
                weight = (~batch["is_pad"]).sum().item() * model.action_dim
                loss = model.fm_loss(batch["actions"], batch["qpos"], batch["images"], batch["is_pad"])
                total += loss.item() * weight
                count += weight
                if num_steps:
                    prediction = sample_actions(model, batch["qpos"], batch["images"], num_steps,
                                                generator=sample_generator)
                    scale = stats["action_std"].to(device)
                    mse = masked_mse(prediction * scale, batch["actions"] * scale, batch["is_pad"])
                    action_total += mse.item() * weight
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("Validation loader is empty")
    metrics = {"flow_mse": total / count}
    if num_steps:
        metrics["action_mse_pixels"] = action_total / count
    return metrics


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pusht"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pusht"))
    parser.add_argument("--episodes", type=int, default=206)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2, help="Default depth for both transformers")
    parser.add_argument("--encoder-layers", type=int, help="Override observation encoder depth")
    parser.add_argument("--decoder-layers", type=int, help="Override action decoder depth")
    parser.add_argument("--backbone", choices=["cnn", "resnet18"], default="cnn")
    parser.add_argument("--obs-steps", type=int, choices=[1, 2], default=1)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 = full epoch")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 = full validation")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--restart-lr-schedule", action="store_true",
                        help="On resume, start a new cosine decay from --lr over the remaining updates")
    parser.add_argument("--max-steps", type=int, default=0, help="Total optimizer updates; overrides --epochs when positive")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    args.encoder_layers = args.layers if args.encoder_layers is None else args.encoder_layers
    args.decoder_layers = args.layers if args.decoder_layers is None else args.decoder_layers
    if min(args.epochs, args.batch_size, args.chunk_size, args.threads, args.layers) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, chunk-size, threads, layers and lr must be positive")
    if min(args.encoder_layers, args.decoder_layers) < 1:
        parser.error("encoder-layers and decoder-layers must be positive")
    if min(args.max_train_batches, args.max_val_batches, args.weight_decay) < 0:
        parser.error("batch limits and weight decay must be nonnegative")
    if args.max_steps < 0 or min(args.save_every, args.log_every) < 1:
        parser.error("max-steps must be nonnegative; save-every and log-every must be positive")
    if not 0 <= args.ema_decay < 1:
        parser.error("ema-decay must be in [0, 1)")
    if args.restart_lr_schedule and (not args.resume or not args.max_steps):
        parser.error("restart-lr-schedule requires --resume and a positive --max-steps")
    return args


def make_scheduler(args, optimizer, previous, global_step, batches_per_epoch):
    schedule_start_step = previous.get("schedule_start_step", 0) if previous else 0
    if args.restart_lr_schedule:
        if args.lr_schedule != "cosine":
            raise ValueError("restart-lr-schedule requires a cosine checkpoint")
        schedule_start_step = global_step
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = args.lr
    schedule_steps = (args.max_steps or args.epochs * batches_per_epoch) - schedule_start_step
    scheduler = None
    if args.lr_schedule == "cosine":
        if previous and not args.restart_lr_schedule and previous.get("schedule_steps") != schedule_steps:
            raise ValueError("Keep the original max-steps, or explicitly use --restart-lr-schedule")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_steps, eta_min=0.0)
        if previous and not args.restart_lr_schedule:
            scheduler.load_state_dict(previous["scheduler_state_dict"])
    return scheduler, schedule_start_step, schedule_steps


def make_ema(model, args, previous):
    ema_model = None
    if args.ema:
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)
        if previous:
            ema_model.module.load_state_dict(previous["ema_model_state_dict"])
            ema_model.n_averaged.fill_(previous["ema_updates"])
        ema_model.eval()

    return ema_model


def train_step(model, batch, optimizer, scheduler, ema_model):
    optimizer.zero_grad(set_to_none=True)
    loss = model.fm_loss(batch["actions"], batch["qpos"], batch["images"], batch["is_pad"])
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    update_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema_model is not None:
        ema_model.update_parameters(model)
    return loss.item(), update_lr


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output_dir / "last.pt").exists():
        raise ValueError("Output already contains a run; use --resume or a different --output-dir")
    previous = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if previous:
        if previous["dataset"]["revision"] != REVISION:
            raise ValueError("Checkpoint dataset revision differs from loader")
        train_ids, val_ids = previous["train_episodes"], previous["val_episodes"]
        args.seed = previous["seed"]
        args.ema = previous.get("ema_model_state_dict") is not None
        args.ema_decay = previous.get("ema_decay", 0.999)
        args.lr_schedule = previous.get("lr_schedule", "constant")
    else:
        train_ids, val_ids = split_episodes(args.episodes, args.val_fraction, args.seed)
    episodes = load_episodes(args.data_dir, sorted(train_ids + val_ids))
    stats = previous["stats"] if previous else fit_stats([episodes[i] for i in train_ids])
    config = previous["config"] if previous else dict(
        state_dim=2, action_dim=2, num_cameras=1, chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        nheads=4, dropout=0.1,
        backbone=args.backbone, obs_steps=args.obs_steps,
    )
    model = FM(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best = 1, float("inf")
    global_step, resume_batch = 0, 0
    if previous:
        model.load_state_dict(previous["model_state_dict"])
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        start_epoch, best = previous["epoch"] + 1, previous["best_val_loss"]
        global_step = previous.get("global_step", max(
            (int(state["step"]) for state in previous["optimizer_state_dict"]["state"].values()), default=0))
        if not previous.get("epoch_complete", True):
            start_epoch = previous["epoch"]
            resume_batch = previous["batch_in_epoch"]
        torch.set_rng_state(previous["rng_state"])
        if torch.device(args.device).type == "cuda" and previous.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng_state"])
    if not args.max_steps and start_epoch > args.epochs:
        raise ValueError("--epochs is the total target and must exceed the resumed epoch")
    if args.max_steps and global_step >= args.max_steps:
        raise ValueError("Checkpoint already reached --max-steps; choose a larger total target")
    train_data = PushTDataset([episodes[i] for i in train_ids], config["chunk_size"], stats, model.obs_steps)
    val_data = PushTDataset([episodes[i] for i in val_ids], config["chunk_size"], stats, model.obs_steps)
    generator = torch.Generator()
    pin = torch.device(args.device).type == "cuda"
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, pin_memory=pin)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, pin_memory=pin)
    batches_per_epoch = min(len(train_loader), args.max_train_batches or len(train_loader))
    if args.max_steps:
        args.epochs = start_epoch - 1 + math.ceil((args.max_steps - global_step + resume_batch) / batches_per_epoch)
    scheduler, schedule_start_step, schedule_steps = make_scheduler(
        args, optimizer, previous, global_step, batches_per_epoch,
    )
    ema_model = make_ema(model, args, previous)

    def make_checkpoint(epoch, batch_in_epoch, epoch_complete):
        return dict(
            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            config=config, stats=stats, epoch=epoch, best_val_loss=best, seed=args.seed,
            global_step=global_step, batch_in_epoch=batch_in_epoch, epoch_complete=epoch_complete,
            ema_model_state_dict=ema_model.module.state_dict() if ema_model is not None else None,
            ema_updates=int(ema_model.n_averaged) if ema_model is not None else 0,
            ema_decay=args.ema_decay,
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
            lr_schedule=args.lr_schedule, schedule_steps=schedule_steps,
            schedule_start_step=schedule_start_step,
            train_episodes=train_ids, val_episodes=val_ids,
            dataset=dict(repo_id=REPO_ID, revision=REVISION),
            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if pin else None,
            train_args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )

    def append_metrics(filename, row):
        with (args.output_dir / filename).open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    print(f"Train: {len(train_ids)} episodes/{len(train_data)} frames; "
          f"validation: {len(val_ids)} episodes/{len(val_data)} frames; device={args.device}; "
          f"step={global_step}; target_steps={args.max_steps or 'epoch-based'}", flush=True)
    log_total, log_count, log_updates = 0.0, 0, 0
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.monotonic()
        generator.manual_seed(args.seed + epoch)
        model.train()
        total, count = 0.0, 0
        batch_in_epoch = resume_batch if epoch == start_epoch else 0
        for index, batch in enumerate(train_loader):
            if index >= batches_per_epoch:
                break
            if epoch == start_epoch and index < resume_batch:
                continue
            batch = to_device(batch, args.device)
            loss, update_lr = train_step(model, batch, optimizer, scheduler, ema_model)
            global_step += 1
            batch_in_epoch = index + 1
            weight = (~batch["is_pad"]).sum().item() * model.action_dim
            total += loss * weight
            count += weight
            log_total += loss * weight
            log_count += weight
            log_updates += 1
            if global_step % args.log_every == 0 or global_step == args.max_steps:
                append_metrics("steps.jsonl", dict(step=global_step, epoch=epoch,
                    train_loss=log_total / log_count, updates=log_updates,
                    lr=update_lr))
                log_total, log_count, log_updates = 0.0, 0, 0
            if global_step % args.save_every == 0:
                checkpoint = make_checkpoint(epoch, batch_in_epoch, batch_in_epoch >= batches_per_epoch)
                save_checkpoint(args.output_dir / f"step_{global_step:06d}.pt", checkpoint)
                save_checkpoint(args.output_dir / "last.pt", checkpoint)
            if args.max_steps and global_step >= args.max_steps:
                break
        evaluation_model = ema_model.module if ema_model is not None else model
        val = validate(evaluation_model, val_loader, args.device, args.max_val_batches)
        if not torch.isfinite(torch.tensor(val["flow_mse"])):
            raise RuntimeError("Nonfinite validation loss")
        improved = val["flow_mse"] < best
        best = min(best, val["flow_mse"])
        row = dict(epoch=epoch, step=global_step, train_loss=total / count, val_loss=val["flow_mse"],
                   seconds=time.monotonic() - started)
        if ema_model is not None:
            row["raw_val_loss"] = validate(model, val_loader, args.device, args.max_val_batches)["flow_mse"]
            row["seconds"] = time.monotonic() - started
        checkpoint = make_checkpoint(epoch, batch_in_epoch, batch_in_epoch >= batches_per_epoch)
        save_checkpoint(args.output_dir / "last.pt", checkpoint)
        if global_step % args.save_every == 0 or global_step == args.max_steps:
            save_checkpoint(args.output_dir / f"step_{global_step:06d}.pt", checkpoint)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", checkpoint)
        append_metrics("metrics.jsonl", row)
        if args.max_steps and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
