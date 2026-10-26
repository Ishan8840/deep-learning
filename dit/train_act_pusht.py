"""Train the user's ACT model on the same held-out PushT split as the DiT run."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse the sibling ACT implementation without copying or changing its model.
ACT_SOURCE = Path(__file__).resolve().parent.parent / "act" / "act.py"
sys.path.insert(0, str(ACT_SOURCE.parent))
from act import ACT, act_loss
from pusht_cache import CachedPushTDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True,
                        help="DiT checkpoint supplying only dataset split and normalization, not model weights")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pusht"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.hidden_dim, args.chunk_size,
           args.eval_every, args.log_every, args.patience) < 1:
        parser.error("step counts and dimensions must be positive")
    if args.hidden_dim % 4 or not 0 <= args.warmup_steps < args.steps:
        parser.error("hidden-dim must divide by 4, and warmup-steps must be within steps")
    if args.lr <= 0 or args.grad_clip <= 0 or min(args.kl_weight, args.weight_decay, args.min_delta) < 0:
        parser.error("Invalid optimizer/loss/early stopping settings")
    return args


def autocast(args):
    return nullcontext() if args.amp == "none" else torch.autocast("cuda", dtype=torch.bfloat16)


def prepare(batch, stats, device):
    action = batch["action"].to(device, non_blocking=True)
    state = batch["observation.state"].to(device, non_blocking=True)
    for key, value in (("action", action), ("observation.state", state)):
        item = stats[key]
        normalized = 2 * (value - item["min"]) / (item["max"] - item["min"]).clamp_min(1e-6) - 1
        if key == "action":
            action = normalized
        else:
            state = normalized
    # ACT's original trainer expects [0, 1] images, with a camera axis.
    image = batch["observation.image"].to(device, non_blocking=True).unsqueeze(1)
    is_pad = batch["action_is_pad"].to(device, non_blocking=True)
    return state, image, action, is_pad


@torch.inference_mode()
def validate(model, loader, stats, args, device):
    model.eval()
    normalized_error = pixel_error = valid_entries = 0.0
    scale = (stats["action"]["max"] - stats["action"]["min"]).clamp_min(1e-6) * 0.5
    for batch in loader:
        state, image, target, is_pad = prepare(batch, stats, device)
        # No demonstrated actions: ACT uses z=0, exactly as at inference.
        with autocast(args):
            predicted, _, _ = model(state, image)
        error = (predicted.float() - target).abs()
        valid = ~is_pad
        normalized_error += error[valid].sum().item()
        pixel_error += (error * scale)[valid].sum().item()
        valid_entries += valid.sum().item() * target.shape[-1]
    model.train()
    return {"val_l1": normalized_error / valid_entries,
            "val_l1_pixels": pixel_error / valid_entries}


def atomic_save(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("Use --amp none without a BF16-capable CUDA GPU")
    if args.resume is None and (args.output_dir / "last.pt").exists():
        raise FileExistsError("Choose a new output directory or pass --resume")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    reference = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)
    repo_id, revision = reference["config"]["repo_id"], reference["config"]["revision"]
    train_episodes, val_episodes = reference["train_episodes"], reference["val_episodes"]
    stats_cpu = reference["normalization"]
    assert not set(train_episodes) & set(val_episodes)
    del reference
    stats = {key: {name: value.to(device) for name, value in item.items()} for key, item in stats_cpu.items()}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    metadata = LeRobotDatasetMetadata(repo_id, revision=revision)
    kwargs = dict(repo_id=repo_id, revision=revision, video_backend="pyav",
                  delta_timestamps={"action": [i / metadata.fps for i in range(args.chunk_size)]})
    cache = {}
    train = CachedPushTDataset(LeRobotDataset(episodes=train_episodes, **kwargs), args.chunk_size, cache)
    validation = CachedPushTDataset(LeRobotDataset(episodes=val_episodes, **kwargs), args.chunk_size, cache)
    del cache
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=device.type == "cuda", generator=loader_generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(args.seed + 1))
    model_kwargs = dict(state_dim=2, action_dim=2, chunk_size=args.chunk_size,
                        num_cameras=1, hidden_dim=args.hidden_dim, latent_dim=32,
                        nheads=4, encoder_layers=2, decoder_layers=2, dropout=0.1)
    model = ACT(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_multiplier(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / (args.steps - args.warmup_steps)
        return (1 + math.cos(math.pi * min(progress, 1.0))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    start_step = best_step = bad_validations = 0
    best_val = reference_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in config:
            if key not in ("resume", "output_dir", "reference_checkpoint") and config[key] != checkpoint["training_config"][key]:
                raise ValueError(f"Resume requires original {key}")
        if train_episodes != checkpoint["train_episodes"] or val_episodes != checkpoint["val_episodes"]:
            raise ValueError("Episode split changed")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step, best_step = checkpoint["step"], checkpoint["best_step"]
        best_val, reference_val = checkpoint["best_val_l1"], checkpoint["early_stop_reference"]
        bad_validations = checkpoint["bad_validations"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        loader_generator.set_state(checkpoint["loader_rng"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model_hash = hashlib.sha256(ACT_SOURCE.read_bytes()).hexdigest()
    print(json.dumps({"event": "start", "model_kwargs": model_kwargs,
                      "train_episodes": len(train_episodes), "val_episodes": len(val_episodes),
                      "train_frames": len(train), "val_frames": len(validation),
                      "parameters": sum(value.numel() for value in model.parameters()),
                      "selection_metric": "val_l1 (no actions supplied, z=0)", **config}), flush=True)
    model.train()
    iterator = iter(train_loader)
    totals, count, started = {name: 0.0 for name in ("loss", "l1", "kl")}, 0, time.monotonic()
    stopped_early, step = False, start_step
    with (args.output_dir / "metrics.jsonl").open("a") as log:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            state, image, target, is_pad = prepare(batch, stats, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                predicted, mu, logvar = model(state, image, target, is_pad)
            # Keep KL and reconstruction loss in FP32 even with BF16 forward ops.
            losses = act_loss(predicted.float(), target, mu.float(), logvar.float(), is_pad, args.kl_weight)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError(f"Nonfinite ACT loss at step {step}")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            count += 1
            for name in totals:
                totals[name] += losses[name].item()
            record = {"step": step, "lr": optimizer.param_groups[0]["lr"]}
            if step % args.log_every == 0 or step == args.steps:
                record.update({f"train_{name}": value / count for name, value in totals.items()})
                record["elapsed_s"] = time.monotonic() - started
                totals, count = {name: 0.0 for name in totals}, 0
            improved = False
            if step % args.eval_every == 0 or step == args.steps:
                val = validate(model, val_loader, stats, args, device)
                if not all(math.isfinite(value) for value in val.values()):
                    raise FloatingPointError("Nonfinite ACT validation loss")
                improved = val["val_l1"] < best_val
                if improved:
                    best_val, best_step = val["val_l1"], step
                if val["val_l1"] < reference_val - args.min_delta:
                    reference_val, bad_validations = val["val_l1"], 0
                else:
                    bad_validations += 1
                stopped_early = bad_validations >= args.patience
                record.update(val, best_val_l1=best_val, best_step=best_step, bad_validations=bad_validations)
            if len(record) > 2:
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
            if step % args.eval_every == 0 or step == args.steps:
                payload = dict(
                    model_state_dict=model.state_dict(), config=model_kwargs,
                    optimizer_state_dict=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    step=step, best_step=best_step, best_val_l1=best_val,
                    early_stop_reference=reference_val, bad_validations=bad_validations,
                    training_config=config, normalization=stats_cpu,
                    train_episodes=train_episodes, val_episodes=val_episodes,
                    preprocessing={"image_normalization": "x/255", "image_shape": [96, 96, 3],
                                   "camera_axis": True, "fps": metadata.fps},
                    dataset={"repo_id": repo_id, "revision": revision}, act_source_sha256=model_hash,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    loader_rng=loader_generator.get_state(),
                )
                atomic_save(args.output_dir / "last.pt", payload)
                if improved:
                    atomic_save(args.output_dir / "best.pt", payload)
            if stopped_early:
                print(f"Early stopping at {step}; best prior-action L1 {best_val:.6f} at {best_step}", flush=True)
                break
    summary = dict(status="early_stopped" if stopped_early else "completed", step=step,
                   best_step=best_step, best_val_l1=best_val,
                   selection_metric="held-out normalized action L1 with z=0 and no demonstrated actions",
                   best_checkpoint=str(args.output_dir / "best.pt"))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "finished", **summary}), flush=True)


if __name__ == "__main__":
    main()
