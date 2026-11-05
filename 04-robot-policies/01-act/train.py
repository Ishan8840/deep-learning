"""Train ACT on prepared tensors, or run a small synthetic smoke experiment."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from act import ACT, act_loss


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Tensor dictionary saved with torch.save")
    source.add_argument("--synthetic", action="store_true", help="Small toy data for checking the pipeline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0 or args.kl_weight < 0:
        parser.error("epochs, batch-size, threads and lr must be positive; kl-weight must be nonnegative")
    return args, parser


def load_dataset(args, parser):
    """Load aligned observations/action chunks and check their shapes and masks."""
    if args.synthetic:
        qpos = torch.randn(32, 14)
        images = torch.rand(32, 1, 3, 32, 32)
        actions = qpos[:, None].repeat(1, 8, 1) + torch.linspace(0, 0.2, 8)[None, :, None]
        data = dict(qpos=qpos, images=images, actions=actions)
    else:
        data = torch.load(args.data, map_location="cpu", weights_only=True)
    qpos, images, actions = (data[key].float() for key in ("qpos", "images", "actions"))
    is_pad = data.get("is_pad", torch.zeros(actions.shape[:2], dtype=torch.bool))
    if qpos.ndim != 2 or images.ndim != 5 or actions.ndim != 3:
        parser.error("Expected qpos (N,S), images (N,C,3,H,W), actions (N,K,A)")
    if len(qpos) == 0 or not (len(qpos) == len(images) == len(actions)):
        parser.error("Input tensors must have matching nonzero sample counts")
    if is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool:
        parser.error("is_pad must be boolean with shape (N,K)")
    if not (~is_pad).any(dim=1).all():
        parser.error("Each sample must contain at least one valid target")
    if not torch.isfinite(qpos).all() or not torch.isfinite(images).all() or not torch.isfinite(actions[~is_pad]).all():
        parser.error("Inputs and valid actions must be finite")
    if images.min() < 0 or images.max() > 1:
        parser.error("Images must be scaled to [0,1]")
    return TensorDataset(qpos, images, actions, is_pad)


def train_epoch(model, loader, optimizer, device, kl_weight):
    """One pass through the dataset: predict, compute loss, backpropagate, update."""
    model.train()
    totals = dict(loss=0.0, l1=0.0, kl=0.0)
    for qpos, images, actions, is_pad in loader:
        qpos = qpos.to(device)
        images = images.to(device)
        actions = actions.to(device)
        is_pad = is_pad.to(device)

        predicted_actions, mu, logvar = model(qpos, images, actions, is_pad)
        losses = act_loss(predicted_actions, actions, mu, logvar, is_pad, kl_weight)
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("Nonfinite training loss")

        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Weight batch metrics by sample count, including the last smaller batch.
        for name in totals:
            totals[name] += losses[name].item() * len(qpos)
    return {name: total / len(loader.dataset) for name, total in totals.items()}


def main():
    args, parser = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    dataset = load_dataset(args, parser)
    qpos, images, actions, _ = dataset.tensors
    config = dict(
        state_dim=qpos.shape[1],
        action_dim=actions.shape[2],
        chunk_size=actions.shape[1],
        num_cameras=images.shape[1],
        hidden_dim=args.hidden_dim,
    )
    model = ACT(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, loader, optimizer, args.device, args.kl_weight)
        row = dict(epoch=epoch, **metrics)
        history.append(row)
        print(f"Epoch {epoch}: loss={row['loss']:.4f} l1={row['l1']:.4f} kl={row['kl']:.4f}", flush=True)
    # Save architecture settings alongside weights so inference can rebuild ACT.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "synthetic": args.synthetic,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    with (args.output_dir / "losses.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "l1", "kl"])
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
