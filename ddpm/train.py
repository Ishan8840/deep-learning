"""Train the existing DDPM on MNIST. Training runs only when invoked directly."""

import argparse
import csv
from pathlib import Path

import matplotlib
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ddpm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs, including resumed epochs")
    parser.add_argument("--resume", type=Path, help="Restore model, optimizer, and loss history")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=0.0001)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--coarsen-from-1000", action="store_true",
                        help="Combine equal groups of a 1000-step linear schedule")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=root.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "training")
    parser.add_argument("--download", action="store_true", help="Download MNIST if missing")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or not args.lr > 0:
        parser.error("batch-size, epochs, and lr must be positive")
    if args.timesteps < 2 or not 0 < args.beta_start <= args.beta_end < 1:
        parser.error("timesteps must be >= 2 and 0 < beta-start <= beta-end < 1")
    if args.coarsen_from_1000 and 1000 % args.timesteps != 0:
        parser.error("coarsen-from-1000 requires timesteps to divide 1000")

    device = torch.device(args.device)
    dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=args.download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    # q_sample reads these module-level tensors directly.
    # Build on CPU before moving to the device for reproducible checkpoint checks.
    ddpm.betas = torch.linspace(args.beta_start, args.beta_end, args.timesteps)
    if args.coarsen_from_1000:
        reference_betas = torch.linspace(args.beta_start, args.beta_end, 1000)
        # Products preserve alpha_bar at the end of each original group.
        ddpm.betas = 1 - (1 - reference_betas).reshape(args.timesteps, -1).prod(dim=1)
    ddpm.alphas = 1 - ddpm.betas
    ddpm.alpha_bars = torch.cumprod(ddpm.alphas, dim=0)
    ddpm.betas = ddpm.betas.to(device)
    ddpm.alphas = ddpm.alphas.to(device)
    ddpm.alpha_bars = ddpm.alpha_bars.to(device)
    print(f"Diffusion steps: {args.timesteps}; final alpha_bar: {ddpm.alpha_bars[-1].item():.8f}",
          flush=True)
    model = ddpm.UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if not torch.equal(checkpoint["betas"], ddpm.betas.cpu()):
            raise ValueError("Checkpoint schedule does not match ddpm.py")
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch > args.epochs:
            parser.error("epochs must exceed the checkpoint's completed epoch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        losses = list(checkpoint["losses"])
        print(f"Resuming from epoch {start_epoch - 1} at lr={args.lr}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_images = 0
        for x0, _ in loader:
            x0 = x0.to(device, non_blocking=device.type == "cuda")
            loss = ddpm.train_step(model, optimizer, x0)
            total_loss += loss * x0.size(0)
            total_images += x0.size(0)

        # Weight by batch size so the final, smaller batch counts correctly.
        losses.append(total_loss / total_images)
        print(f"Epoch {epoch}/{args.epochs}: loss={losses[-1]:.6f}", flush=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "betas": ddpm.betas.detach().cpu(),
            "config": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        }, args.output_dir / f"checkpoint_{epoch:03d}.pt")

        with (args.output_dir / "losses.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "training_loss"])
            writer.writerows(enumerate(losses, start=1))

        fig, ax = plt.subplots()
        ax.plot(range(1, epoch + 1), losses, marker="o")
        ax.set(xlabel="Epoch", ylabel="Average training loss (MSE)",
               title="MNIST DDPM training loss")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output_dir / "training_loss.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
