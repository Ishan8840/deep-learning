"""Compare single-step clean estimates across noise levels on one MNIST image."""

import argparse
from pathlib import Path

import matplotlib
import torch
from torchvision import datasets, transforms

import ddpm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=root / "outputs/training/checkpoint_020.pt")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint,
                            map_location="cpu", weights_only=True)
    if not torch.equal(checkpoint["betas"], ddpm.betas.cpu()):
        raise ValueError("Checkpoint schedule does not match ddpm.py")
    ddpm.betas = ddpm.betas.to(device)
    ddpm.alphas = ddpm.alphas.to(device)
    ddpm.alpha_bars = ddpm.alpha_bars.to(device)
    model = ddpm.UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = datasets.MNIST(
        root=root.parent / "data", train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
        ]),
    )
    image, label = dataset[0]
    x0 = image.unsqueeze(0).to(device)
    indices = [100, 300, 500, 700, 999]
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 13), layout="constrained")
    print("Index | Alpha bar | Noise MSE | Noisy image MSE | Clean estimate MSE")
    metrics = ["t,alpha_bar,noise_mse,noisy_image_mse,clean_estimate_mse"]
    for row, index in enumerate(indices):
        t = torch.tensor([index], device=device, dtype=torch.long)
        # Reset the seed to reuse the same noise realization at each noise level.
        torch.manual_seed(0)
        xt, eps = ddpm.q_sample(x0, t)
        eps_pred = model(xt, t)
        alpha_bar = ddpm.alpha_bars[t].view(-1, 1, 1, 1)
        x0_hat = (xt - (1 - alpha_bar).sqrt() * eps_pred) / alpha_bar.sqrt()
        assert torch.isfinite(x0_hat).all()

        values = [alpha_bar.item(), (eps_pred - eps).square().mean().item(),
                  (xt - x0).square().mean().item(), (x0_hat - x0).square().mean().item()]
        print(f"{index:5d} | " + " | ".join(f"{value:.6f}" for value in values))
        metrics.append(",".join(map(str, [index] + values)))
        for ax, tensor, title in zip(
            axes[row], [x0, xt, x0_hat],
            [r"Original $x_0$", r"Noisy $x_t$", r"Estimated $\hat{x}_0$"],
        ):
            # Clamp only the display copy; metrics use the raw estimate.
            display = ((tensor[0, 0].cpu() + 1) / 2).clamp(0, 1)
            ax.imshow(display.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{title} — t={index}")
            ax.axis("off")
    fig.suptitle(f"MNIST test digit {label}: single-step reconstruction at different noise levels")
    output = args.output_dir / "ddpm_denoising_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    output.with_suffix(".csv").write_text("\n".join(metrics) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
