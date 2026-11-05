from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import save_image
from torch.utils.data import random_split
import torch.nn.functional as F


training_data = datasets.MNIST(
    root=Path(__file__).resolve().parents[2] / "data",
    train=True,
    download=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.MNIST(
    root=Path(__file__).resolve().parents[2] / "data",
    train=False,
    download=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)


train_size = int(0.9 * len(training_data))
val_size = len(training_data) - train_size
train_subset, val_subset = random_split(training_data, [train_size, val_size])

train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU()
            )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        h = self.encoder(x)

        mu = self.mu(h)
        logvar = self.logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std

        x_h = self.decoder(z)

        return x_h, mu, logvar


def kl_loss(mu, logvar):
    kl = 0.5 * torch.sum(
            mu.pow(2) + logvar.exp() - 1 - logvar,
            dim=1
        )

    return kl.mean()


def reconstruction_loss(x_h, x):
    x = x.flatten(start_dim=1)

    return F.binary_cross_entropy(
        x_h, x, reduction="sum"
    ) / x.size(0)

def vae_loss(mu, logvar, x_h, x, beta=1):
    return reconstruction_loss(x_h, x) + beta * kl_loss(mu, logvar)


@torch.inference_mode()
def plot_latent_space(model, dataloader, device, output_path):
    """Plot posterior means for MNIST images, colored by their digit labels."""
    if model.mu.out_features != 2:
        raise ValueError("Latent space visualization requires latent_dim=2")

    was_training = model.training
    model.eval()
    try:
        encodings, labels = [], []
        for images, digit_labels in dataloader:
            h = model.encoder(images.to(device).flatten(start_dim=1))
            # Use the mean to avoid adding noise from latent sampling.
            encodings.append(model.mu(h).cpu())
            labels.append(digit_labels.cpu())
        encodings = torch.cat(encodings).numpy()
        labels = torch.cat(labels).numpy()
    finally:
        model.train(was_training)

    fig = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    scatter = ax.scatter(
        encodings[:, 0], encodings[:, 1], c=labels,
        cmap="tab10", norm=BoundaryNorm([i - 0.5 for i in range(11)], 10),
        s=6, alpha=0.7, linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, ticks=range(10), label="Digit label")
    ax.set(
        title="MNIST test digits in VAE latent space",
        xlabel="Latent dimension 1 (mean)",
        ylabel="Latent dimension 2 (mean)",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved latent space visualization to {output_path}")


@torch.inference_mode()
def plot_latent_interpolation(model, dataset, device, output_path,
                              digit_a=1, digit_b=7, steps=11):
    """Decode a straight line between the means of two test digit encodings."""
    if digit_a == digit_b or not all(0 <= d <= 9 for d in (digit_a, digit_b)):
        raise ValueError("Choose two different digit labels from 0 to 9")
    if steps < 2:
        raise ValueError("Interpolation needs at least two steps")

    # Select the first example of each label for a reproducible comparison.
    examples = {}
    for image, label in dataset:
        if label in (digit_a, digit_b) and label not in examples:
            examples[label] = image
        if len(examples) == 2:
            break
    if len(examples) != 2:
        raise ValueError("Dataset must contain both requested digit labels")
    originals = torch.stack([examples[digit_a], examples[digit_b]])

    was_training = model.training
    model.eval()
    try:
        means = model.mu(model.encoder(originals.to(device).flatten(start_dim=1)))
        t = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
        z = (1 - t) * means[0] + t * means[1]
        decoded = model.decoder(z).reshape(steps, 28, 28).cpu()
    finally:
        model.train(was_training)

    fig = Figure(figsize=(1.2 * (steps + 2), 2.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, steps + 2)
    images = [originals[0, 0], *decoded, originals[1, 0]]
    titles = [f"Input: {digit_a}", *[f"t={v:.1f}" for v in t.cpu().flatten()],
              f"Input: {digit_b}"]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Latent interpolation: {digit_a} → {digit_b} (encoder means)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved latent interpolation to {output_path}")
    return z.cpu(), decoded


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interpolate-only", action="store_true",
                        help="Load autoencoder.pt and plot interpolation without retraining")
    parser.add_argument("--digit-a", type=int, choices=range(10), default=1)
    parser.add_argument("--digit-b", type=int, choices=range(10), default=7)
    args = parser.parse_args()
    if args.digit_a == args.digit_b:
        parser.error("Choose two different digits")

    model = AutoEncoder(latent_dim=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.interpolate_only:
        model.load_state_dict(torch.load("autoencoder.pt", map_location=device,
                                         weights_only=True))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Keep the same eight validation images for every epoch's comparison.
        preview_images = next(iter(val_dataloader))[0][:8].to(device)
        # Sample once so changes across epochs reflect decoder training.
        random_latents = torch.randn(
            len(preview_images), model.decoder[0].in_features, device=device
        )
        preview_dir = Path("reconstructions")
        preview_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(10):
            model.train()
            train_loss = 0.0

            for images, _ in train_dataloader:
                images = images.to(device)
                targets = images.flatten(start_dim=1)  # [batch_size, 784]

                optimizer.zero_grad()
                x_h, mu, logvar = model(images)
                loss = vae_loss(mu, logvar, x_h, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, _ in val_dataloader:
                    images = images.to(device)
                    x_h, mu, logvar = model(images)
                    loss = vae_loss(mu, logvar, x_h, images.flatten(start_dim=1))
                    val_loss += loss.item() * images.size(0)

                preview_reconstructed, _, _ = model(preview_images)
                random_generated = model.decoder(random_latents)

                comparison = torch.cat([
                    preview_images,
                    preview_reconstructed.reshape_as(preview_images),
                    random_generated.reshape_as(preview_images),
                ])

                # Rows: originals, reconstructions, decoded random latent vectors.
                save_image(
                    comparison.cpu(),
                    preview_dir / f"epoch_{epoch + 1:02d}.png",
                    nrow=len(preview_images),
                )

            train_loss /= len(train_dataloader.dataset)
            val_loss /= len(val_dataloader.dataset)

            print(
                f"Epoch {epoch + 1:02d} | "
                f"train loss: {train_loss:.6f} | val loss: {val_loss:.6f}"
            )

        torch.save(model.state_dict(), "autoencoder.pt")
        plot_latent_space(model, test_dataloader, device, "latent_space.png")
    plot_latent_interpolation(model, test_data, device, "latent_interpolation.png",
                              digit_a=args.digit_a, digit_b=args.digit_b)


if __name__ == "__main__":
    main()
