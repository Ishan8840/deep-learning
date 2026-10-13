from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import save_image
from torch.utils.data import random_split
import torch.nn.functional as F


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)


train_size = int(0.9 * len(training_data))
val_size = len(training_data) - train_size
train_subset, val_subset = random_split(training_data, [train_size, val_size])

train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU()
            )
        
        self.mu = nn.Linear(256, 32)
        self.logvar = nn.Linear(256, 32)

        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
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

def vae_loss(mu, logvar, x_h, x, beta=4):
    return reconstruction_loss(x_h, x) + beta * kl_loss(mu, logvar)



model = AutoEncoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
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
