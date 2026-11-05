"""A compact LeNet-style CNN for MNIST."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def loaders(args):
    if args.smoke_test:
        generator = torch.Generator().manual_seed(args.seed)
        images = torch.randn(128, 1, 28, 28, generator=generator)
        labels = torch.randint(0, 10, (128,), generator=generator)
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True), None
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(args.data_dir, train=True, download=args.download, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=args.download, transform=transform)
    return (DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
            DataLoader(test_set, batch_size=args.batch_size))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = count = 0
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1)
        correct += (predictions == labels.to(device)).sum().item()
        count += len(labels)
    return correct / count


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader = loaders(args)
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = nn.functional.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        message = f"epoch {epoch} | train loss {total_loss / len(train_loader.dataset):.4f}"
        if test_loader is not None:
            message += f" | test accuracy {evaluate(model, test_loader, device):.1%}"
        print(message)
    if args.smoke_test:
        print("Synthetic labels verify the training path, not model quality.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(parser.parse_args())
