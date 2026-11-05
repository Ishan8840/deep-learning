"""Train the ViT on CIFAR-10.

    python train.py --download
    python train.py --smoke-test --epochs 1
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import Transformer


def make_datasets(args):
    if args.smoke_test:
        # Small, deterministic datasets for exercising the entire training loop.
        generator = torch.Generator().manual_seed(1337)
        return tuple(
            TensorDataset(torch.randn(n, 3, 32, 32, generator=generator),
                          torch.randint(10, (n,), generator=generator))
            for n in (32, 16)
        )

    from torchvision import datasets, transforms

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return (
        datasets.CIFAR10(args.data, train=True, download=args.download,
                         transform=train_transform),
        datasets.CIFAR10(args.data, train=False, download=args.download,
                         transform=test_transform),
    )


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).parent / 'vit.pt')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error('epochs, batch-size, and lr must be positive')

    torch.manual_seed(1337)
    train_data, test_data = make_datasets(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    config = dict(img_size=32, patch=4, in_ch=3, d_model=128,
                  head_dim=32, num_heads=4, num_classes=10)
    model = Transformer(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'device {args.device} | {n_params / 1e6:.2f}M params')
    if args.smoke_test:
        print('Synthetic data: metrics only verify that training runs.')

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, args.device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, args.device)
        print(f'epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.1%} '
              f'| test loss {test_loss:.4f} acc {test_acc:.1%} '
              f'| {time.time() - started:.1f}s')

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config,
                'classes': getattr(train_data, 'classes', list(map(str, range(10)))),
                'optimizer': optimizer.state_dict(), 'epoch': args.epochs,
                'synthetic': args.smoke_test}, args.checkpoint)
    print(f'saved {args.checkpoint}')


if __name__ == '__main__':
    main()
