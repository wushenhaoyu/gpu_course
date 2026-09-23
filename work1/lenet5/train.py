"""Minimal MNIST trainer for the clean LeNet-5 baseline."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            correct += (logits.argmax(1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--save", type=Path, default=Path("lenet5_mnist.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(args.data, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=1024, pin_memory=device.type == "cuda")

    model = LeNet5().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            samples += labels.numel()
        print(f"epoch {epoch:2d}: loss={loss_sum / samples:.4f}, test_acc={evaluate(model, test_loader, device):.2%}")

    torch.save(model.state_dict(), args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
