"""A clean LeNet-5-style network for 1x28x28 MNIST inputs."""

import torch
from torch import nn


class LeNet5(nn.Module):
    """1→6→16→120→84→10 with Tanh and 2x2 average pooling."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),       # 28 -> 24
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 24 -> 12
            nn.Conv2d(6, 16, kernel_size=5),       # 12 -> 8
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 8 -> 4
            nn.Conv2d(16, 120, kernel_size=4),     # 4 -> 1
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
            raise ValueError(f"expected [B, 1, 28, 28], got {tuple(x.shape)}")
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = LeNet5()
    x = torch.randn(4, 1, 28, 28)
    print(model)
    print("output shape:", tuple(model(x).shape))
