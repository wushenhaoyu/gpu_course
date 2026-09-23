import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # 28 -> 24
        self.pool = nn.MaxPool2d(kernel_size=2)       # 24 -> 12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 12 -> 8
                                                     # pool: 8 -> 4

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = LeNet()

x = torch.randn(32, 1, 28, 28)
y = model(x)

print(y.shape)