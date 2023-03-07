import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    A simple CNN.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=1,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(
        #     in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        # )
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward.
        """
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        # x = self.conv2(x)
        # x = nn.functional.relu(x)
        # x = self.pool2(x)
        # print(x.shape)
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
