import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    A simple CNN.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=0
            ),
            nn.Flatten(),
            nn.Linear(in_features=32 * 7 * 7, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward.
        """
        # print(summary(self.cnn, (3, 32, 32)))
        return self.cnn(x)
