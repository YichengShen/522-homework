import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    A simple CNN.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        num_features = 16
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_features,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_features),
            # nn.Conv2d(
            #     in_channels=num_features,
            #     out_channels=num_features,
            #     kernel_size=3,
            #     stride=2,
            #     padding=0,
            # ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(num_features=num_features),
            # nn.Conv2d(
            #     in_channels=num_features,
            #     out_channels=num_features,
            #     kernel_size=3,
            #     stride=2,
            #     padding=0,
            # ),
        )
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=num_features, out_features=num_classes)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=16 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward.
        """
        x = self.cnn(x)
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
