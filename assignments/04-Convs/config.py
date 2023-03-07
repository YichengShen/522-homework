from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize


class CONFIG:
    batch_size = 128
    num_epochs = 10

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)

    transforms = Compose(
        [
            ToTensor(),
            Normalize(
                [0.49137255, 0.48235294, 0.44666667],
                [0.24705882, 0.24352941, 0.26156863],
            ),
        ]
    )
