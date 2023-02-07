import torch
import torch.nn as nn
from typing import Callable


class MLP(torch.nn.Module):
    """
    A multilayer perceptron (MLP) class.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = activation()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        self.layers.append(self.activation)
        for i in range(self.hidden_count):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(self.activation)
        self.layers.append(nn.Dropout(p=0.1))
        self.layers.append(nn.Linear(self.hidden_size, self.num_classes))

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                initializer(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: int) -> int:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = layer(x)
        return x
