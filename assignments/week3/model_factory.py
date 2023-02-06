import torch
from model import MLP
import os

os.system("pip install pyyaml")
import yaml


def create_model(input_dim: int, output_dim: int) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (list): The dimensions of the hidden layers.

    Returns:
        MLP: The created model.

    """
    with open("config.yaml", "r") as stream:
        d = yaml.safe_load(stream)
        print(d)

    return MLP(
        input_dim,
        d["size"],
        output_dim,
        d["layers"],
        torch.nn.Mish,
        torch.nn.init.kaiming_normal_,
    )
