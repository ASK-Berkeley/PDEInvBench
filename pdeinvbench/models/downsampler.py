import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, Linear, ReLU

from pdeinvbench.utils.types import PDE


class ConvDownsampler(nn.Module):
    """
    Multi-layer convolutional downsampler for spatial dimension reduction.

    Stacks multiple ConvBlock layers, each consisting of a convolutional layer,
    ReLU activation, dropout, and max pooling. Supports both 1D and 2D spatial
    dimensions.

    Parameters
    ----------
    n_layers : int
        Number of convolutional blocks to stack.
    input_dimension : int
        Spatial dimensionality of the input (1 for 1D, 2 for 2D).
        Determines whether to use Conv1d or Conv2d operations.
    in_channels : int
        Number of input channels. Note: this stays constant across all layers
        in the current implementation.
    out_channels : int
        Number of output channels for each convolutional layer.
    kernel_size : int
        Size of the convolving kernel.
    stride : int
        Stride of the convolution operation.
    padding : int
        Zero-padding added to both sides of the input.
    dropout : float
        Dropout probability applied after each ReLU activation.
    """

    def __init__(
        self,
        n_layers: int,
        input_dimension: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dropout: float,
    ):
        super(ConvDownsampler, self).__init__()

        self.n_layers = n_layers
        self.input_dimension = input_dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                ConvBlock(
                    input_dimension,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dropout,
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlock(nn.Module):
    """
    Conv block with a convolutional layer, ReLU activation and maxpooling.
    """

    def __init__(
        self,
        input_dimension: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dropout: float,
    ):
        super(ConvBlock, self).__init__()

        if input_dimension == 2:
            self.conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(padding, padding),
            )
            self.maxpool = nn.MaxPool2d(2)
        elif input_dimension == 1:
            self.conv = Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.maxpool = nn.MaxPool1d(2)
        else:
            raise ValueError("Input dimension must be 1 or 2.")

        self.relu = ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.maxpool(x)
        return x


class IdentityMap(nn.Module):
    """
    Identity downsampler to use for darcy flow. Since Fno maps to function spaces,
    there is no spatial downsampling that needs to be done.
    """

    def __init__(self, **kwargs):
        super(IdentityMap, self).__init__()
        # Ignore all kwargs - IdentityMap doesn't need any configuration

    def forward(self, x):
        return x
