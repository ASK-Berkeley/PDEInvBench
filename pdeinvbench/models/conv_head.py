import torch
from jaxtyping import Float
from torch import Tensor, nn


def to_binary_mask(x: Float[Tensor, "batch 1 nx ny"]) -> Float[Tensor, "batch 1 nx ny"]:
    """
    converts x into a binary mask using torch exp.
    """
    # return 1 / (1 + torch.exp(-x))
    return nn.functional.sigmoid(x)


class ConvHead(nn.Module):
    """
    Simple convolution head which uses pointwise convolutions to generate a segmentation map.
    The segmentation map is binary. All convolutions are done pointwise (kernel size = 1)
    """

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        n_layers: int,
        dropout: float,
        activation: str,
        out_dim: int = 1,
    ) -> None:
        super(ConvHead, self).__init__()
        activation_fn = None
        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise NotImplementedError(
                f"Activation function not implemented {activation_fn}"
            )
        layers = []
        # Initial layer
        layers.append(nn.Conv2d(in_dim, hidden_dim, 1))
        layers.append(activation_fn)
        if dropout != 0:
            layers.append(nn.Dropout(p=dropout))

        for _ in range(n_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(activation_fn)
            if dropout != 0:
                layers.append(nn.Dropout(p=dropout))

        # output layer
        layers.append(nn.Conv2d(hidden_dim, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "batch channel nx ny"]):
        return to_binary_mask(self.layers(x))
