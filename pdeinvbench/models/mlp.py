import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP with a variable number of hidden layers and activation functions.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        dropout: float,
        out_dim: int,
        num_layers: int,
        activation: str,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(in_dim, hidden_size))
        if dropout != 0:
            self.layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            if dropout != 0:
                self.layers.append(nn.Dropout(dropout))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, out_dim))

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation to all but last layer
                x = self.activation(x)
        return x
