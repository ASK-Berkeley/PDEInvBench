import pdb
from typing import Dict, Tuple

import jaxtyping
import torch
import typeguard
from jaxtyping import Float, jaxtyped

"""
Set of utility functions for data transformations.
"""


@jaxtyped(typechecker=typeguard.typechecked)
def collapse_time_and_channels(
    x: Float[torch.Tensor, "time channel xspace yspace"],
) -> Float[torch.Tensor, "time*channel xspace yspace"]:
    """
    Collapses the time and channel dimensions of a tensor into a single dimension.
    NOTE: This is only applicable to 2D systems and this is NOT batched!
    We do this to be compatible with FNO. FNO can't handle multiple function outputs
    at once since we're already using the channel dimension to represent time.
    :param x: Input tensor of shape (time, channel, xspace, yspace).
    :return: Output tensor of shape (time*channel, xspace, yspace).
    """
    x_flattened = torch.flatten(x, start_dim=0, end_dim=1)
    return x_flattened


@jaxtyped(typechecker=typeguard.typechecked)
def collapse_time_and_channels_torch_transform(
    batch: Tuple[
        Float[torch.Tensor, "time_n_past in_channels xspace yspace"],
        Dict[
            str, Float[torch.Tensor,
                       "param"] | Float[torch.Tensor, "xspace yspace 1"]
        ],
    ],
) -> Tuple[
    Float[torch.Tensor, "time_n_past*in_channels xspace yspace"],
    Dict[str, Float[torch.Tensor, "param"] |
         Float[torch.Tensor, "xspace yspace 1"]],
]:
    """
    Wrapper for ```collapse_time_and_channels``` to be used with PyTorch's dataloader transforms.
    Accepts a batch and for the first two elements of the batch, collapses the time and channel dimensions.
    :param batch: Tuple of (input, pde_params).
    :return: Tuple of (input, target, pde_params)
    """
    input, pde_params = batch
    input = collapse_time_and_channels(input)
    return input, pde_params


@jaxtyped(typechecker=typeguard.typechecked)
def expand_time_and_channels(
    x: Float[torch.Tensor, "timexchannel xspace yspace"],
    num_channels: int = -1,
    num_timesteps: int = -1,
) -> Float[torch.Tensor, "time channel xspace yspace"]:
    """
    Expands the time and channel dimensions of a tensor into separate dimensions.
    Either number of channels or number of timesteps must be specified.
    NOTE: This is only applicable to 2D systems.
    :param x: Input tensor of shape (time*channel, xspace, yspace).
    :param num_channels: Number of channels to expand to. OPTIONAL if num_timesteps is specified.
    :param num_timesteps: Number of timesteps to expand to. OPTIONAL if num_channels is specified.
    :return: Output tensor of shape (time, channel, xspace, yspace).
    """
    assert (
        num_channels != -1 or num_timesteps != -1
    ), "Either num_channels or num_timesteps must be specified!"
    if num_channels != -1:
        # Case we infer the number of timesteps
        x_unflattened = torch.unflatten(x, 0, (-1, num_channels))
    else:
        # Case we infer the number of channels
        x_unflattened = torch.unflatten(x, 0, (num_timesteps, -1))
    return x_unflattened
