import torch
import numpy as np
from jaxtyping import jaxtyped
import typeguard
import typing
from functorch.dim import tree_map
import torch


@jaxtyped(typechecker=typeguard.typechecked)
def torch_to_numpy(tensor: typing.Any) -> np.ndarray | float:
    """
    Convert a torch tensor to a numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


@jaxtyped(typechecker=typeguard.typechecked)
def torch_dict_to_numpy(d: dict) -> dict:
    return tree_map(torch_to_numpy, d)


@jaxtyped(typechecker=typeguard.typechecked)
def compute_grad_norm(model: torch.nn.Module, grads: None) -> float:

    total_norm = 0
    if grads is not None:
        for p in grads:
            param_norm = p.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def is_numpy(x: typing.Any) -> bool:
    return isinstance(x, np.ndarray)
