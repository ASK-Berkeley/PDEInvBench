import importlib
from typing import Callable

import typeguard
from hydra.utils import instantiate
from jaxtyping import Float, jaxtyped
from omegaconf import DictConfig

from pdeinvbench.utils.types import (
    HIGH_RESOLUTION_PDE_SPATIAL_SIZE,
    PDE,
    PDE_SPATIAL_SIZE,
)


@jaxtyped(typechecker=typeguard.typechecked)
def get_function_from_string(string: str) -> Callable:
    """
    Converts a function specified as a string to an actual function object. Used in hydra configs.
    """
    module_name, function_name = string.rsplit(".", 1)
    # Import the module dynamically
    module = importlib.import_module(module_name)
    # Get the function object
    function = getattr(module, function_name)
    return function


def resolve_pde_resolution(cfg: DictConfig) -> None:
    """
    Simple utility method which checks if we are using the high resolution data.
    If we are, it updates the types::PDE_SPATIAL_SIZE dict. Currently only
    works with the inverse setting. Assumes keys cfg:high_resolution[bool] and
    cfg.data.downsample_factor[int] exist.
    """
    assert "high_resolution" in cfg, "No key 'high_resolution' found in hydra config"
    assert (
        "data" in cfg and "downsample_factor" in cfg.data
    ), "No key 'data' or 'data.downsample_factor' found in hydra config"
    high_resolution: bool = cfg.high_resolution
    downsample_factor: int = cfg.data.downsample_factor
    pde: PDE = instantiate(cfg.data.pde)

    if high_resolution:
        assert (
            pde in HIGH_RESOLUTION_PDE_SPATIAL_SIZE
        ), f"Could not find {pde} in high resolution PDE size mapping."

    resolution: list[int] = (
        HIGH_RESOLUTION_PDE_SPATIAL_SIZE[pde]
        if high_resolution
        else PDE_SPATIAL_SIZE[pde]
    )
    if (
        downsample_factor == 0
    ):  # Ensures that dynamic setting works without downsampling
        downsample_factor = 1
    new_resolution: list[float] = [
        res / downsample_factor for res in resolution]
    # only allow downsampling to an integer factor
    for res in new_resolution:
        assert (
            int(res) == res
        ), f"Downsample factor leads to non-integer resolution {res}"

    new_resolution: list[int] = [int(res) for res in new_resolution]
    PDE_SPATIAL_SIZE[pde] = new_resolution


def _resolve_non_uniform_grid(cfg: DictConfig) -> None:
    """
    Simple utility method which resolves the PDE SPATIAL SIZE for the non-uniform grid.
    Assumes keys cfg:data.grid_transform[str] == "non_uniform_mesh" and cfg:data.pde[PDE] exist.
    """
    pde: PDE = instantiate(cfg.data.pde)
    axis = cfg.grid_transform.axis
    drop_ratio = cfg.grid_transform.drop_ratio
    if pde == PDE.KortewegDeVries1D:
        assert axis == "x", "input is 1D, so axis must be x"
    assert 0 <= drop_ratio <= 1, "drop_ratio must be between 0 and 1"
    x_uniform, y_uniform = PDE_SPATIAL_SIZE[pde]
    grid = x_uniform if axis == "x" else y_uniform
    n = grid
    n_drop = int(round(n * drop_ratio))   # 64 * 0.05 = 3.2 -> 3
    n_keep = n - n_drop

    if axis == "x":
        PDE_SPATIAL_SIZE[pde][0] = n_keep
    elif axis == "y":
        PDE_SPATIAL_SIZE[pde][1] = n_keep
    else:
        raise ValueError(f"Invalid axis: {axis}, must be 'x' or 'y'")


def resolve_grid_transform(cfg: DictConfig) -> None:
    """
    Simple utility method which checks if we are doing any grid transform to the pde fields.
    If we are, it updates the types::PDE_SPATIAL_SIZE dict. Currently only
    works with the non_uniform_mesh setting.
    """
    assert "grid_transform" in cfg, "No key 'grid_transform' found in hydra config"
    assert (
        "data" in cfg and "grid_transform" in cfg.data
    ), "No key 'data' or 'data.grid_transform' found in hydra config"

    if cfg.grid_transform is None:
        return None
    grid_transform: str = cfg.grid_transform['_target_']
    if grid_transform == "pdeinvbench.data.grid_transforms.non_uniform_mesh":
        return _resolve_non_uniform_grid(cfg)
    else:
        raise ValueError(
            f"Invalid grid transform: {grid_transform}, must be 'non_uniform_mesh' or None")
