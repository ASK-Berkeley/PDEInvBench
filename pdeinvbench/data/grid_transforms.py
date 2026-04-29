from jaxtyping import Float, jaxtyped
import torch
import typeguard
from typing import Optional, Union


@jaxtyped(typechecker=typeguard.typechecked)
def non_uniform_mesh(
    u: Union[Float[torch.Tensor, "timexchannel xspace yspace"], Float[torch.Tensor, "timexchannel xspace"]],
    spatial_grid: tuple[Float[torch.Tensor, "xspace"], Optional[Float[torch.Tensor, "yspace"]]],
    axis: str = "x",
    drop_ratio: float = 0.05
) -> tuple[Float[torch.Tensor, "timexchannel ..."], tuple[Float[torch.Tensor, "..."], Optional[Float[torch.Tensor, "..."]]]]:
    """
    Create a non-uniform mesh for the data by uniformly sampling random drop_ratio of the coordinates along a given axis.
    Args:
        u: The data to create a non-uniform mesh for.
        spatial_grid: The uniform mesh for the x-axis and y-axis.
        axis: The axis to create a non-uniform mesh for.
        drop_ratio: The ratio of points to drop from the uniform mesh.
    Returns:
        Data on a non-uniform mesh, the new spatial grid.
    """

    assert axis in ["x", "y"], "Invalid axis"
    assert 0 <= drop_ratio <= 1, "drop_ratio must be between 0 and 1"

    x_uniform, y_uniform = spatial_grid
    if drop_ratio == 0:
        return u, spatial_grid, non_uniform_mask

    if y_uniform is None:
        assert axis == "x", "input is 1D, so axis must be x"

    grid = x_uniform if axis == "x" else y_uniform
    n = grid.shape[-1]
    n_drop = int(round(n * drop_ratio))   # 64 * 0.05 = 3.2 -> 3
    n_keep = n - n_drop

    # get random indices to drop
    non_uniform_mask = torch.randperm(n, device=x_uniform.device)

    # dropping drop_ratio of the grid points
    non_uniform_mask = non_uniform_mask[:n_keep]

    # sort them so original order is preserved
    non_uniform_mask = non_uniform_mask.sort().values

    x_non_uniform = x_uniform
    y_non_uniform = y_uniform
    if axis == "x":
        u_non_uniform = u[:, non_uniform_mask]
        x_non_uniform = x_uniform.index_select(dim=0, index=non_uniform_mask)
    else:
        u_non_uniform = u[:, :, non_uniform_mask]
        y_non_uniform = y_uniform.index_select(dim=0, index=non_uniform_mask)

    return u_non_uniform, (x_non_uniform, y_non_uniform)
