from typing import Dict, List, Tuple, Union

import torch
import typeguard
from jaxtyping import Float, jaxtyped

from pdeinvbench.utils.types import (
    Type1DRPartialsTuple,
    Type2DRDPartialsTuple,
    TypeBatchSolField1D,
    TypePartials2D,
    TypeXGrid,
    TypeYGrid,
)


@jaxtyped(typechecker=typeguard.typechecked)
def partials_torch_1d_systems(
    solution_field: TypeBatchSolField1D,
    x: TypeXGrid,
    dt: torch.Tensor,
) -> Type1DRPartialsTuple:
    """
    Compute the spatial and temporal partial differentials for 1d systems.
        solution field: solution field
        x: spatial grid
        dt: time differential
    Returns:
        data_x: X spatial gradient (du/dx)
        data_x_usqr: spatial gradient of u^2/2 (du^2/dx)
        data_xx: X spatial second gradient (d^2u/dx^2)
        data_t: temporal gradient (du/dt)
        (All return arguments are same shape as data)
    """
    x_axis = -1
    t_axis = -2

    # take first batch element of spatial and time grids (all the same, artifact of dataloader)
    if len(x.shape) == 2:
        x = x[0]

    data_x = torch.gradient(solution_field, spacing=(x,), dim=x_axis)[0]
    data_x_usqr = torch.gradient(
        solution_field * solution_field / 2, spacing=(x,), dim=x_axis
    )[0]
    data_xx = torch.gradient(data_x, spacing=(x,), dim=x_axis)[0]
    data_t = torch.gradient(solution_field, spacing=dt, dim=t_axis)[0]
    return data_x, data_x_usqr, data_xx, data_t


@jaxtyped(typechecker=typeguard.typechecked)
def partials_torch_2d_systems(
    solution_field: TypePartials2D,
    x: TypeXGrid,
    y: TypeYGrid,
    dt: torch.Tensor,
) -> Type2DRDPartialsTuple:
    """
    Compute the spatial and temporal partial differentials for 2D systems.
        solution_field: solution field
        x, y: spatial grids
        dt: time differential
    Returns:
        data_x: X spatial gradient (du/dx)
        data_y: Y spatial gradient (du/dy)
        data_xx: X spatial second gradient (d^2u/dx^2)
        data_yy: Y spatial second gradient (d^2u/dy^2)
        data_t: temporal gradient (du/dt)
    """

    # take first batch element of spatial grids (all the same, artifact of dataloader)
    if len(x.shape) == 2:
        x = x[0]
    if len(y.shape) == 2:
        y = y[0]
    y_axis = -1
    x_axis = -2
    t_axis = -3
    data_x = torch.gradient(solution_field, spacing=(x,), dim=x_axis)[0]
    data_xx = torch.gradient(data_x, spacing=(x,), dim=x_axis)[0]
    data_y = torch.gradient(solution_field, spacing=(y,), dim=y_axis)[0]
    data_yy = torch.gradient(data_y, spacing=(y,), dim=y_axis)[0]
    data_t = torch.gradient(solution_field, spacing=dt, dim=t_axis)[0]
    return data_x, data_y, data_xx, data_yy, data_t
