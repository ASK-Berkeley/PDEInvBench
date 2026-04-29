from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import typeguard
from jaxtyping import Float, jaxtyped
from numpy import spacing
from torch import Tensor

from pdeinvbench.losses.finite_differences import (
    partials_torch_1d_systems,
    partials_torch_2d_systems,
)
from pdeinvbench.losses.fluids import navier_stokes_residual, turbulent_flow_residual
from pdeinvbench.utils.types import (
    PDE,
    Type1DKDVPartialsReturnType,
    Type1DRDPartialsReturnType,
    Type2DRDPartialsReturnType,
    TypeAdvectionPartialsReturnType,
    TypeBatchSolField1D,
    TypeBatchSolField2D,
    TypeBurgersPartialsReturnType,
    TypeNSPartials2D,
    TypeParam,
    TypePartials1D,
    TypePartials2D,
    TypeTimeGrid,
    TypeUnBatchedNSPartials2D,
    TypeUnBatchedNSResiduals2D,
    TypeUnBatchSolField2D,
    TypeXGrid,
    TypeYGrid,
)


@jaxtyped(typechecker=typeguard.typechecked)
def get_pde_residual_function(pde_name: PDE) -> Callable:
    """
    Get PDE residual function for the given pde
    """

    if pde_name == PDE.ReactionDiffusion2D:
        return reaction_diff_2d_residual_compute

    if pde_name == PDE.NavierStokes2D:
        return navier_stokes_2d_residual_compute

    if pde_name == PDE.TurbulentFlow2D:
        return turbulent_flow_2d_residual_compute

    if pde_name == PDE.KortewegDeVries1D:
        return kdv_1d_residual_compute

    if pde_name == PDE.DarcyFlow2D:
        return darcy_flow_2d_residual_compute

    raise ValueError(f"Unknown PDE type: {pde_name}. No suitable residual function.")


@jaxtyped(typechecker=typeguard.typechecked)
def reaction_diff_2d_residual_compute(
    solution_field: TypeBatchSolField2D,
    pde_params: TypeParam,
    spatial_grid: List[Union[TypeXGrid, TypeYGrid]],
    t: TypeTimeGrid,
    return_partials=False,
) -> Type2DRDPartialsReturnType:
    """
    Compute the PDE residual for 2D Reaction Diffusion.
    R_u = u - u^3 - k - v
    R_v = u - v
    Eqn 1: du/dt = D_u * d^2u/dx^2 + D_u * d^2u/dy^2 + R_u
    Eqn 2: dv/dt = D_v * d^2v/dx^2 + D_v * d^2v/dy^2 + R_v
    args:
        solution_field: solution_field of 2D Reaction Diffusion
        x,y : spatial grids
        t: temporal grid
        k, du, dv: 2d Reaction Diffusion parameters
        return_partials: Flag to return partial derivatives in 2D Reaction Diffusion equation
    """
    u = solution_field[:, :, 0]
    v = solution_field[:, :, 1]
    k, du, dv = pde_params["k"], pde_params["Du"], pde_params["Dv"]
    x, y = spatial_grid
    if len(t.shape) == 2:
        dt = t[0, 1] - t[0, 0]
    else:
        dt = t[1] - t[0]

    u_x, u_y, u_xx, u_yy, u_t = partials_torch_2d_systems(u, x, y, dt)
    v_x, v_y, v_xx, v_yy, v_t = partials_torch_2d_systems(v, x, y, dt)

    # batch x time x space x space
    # No channel dimension since we extracted U, V out of channels
    k = torch.reshape(k, (-1, 1, 1, 1))
    du = torch.reshape(du, (-1, 1, 1, 1))
    dv = torch.reshape(dv, (-1, 1, 1, 1))

    # 2d reaction diffusion equations
    ru = u - (u**3) - k - v
    rv = u - v
    eqn1 = du * u_xx + du * u_yy + ru - u_t
    eqn2 = dv * v_xx + dv * v_yy + rv - v_t

    # Expand both equations to have a channel dimension we concatenate them along
    pde_residual = torch.stack([eqn1, eqn2], dim=2)
    if return_partials:
        u_partials = torch.cat([u_x, u_y, u_xx, u_yy, u_t], dim=1)
        v_partials = torch.cat([v_x, v_y, v_xx, v_yy, v_t], dim=1)
        partials = torch.stack([u_partials, v_partials], dim=2)
        return pde_residual, partials
    else:
        return pde_residual


@jaxtyped(typechecker=typeguard.typechecked)
def navier_stokes_velocity_from_vorticity(
    w: Float[torch.Tensor, "xspace yspace"],
) -> Tuple[Float[torch.Tensor, "xspace yspace"], Float[torch.Tensor, "xspace yspace"]]:
    """
    Computes the velocity field from the vorticity field.
    :param w: Vorticity of shape (Nx, Ny)
    :return: Tuple of (vx, vy) velocity fields of shape (Nx, Ny)
    """

    what = torch.fft.fft2(w)
    nx, ny = w.shape[-2:]

    # Compute wave numbers
    kx = torch.tile(
        torch.fft.fftfreq(nx, device=what.device)[:, None] * nx * 2 * torch.pi, (1, ny)
    )
    ky = torch.tile(
        torch.fft.fftfreq(ny, device=what.device)[None, :] * ny * 2 * torch.pi, (nx, 1)
    )

    # Compute negative laplacian
    lap = kx**2 + ky**2
    lap[0, 0] = 1

    # Compute velocities
    vx = torch.fft.irfft2(what * 1j * ky / lap, what.shape)
    vy = torch.fft.irfft2(what * -1j * kx / lap, what.shape)

    return vx, vy


@jaxtyped(typechecker=typeguard.typechecked)
def navier_stokes_2d_residual_compute(
    solution_field: TypeBatchSolField2D,
    pde_params: TypeParam,
    spatial_grid: List[Union[TypeXGrid, TypeYGrid]],
    t: TypeTimeGrid,
    return_partials: bool = False,
) -> (
    Float[torch.Tensor, "batch time 1 xspace yspace"]
    | tuple[
        Float[torch.Tensor, "batch time 1 xspace yspace"],
        Float[torch.Tensor, "batch time 3 xspace yspace"],
    ]
):
    """
    Compute the PDE residual for 2D unforced Navier Stokes in vorticity form. Equation:
    dw/dt + (u * \\nabla w) - re * \\lap w = 0
    args:
        solution_field: solution_field of 2D Navier Stokes
        pde_params: Dictionary of parameters for 2D Navier Stokes. The only key should be "re" for the reynolds number.
        x,y : spatial grids
        t: temporal grid
        return_partials: Flag to return partial derivatives in 2D Navier Stokes equation
    Also see fluids.py::navier_stokes_residual
    """

    re = pde_params["re"]
    # remove batch dim
    x = spatial_grid[0][0]
    y = spatial_grid[1][0]
    t = t[0]
    # Channel last representation
    solution_field = torch.permute(solution_field, (0, 1, 3, 4, 2))

    residual_func = torch.vmap(
        navier_stokes_residual, in_dims=(0, None, None, None, 0, None)
    )

    if return_partials:
        residual, partials = residual_func(solution_field, t, x, y, re, return_partials)
    else:
        residual = residual_func(solution_field, t, x, y, re, return_partials)
    # B, T, X, Y, C -> B, T, C, X, Y
    residual = torch.permute(residual, (0, 1, 4, 2, 3))

    if return_partials:
        return residual, partials
    return residual


@jaxtyped(typechecker=typeguard.typechecked)
def turbulent_flow_2d_residual_compute(
    solution_field: TypeBatchSolField2D,
    pde_params: TypeParam,
    spatial_grid: List[Union[TypeXGrid, TypeYGrid]],
    t: TypeTimeGrid,
    return_partials: bool = False,
) -> (
    Float[torch.Tensor, "batch time 1 xspace yspace"]
    | tuple[
        Float[torch.Tensor, "batch time 1 xspace yspace"],
        Float[torch.Tensor, "batch time 3 xspace yspace"],
    ]
):
    """
    Computes residual for forced 2D TF. See fluids.py::turbulent_flow_residual.
    """
    nu = pde_params["nu"]
    # remove batch dim
    x = spatial_grid[0][0]
    y = spatial_grid[1][0]
    t = t[0]
    # Channel last representation
    solution_field = torch.permute(solution_field, (0, 1, 3, 4, 2))

    residual_func = torch.vmap(
        turbulent_flow_residual, in_dims=(0, None, None, None, 0, None)
    )

    if return_partials:
        residual, partials = residual_func(solution_field, t, x, y, nu, return_partials)
    else:
        residual = residual_func(solution_field, t, x, y, nu, return_partials)
    # B, T, X, Y, C -> B, T, C, X, Y
    residual = torch.permute(residual, (0, 1, 4, 2, 3))

    if return_partials:
        return residual, partials
    return residual


@jaxtyped(typechecker=typeguard.typechecked)
def kdv_1d_residual_compute(
    solution_field: TypePartials1D,
    pde_params: TypeParam,
    spatial_grid: List[TypeXGrid],
    t: TypeTimeGrid,
    return_partials: bool = False,
) -> Type1DKDVPartialsReturnType:
    """
    Compute the PDE residual for 1D Korteweg de Vries.
    du/dt + 6u * du/dx + delta**2 * d^3u/dx^3 = 0
    args:
        solution_field: solution_field of 1D KDV
        x : spatial grids
        t: temporal grid
        delta: 1d KDV parameter
        return_partials: Flag to return partial derivatives in 1D KDV
    """
    # Spatial grid is a tuple of tensors
    # Each tensor is of shape B x Nx
    x = spatial_grid[0]
    if len(t.shape) == 2:
        dt = t[0, 1] - t[0, 0]
    else:
        dt = t[1] - t[0]
    u = solution_field
    delta = pde_params["delta"]
    delta = delta.unsqueeze(-1)

    data_x, _, data_xx, data_t = partials_torch_1d_systems(u, x, dt)

    # We still need data_xxx
    x_axis = -1
    # Since each spatial is B x Nx, we need to grab a single x
    x = x[0]
    data_xxx = torch.gradient(data_xx, spacing=(x,), dim=x_axis)[0]

    residual = data_t + u * data_x + delta**2 * data_xxx
    if return_partials:
        u_partials = torch.cat([data_x, data_xx, data_xxx, data_t], dim=1)
        return residual, u_partials

    return residual


def _single_darcy_flow_residual(
    solution_field: Float[Tensor, "1 1 nx ny"],  # time, channel, nx, ny
    binary_coeffs: Float[Tensor, "1 nx ny"],
    max_val: float,
    min_val: float,
    dx: float,
    dy: float,
    return_partials: bool = False,
) -> (
    Float[Tensor, "1 1 nx ny"]
    | tuple[Float[Tensor, "1 1 nx ny"], Float[Tensor, "1 4 nx ny"]]
):
    # Note: As shorthand to fit the math, we use u = solution_field and a = coeffs
    u = solution_field
    # prune time dim
    u = torch.squeeze(u, dim=0)

    # Denormalize coeffs
    normalized_diff = max_val - min_val
    a = (binary_coeffs * normalized_diff) + min_val
    forcing_func = torch.ones_like(u)
    _, nx, ny = u.shape

    ux = torch.gradient(u, spacing=dx, dim=1)[0]
    uy = torch.gradient(u, spacing=dy, dim=2)[0]

    aux = a * ux
    auy = a * uy

    auxx = torch.gradient(aux, spacing=dx, dim=1)[0]
    auyy = torch.gradient(auy, spacing=dy, dim=2)[0]
    lhs = -(auxx + auyy)
    residual = lhs - forcing_func
    # Add back in time dim
    residual = torch.unsqueeze(residual, dim=0)

    if return_partials:
        uxx = torch.gradient(ux, spacing=dx, dim=1)[0]
        uyy = torch.gradient(uy, spacing=dy, dim=2)[0]
        # 1, nx, ny -> num_partials, nx, ny
        partials = torch.cat([ux, uy, uxx, uyy], dim=0)
        # Add time dim: 1, nx, ny -> 1, 1, nx, ny
        partials = torch.unsqueeze(partials, dim=0)
        return residual, partials
    return residual


@jaxtyped(typechecker=typeguard.typechecked)
def darcy_flow_2d_residual_compute(
    solution_field: Float[Tensor, "batch 1 1 nx ny"],
    pde_params: TypeParam,
    spatial_grid: List[Union[TypeXGrid, TypeYGrid]],
    t: TypeTimeGrid,
    return_partials: bool = False,
) -> (
    Float[Tensor, "batch 1 nx ny"]
    | tuple[Float[Tensor, "batch 1 1 nx ny"], Float[Tensor, "batch 1 4 nx ny"]]
):
    """
    Compute the 2D Darcy Flow residual. Darcy flow is time independent so t is
    not used. PDE:

    args:
        solution_field: Solution field of 2D darcy flow (batch, channel, nx, ny)
        x: spatial grids
        t: temporal grid (unused)
        params: coeff field [tensor same shape as solution_field]
        return_partials: bool to return partial derivatives
    """
    x, y = spatial_grid
    x = x[0]
    y = y[0]
    dx = (x[1] - x[0]).item()
    dy = (y[1] - y[0]).item()
    max_vals = pde_params["max_val"]
    min_vals = pde_params["min_val"]
    binary_coeffs = pde_params["coeff"]

    del t
    return torch.vmap(
        _single_darcy_flow_residual,
        in_dims=(
            0,
            0,
            0,
            0,
            None,
            None,
            None,
        ),
        out_dims=0,
    )(
        solution_field,
        binary_coeffs,
        max_vals,
        min_vals,
        dx,
        dy,
        return_partials,
    )
