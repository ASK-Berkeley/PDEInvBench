import os
from functools import partial

import h5py as h
import numpy as np
import torch
from jaxtyping import Complex, Float
from torch import Tensor, vmap

"""
Methods to compute the pde residual of turbulent flow and navier stokes.
"""


def _maybe_unsqueeze_3d(
    u: Float[Tensor, "nt nx ny"] | Float[Tensor, "nt nx ny 1"],
) -> Float[Tensor, "nt nx ny 1"]:
    """
    Given a tensor, makes sure that the last dimension is 1 (channel dim).
    Helps to ensure number of channels consistency.
    NOTE: This should only be used within this file. Assumes that u is an unbatched fluid field.
    Also always assumes well-formed input.
    """
    return u if len(u.shape) == 4 and u.shape[-1] == 1 else torch.unsqueeze(u, dim=-1)


def _maybe_unsqueeze_2d(
    u: Float[Tensor, "nx ny"] | Float[Tensor, "nx ny 1"],
) -> Float[Tensor, "nx ny 1"]:
    """
    Same as 3d version but assumes a tensor of shape (nx, ny, 1)
    """
    return u if len(u.shape) == 3 and u.shape[-1] == 1 else torch.unsqueeze(u, dim=-1)


@partial(vmap, in_dims=(0, None, None, None))
def compute_stream_function(
    vorticity: Float[Tensor, "nx ny 1"],
    dx: float,
    dy: float,
    fourier: bool = False,
) -> Float[Tensor, "nx ny 1"] | Complex[Tensor, "nx ny 1"]:
    """
    Compute the stream function psi. If :arg fourier: returns the fft coefficients.
    Otherwise, returns the real components.
    :args:
        - vorticity: (nx, ny, 1) vorticity in real space
        - dx: float
        - dy: float
        - fourier: bool - whether to return fft coeffs or real space
    """
    w = vorticity
    device = w.device
    w = torch.squeeze(w)
    what = torch.fft.fft2(w)
    nx, ny = w.shape
    kx = torch.fft.fftfreq(nx, d=dx) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=dy) * 2 * torch.pi
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    kx, ky = kx.to(device), ky.to(device)
    wavenumbers_squared = kx**2 + ky**2
    # stream function = psi
    psi_hat = torch.zeros_like(
        what, device=what.device
    )  # NOTE: zeros_like implicit broadcasts to cfloat (this might change in the future)
    psi_hat[wavenumbers_squared > 0] = what[wavenumbers_squared > 0] / (
        -wavenumbers_squared[wavenumbers_squared > 0]
    )

    if fourier:
        return _maybe_unsqueeze_2d(psi_hat)
    else:
        return _maybe_unsqueeze_2d(torch.fft.ifft2(psi_hat).real)


def compute_first_order_gradient(
    u: Float[Tensor, "nt nx ny 1"], spacing: float, dim: int, fourier: bool = False
):
    """
    Returns the first derivative with respect to :arg dim:. Spacing must be provided (dx).
    """

    if fourier:
        kx = torch.fft.fftfreq(u.shape[1], d=spacing) * 2 * torch.pi
        ky = torch.fft.fftfreq(u.shape[2], d=spacing) * 2 * torch.pi
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        kx, ky = kx.to(u.device), ky.to(u.device)
        kx = kx.unsqueeze(0)
        ky = ky.unsqueeze(0)

        # print("kx.shape, ky.shape, u.shape", kx.shape, ky.shape, u.shape)
        if dim == 1:
            return torch.fft.ifft2(1j * kx * torch.fft.fft2(u)).real
        elif dim == 2:
            return torch.fft.ifft2(1j * ky * torch.fft.fft2(u)).real
        else:
            raise ValueError(f"Invalid dimension: {dim}")
    else:
        return torch.gradient(u, spacing=spacing, dim=dim)[0]


def vorticity_to_velocity(
    vorticity: Float[Tensor, "nt nx ny 1"], dx: float, dy: float
) -> tuple[Float[Tensor, "nt nx ny 1"], Float[Tensor, "nt nx ny 1"]]:
    """
    Given vorticity, dx, dy, returns a pair (vx, vy) corresponding to velocity
    in the x and y directions.
    """
    w = vorticity
    device = w.device
    w = torch.squeeze(w)
    nt, nx, ny = w.shape
    psi_hat = compute_stream_function(vorticity, dx, dy, True)
    psi_hat = torch.squeeze(psi_hat)

    # Compute velcities using psi
    kx = torch.fft.fftfreq(nx, d=dx) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=dy) * 2 * torch.pi
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    kx, ky = kx.to(device), ky.to(device)
    vx_hat = 1j * ky * psi_hat
    vy_hat = -1j * kx * psi_hat
    vx = torch.fft.ifft2(vx_hat).real
    vy = torch.fft.ifft2(vy_hat).real
    return _maybe_unsqueeze_3d(vx), _maybe_unsqueeze_3d(vy)


def compute_advection(
    vorticity: Float[Tensor, "nt nx ny 1"],
    dx: float,
    dy: float,
    return_velocity: bool = False,
) -> (
    Float[Tensor, "nt nx ny 1"]
    | tuple[
        Float[Tensor, "nt nx ny 1"],
        Float[Tensor, "nt nx ny 1"],
        Float[Tensor, "nt nx ny 1"],
    ]
):
    """
    Computes the advection term of Navier stokes: v * nabla w
    :args
        - vorticity: (nt, nx, ny, 1)
        - dx: float
        - dy: float
        - return_velocity (bool): Whether to return adv, vx, vy
    """
    w = vorticity
    w = torch.squeeze(w)
    vx, vy = vorticity_to_velocity(vorticity, dx, dy)
    w_dx = compute_first_order_gradient(w, dx, 1, fourier=True)
    w_dx = _maybe_unsqueeze_3d(w_dx)
    w_dy = compute_first_order_gradient(w, dy, 2, fourier=True)
    w_dy = _maybe_unsqueeze_3d(w_dy)
    adv = vx * w_dx + vy * w_dy
    assert len(adv.shape) == 4, "Incorrect advection shape"
    if return_velocity:
        return _maybe_unsqueeze_3d(adv), vx, vy
    return _maybe_unsqueeze_3d(adv)


def second_order_gradient(
    u: Float[Tensor, "..."], spacing: float, dim: int
) -> Float[Tensor, "..."]:
    """
    Returns the second derivative with respect to :arg dim:. Spacing must be provided (dx).
    """
    return torch.gradient(
        torch.gradient(u, spacing=spacing, dim=dim)[0], spacing=spacing, dim=dim
    )[0]


def laplacian(
    vorticity: Float[Tensor, "nt nx ny 1"], dx: float, dy: float
) -> Float[Tensor, "nt nx ny 1"]:
    """
    Computes the laplacian of vorticity assuming some dx and dy.
    args:
        - vorticity (nt, nx, ny, 1)
        - dx: float
        - dy: float
    """
    w = vorticity
    w = np.squeeze(w)
    nt, nx, ny = w.shape
    w_dxx = second_order_gradient(w, spacing=dx, dim=1)
    w_dyy = second_order_gradient(w, spacing=dy, dim=2)
    return _maybe_unsqueeze_3d(w_dxx + w_dyy)


def wrapper(func):
    def _wrapper(*args):
        new_args = []
        for a in args:
            if isinstance(a, np.ndarray):
                new_args.append(torch.from_numpy(a))
            else:
                new_args.append(a)
        out = func(*new_args)
        return out.cpu().numpy()

    return _wrapper


def _maybe_unsqueeze_np(u):
    return u if u.shape[-1] == 1 else np.expand_dims(u, axis=-1)


def compare_funcs(f1, f2):
    def compare(f1args, f2args):
        reference = _maybe_unsqueeze_np(f1(*f1args))
        newout = wrapper(f2)(*f2args)
        logger.info(
            f"Diff between {f1.__name__} and {f2.__name__} was {np.linalg.norm(reference - newout)}"
        )

    return compare


def turbulent_flow_residual(
    vorticity: Float[Tensor, "nt nx ny 1"],
    t: Float[Tensor, "nt"],
    x: Float[Tensor, "nx"],
    y: Float[Tensor, "ny"],
    nu: float,
    return_partials: bool = False,
):
    """
    Computes the NS equation assuming a damping coeff of 0.1 and Kolmogorov forcing func.
    NOTE: This function is unbatched.
    Eqn:
        -dwdt -v * nable w + nu * lap(w) - alpha*w + f
    """
    dt: float = (t[1] - t[0]).item()
    dx: float = (x[1] - x[0]).item()
    dy: float = (y[1] - y[0]).item()
    nt, nx, ny, _ = vorticity.shape
    dwdt = torch.gradient(vorticity, spacing=dt, dim=0)[0]
    if return_partials:
        advection, vx, vy = compute_advection(
            vorticity, dx=dx, dy=dy, return_velocity=True
        )
    else:
        advection = compute_advection(vorticity, dx=dx, dy=dy)

    # desired equation: dwdt = -v * \nabla w + v \nabla^2 w - alpha*w + f
    alpha = 0.1
    damping_term = alpha * vorticity
    forced_mode = 2
    forcing_func = forced_mode * torch.cos(forced_mode * y)
    forcing_func = torch.reshape(forcing_func, (1, 1, ny, 1))
    forcing_func = torch.broadcast_to(forcing_func, vorticity.shape)
    diffusion_term = nu * laplacian(vorticity, dx=dx, dy=dy)

    # residual = -dwdt + (advection) + diffusion_term - damping_term + forcing_func
    residual = -dwdt + (advection) + diffusion_term - damping_term + forcing_func

    if return_partials:
        dwdx = torch.gradient(vorticity, spacing=dx, dim=1)[0]
        dwdy = torch.gradient(vorticity, spacing=dy, dim=2)[0]
        partials = torch.stack([dwdt, dwdx, dwdy], dim=0)
        # There's a trailing channel dim that we replace
        partials = torch.squeeze(partials, dim=-1)
        return residual, partials
    return residual


def navier_stokes_residual(
    vorticity: Float[Tensor, "nt nx ny 1"],
    t: Float[Tensor, "nt"],
    x: Float[Tensor, "nx"],
    y: Float[Tensor, "ny"],
    re: float,
    return_partials: bool = False,
):
    """
    Computes the unforced NS equation with no damping.
    NOTE: This function is unbatched.
    Eqn:
        -dwdt -v * nable w + nu * lap(w)
    """
    # Compute the viscosity from Re number
    # Map backward: l (characteristic / length scale)
    l = 0.8
    ic_scaling = 1
    nu: float = l * ic_scaling / re
    dt: float = (t[1] - t[0]).item()
    dx: float = (x[1] - x[0]).item()
    dy: float = (y[1] - y[0]).item()
    nt, nx, ny, _ = vorticity.shape
    dwdt = torch.gradient(vorticity, spacing=dt, dim=0)[0]
    if return_partials:
        advection, vx, vy = compute_advection(
            vorticity, dx=dx, dy=dy, return_velocity=True
        )
    else:
        advection = compute_advection(vorticity, dx=dx, dy=dy)

    # desired equation: dwdt = -v * \nabla w + v \nabla^2 w - alpha*w + f
    diffusion_term = nu * laplacian(vorticity, dx=dx, dy=dy)

    residual = -dwdt + (advection) + diffusion_term

    if return_partials:
        dwdx = torch.gradient(vorticity, spacing=dx, dim=1)[0]
        dwdy = torch.gradient(vorticity, spacing=dy, dim=2)[0]
        partials = torch.stack([dwdt, dwdx, dwdy], dim=0)
        partials = torch.squeeze(partials, dim=-1)
        return residual, partials
    return residual


if __name__ == "__main__":
    import argparse

    from fluids_numpy import advection as advection_np_base
    from fluids_numpy import compute_stream_function as compute_stream_function_np
    from fluids_numpy import laplacian as laplacian_np
    from fluids_numpy import tf_residual_numpy
    from loguru import logger

    logger.warning(
        "You should only be running this script as a main function if you are testing the fluids residual computation"
    )

    parser = argparse.ArgumentParser(
        description="Runs a series of tests to check a numpy implementation with the current torch NS+TF residual computations"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="../../../data/2D_turbulent-flow_nu=0.006153085601625313.h5",
    )
    args = parser.parse_args()
    filename = args.filename
    dataset = h.File(filename)
    traj_idx = "0000"
    data = dataset[traj_idx]["data"][:]
    logger.info(f"Data shape {data.shape}")
    t = dataset[traj_idx]["grid/t"][:]
    x = dataset[traj_idx]["grid/x"][:]
    y = dataset[traj_idx]["grid/y"][:]
    nu = os.path.basename(filename).split("=")[-1][:-3]
    nu: float = float(nu)
    dataset.close()

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    logger.info(
        f"Computed residual norm {np.linalg.norm(wrapper(turbulent_flow_residual)(data, t, x, y, nu))}"
    )

    compute_stream_function_np.__name__ = "compute_stream_function_np, fourier=False"
    compare_funcs(compute_stream_function_np, compute_stream_function)(
        (data, x, y, False), (data, dx, dy, False)
    )
    compute_stream_function_np.__name__ = "compute_stream_function_np, fourier=True"
    compare_funcs(compute_stream_function_np, compute_stream_function)(
        (data, x, y, True), (data, dx, dy, True)
    )

    def advection_np(u, x, y):
        return advection_np_base(u, x, y, stream_func=compute_stream_function_np)[0]

    advection_np.__name__ = "advenction_np"
    compare_funcs(advection_np, compute_advection)((data, x, y), (data, dx, dy))

    laplacian_np.__name__ = "laplacian_np"
    compare_funcs(laplacian_np, laplacian)((data, x, y), (data, dx, dy))

    # Compare the velocity conversions
    def advection_np(u, x, y):
        return advection_np_base(u, x, y, stream_func=compute_stream_function_np)[1]

    advection_np.__name__ = "advenction_np for vx"
    compare_funcs(
        advection_np, lambda *args: compute_advection(*args, return_velocity=True)[1]
    )((data, x, y), (data, dx, dy))

    def advection_np(u, x, y):
        return advection_np_base(u, x, y, stream_func=compute_stream_function_np)[2]

    advection_np.__name__ = "advenction_np for vy"
    compare_funcs(
        advection_np, lambda *args: compute_advection(*args, return_velocity=True)[2]
    )((data, x, y), (data, dx, dy))

    compare_funcs(tf_residual_numpy, turbulent_flow_residual)(
        (data, t, x, y, nu), (data, t, x, y, nu)
    )
