from warnings import warn

import numpy as np

warn(
    "DEPRECATED: The file pdeinvbench/losses/fluids_numpy.py is being imported. This is most certainly a bug!"
)


def _maybe_unsqueeze_3d(
    u,
):
    """
    Given a tensor, makes sure that the last dimension is 1 (channel dim).
    Helps to ensure number of channels consistency.
    NOTE: This should only be used within this file. Assumes that u is an unbatched fluid field.
    Also always assumes well-formed input.
    """
    return u


def _maybe_unsqueeze_2d(u):
    """
    Same as 3d version but assumes a tensor of shape (nx, ny, 1)
    """
    return u


def _compute_stream_function(vorticity, x, y, fourier=False):
    """
    from the vorticity, compute the stream function and returns in physical space
    vorticity shape: (nt, nx, ny, 1) Channels is squeezed off
    """
    w = vorticity
    w = np.squeeze(w)
    what = np.fft.fft2(w)
    nx, ny = w.shape[-2], w.shape[-1]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    wavenumbers_squared = kx**2 + ky**2
    # stream function = psi
    psi_hat = np.zeros_like(what, dtype=np.complex128)
    psi_hat[wavenumbers_squared > 0] = what[wavenumbers_squared > 0] / (
        -wavenumbers_squared[wavenumbers_squared > 0]
    )

    if fourier:
        return _maybe_unsqueeze_2d(psi_hat)
    else:
        return np.fft.ifft2(psi_hat).real


def compute_stream_function(vorticity, x, y, fourier=False):
    nt = vorticity.shape[0]
    out = []
    for i in range(nt):
        out.append(_compute_stream_function(vorticity[i], x, y, fourier=fourier))
    return np.asarray(out)


def _compute_stream_function_jaxcfd(vorticity, x, y, fourier=False):
    w = vorticity
    w = np.squeeze(w)
    what = np.fft.fft2(w)
    nx, ny = w.shape[-2], w.shape[-1]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    wavenumbers_squared = kx**2 + ky**2

    # Compute psi
    lap = wavenumbers_squared
    lap[0, 0] = 1
    psi_hat = -1 / lap * what
    if fourier:
        return psi_hat
    return np.fft.ifft2(psi_hat).real


def compute_stream_function_jaxcfd(vorticity, x, y, fourier=False):
    nt = vorticity.shape[0]
    out = []
    for i in range(nt):
        out.append(_compute_stream_function_jaxcfd(vorticity[i], x, y, fourier=fourier))
    return np.asarray(out)


def advection(vorticity, x, y, stream_func):
    psi_hat = stream_func(vorticity, x, y, fourier=True)
    w = vorticity
    w = np.squeeze(w)
    # Compute u, v
    nx, ny = w.shape[-2], w.shape[-1]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    vx_hat = 1j * ky * psi_hat
    vy_hat = -1j * kx * psi_hat
    vx = np.fft.ifft2(vx_hat).real
    vy = np.fft.ifft2(vy_hat).real
    w_dx = np.gradient(w, dx, axis=1)
    w_dy = np.gradient(w, dy, axis=2)

    adv = vx * w_dx + vy * w_dy
    return _maybe_unsqueeze_3d(adv), _maybe_unsqueeze_3d(vx), _maybe_unsqueeze_3d(vy)


def second_order_gradient(field, d_coord, axis):
    return np.gradient(np.gradient(field, d_coord, axis=axis), d_coord, axis=axis)


def laplacian(vorticity, x, y):
    w = vorticity
    w = np.squeeze(w)
    nx, ny = w.shape[-2], w.shape[-1]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    w_dxx = second_order_gradient(w, dx, axis=1)
    w_dyy = second_order_gradient(w, dy, axis=2)
    return w_dxx + w_dyy


def tf_residual_numpy(w, t, x, y, nu):
    alpha = 0.1
    forced_mode = 2
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    f = forced_mode * np.cos(forced_mode * y)  # Forcing function
    # Broadcast to solution field: nt, nx, ny, 1
    f = np.reshape(f, (1, 1, -1, 1))
    f = np.broadcast_to(f, w.shape)
    # Damping term: alpha * w
    damping = alpha * w
    # Compute dwdt
    dt: float = t[1] - t[0]
    dwdt = np.gradient(w, dt, axis=0)

    #### Stream function distraction ####
    # In order to compute v, we need to compute the stream function psi
    # This is necessary to compute the advection term
    psi = compute_stream_function(w, x, y)
    psi = np.expand_dims(psi, axis=-1)

    #### Final stream function = psi ####
    # compute advection
    adv, vx, vy = advection(w, x, y, compute_stream_function)
    prep_plot = lambda a: np.expand_dims(a, axis=-1)
    advection_terms = {"adv": adv, "vx": vx, "vy": vy}
    #### Continue with adv = v * \nabla w
    lap = laplacian(w, x, y)

    ## Diffusion term
    diffusion = nu * lap

    ### Now compute the new residual
    # desired equation: dwdt = -v * \nabla w + v \nabla^2 w - alpha*w + f
    adv = np.expand_dims(adv, axis=-1)
    diffusion = np.expand_dims(diffusion, axis=-1)

    residual = -1 * dwdt + -1 * adv + diffusion - damping + f
    return residual
