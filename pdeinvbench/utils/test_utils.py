import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from functorch.dim import tree_flatten, tree_map
import logging

"""
Helpers for various PyTests.
"""


def prune_boundary(array, dim):
    """
    Prune the boundary of an array.
    """
    if dim == 0:
        return array[1:-2]
    elif dim == 1:
        return array[:, 1:-2]
    elif dim == 2:
        return array[:, :, 1:-2]
    elif dim == 3:
        return array[:, :, :, 1:-2]
    else:
        raise ValueError("Invalid dimension.")


def array_difference_less_than(a, b, val):
    """
    Check if all elements in A - B are less than val.
    """
    return torch.all((a - b) < val)


def generate_synthetic_data_1d(batch_size=4, Nx=100, Nt=1024):
    """
    Generate synthetic data for 1D reaction diffusion.
    """
    x = torch.linspace(0, 1, Nx)
    t = torch.linspace(0, 1, Nt)
    tt, xx = torch.meshgrid(t, x)
    u = torch.sin(xx) * torch.cos(tt)
    du_dx = torch.cos(tt) * torch.cos(xx)
    du_dt = -torch.sin(tt) * torch.sin(xx)
    ddu_dxx = -torch.cos(tt) * torch.sin(xx)
    du_sqr_dx = 2 * (torch.cos(tt) ** 2) * torch.sin(xx) * torch.cos(xx)

    ## Account for batch sizes
    u = u.repeat(batch_size, 1, 1)
    du_dx = du_dx.repeat(batch_size, 1, 1)
    du_dt = du_dt.repeat(batch_size, 1, 1)
    ddu_dxx = ddu_dxx.repeat(batch_size, 1, 1)
    du_sqr_dx = du_sqr_dx.repeat(batch_size, 1, 1)
    return x, t, u, du_dx, du_dt, ddu_dxx, du_sqr_dx


def generate_synthetic_data_2d(batch_size=4, Nx=100, Ny=100, Nt=1024):
    """
    Generate synthetic data to test 2D finite differences. (3D including time).
    """
    x = torch.linspace(0, 1, Nx)
    y = torch.linspace(0, 1, Ny)
    t = torch.linspace(0, 1, Nt)
    tt, xx, yy = torch.meshgrid(t, x, y)
    u = torch.cos(tt) * torch.sin(xx) * y * y
    du_dx = y * y * torch.cos(tt) * torch.cos(xx)
    du_dy = 2 * y * torch.cos(tt) * torch.sin(xx)
    ddu_dxx = -(y * y) * torch.cos(tt) * torch.sin(xx)
    ddu_dyy = 2 * torch.cos(tt) * torch.sin(xx)
    du_dt = -y * y * torch.sin(tt) * torch.sin(xx)

    # Account for batch sizes
    u = u.repeat(batch_size, 1, 1, 1)
    du_dx = du_dx.repeat(batch_size, 1, 1, 1)
    du_dy = du_dy.repeat(batch_size, 1, 1, 1)
    ddu_dxx = ddu_dxx.repeat(batch_size, 1, 1, 1)
    ddu_dyy = ddu_dyy.repeat(batch_size, 1, 1, 1)
    du_dt = du_dt.repeat(batch_size, 1, 1, 1)
    return x, y, t, u, du_dx, du_dy, ddu_dxx, ddu_dyy, du_dt


def create_gif_and_save(data, filename, title, cmap="magma", interval=50):
    """
    Create a gif from a list of images and save it.
    :param data: list of frames
    :param filename: location to save gif
    :param title: title of the gif
    :param cmap: colormap
    :param interval: interval between frames
    """
    vmin = data.min()
    vmax = data.max()
    fig, ax = plt.subplots()
    im = ax.imshow(data[0], animated=True, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im)

    def _update(i):
        im.set_array(data[i])
        return (im,)

    animation_fig = animation.FuncAnimation(
        fig,
        _update,
        frames=len(data),
        interval=interval,
        blit=True,
        repeat_delay=10,
    )
    # Specify writer for GIF - requires pillow: pip install pillow
    try:
        animation_fig.save(filename, writer="pillow")
    except Exception as e:
        # Fallback to imagemagick if pillow fails
        print(f"Pillow writer failed, trying imagemagick: {e}")
        animation_fig.save(filename, writer="imagemagick")
    finally:
        plt.close(fig)  # Clean up to avoid memory leaks
