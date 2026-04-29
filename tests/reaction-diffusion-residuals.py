import h5py
from pdeinvbench.data.utils import extract_params_from_path
from pdeinvbench.utils.types import PDE
from pdeinvbench.utils.test_utils import create_gif_and_save
from pdeinvbench.losses import pde_residuals
from os import makedirs, listdir
from os.path import join
import torch
import numpy as np
import matplotlib.pyplot as plt


"""Simple script that tests the residuals for the PDEs."""

img_dir = "./test-images"


def reaction_diffusion_2d():
    # 2D Reaction Diffusion

    # Plot
    f = "/data/nithinc/meta-pde/reaction-diffusion-2d/train"
    files = [f for f in listdir(f) if f.endswith(".h5") or f.endswith(".hdf5")]
    data_file = join(f, files[0])
    data = h5py.File(data_file, "r")
    x = torch.Tensor(data["0001"]["grid"]["x"][:]).unsqueeze(0)
    y = torch.Tensor(data["0001"]["grid"]["y"][:]).unsqueeze(0)
    t = torch.Tensor(data["0001"]["grid"]["t"][:]).unsqueeze(0)
    sol = torch.Tensor(data["0001"]["data"])
    sol = sol.unsqueeze(0)  # Batch dim
    sol = torch.permute(sol, (0, 1, 4, 2, 3))  # B x T x C x X x Y
    pde_params = extract_params_from_path(files[0], PDE.ReactionDiffusion2D)

    # Compute the residuals
    residual_func = pde_residuals.get_pde_residual_function(PDE.ReactionDiffusion2D)
    residual = residual_func(sol, pde_params, [x, y], t, return_partials=False)
    eqn1 = residual[:, :, 0]
    eqn1_scalar = torch.nn.functional.mse_loss(eqn1, torch.zeros_like(eqn1))
    eqn2 = residual[:, :, 1]
    eqn2_scalar = torch.nn.functional.mse_loss(eqn2, torch.zeros_like(eqn2))
    eqn1 = eqn1.squeeze()
    eqn2 = eqn2.squeeze()

    sol = sol.squeeze()  # Remove batch dim of sol
    # Visualize as GIFs
    create_gif_and_save(
        eqn1,
        join(img_dir, "2d-reaction-diffusion-eqn1.gif"),
        cmap="magma",
        title=f"2D Reaction Diffusion u Residual {eqn1_scalar}",
    )
    create_gif_and_save(
        eqn2,
        join(img_dir, "2d-reaction-diffusion-eqn2.gif"),
        cmap="magma",
        title=f"2D Reaction Diffusion v Residual {eqn2_scalar}",
    )
    create_gif_and_save(
        torch.abs(eqn1) > 1e-2,
        join(img_dir, "2d-reaction-diffusion-eqn1-1e-2.gif"),
        cmap="PuRd",
        title=f"2D Reaction Diffusion u Residual (Threshold 1e-2)",
    )
    create_gif_and_save(
        torch.abs(eqn2) > 1e-2,
        join(img_dir, "2d-reaction-diffusion-eqn2-1e-2.gif"),
        cmap="PuRd",
        title=f"2D Reaction Diffusion v Residual (Threshold 1e-2)",
    )
    create_gif_and_save(
        torch.abs(eqn1) > 1e-4,
        join(img_dir, "2d-reaction-diffusion-eqn1-1e-4.gif"),
        cmap="PuRd",
        title=f"2D Reaction Diffusion u Residual (Threshold 1e-4)",
    )
    create_gif_and_save(
        torch.abs(eqn2) > 1e-4,
        join(img_dir, "2d-reaction-diffusion-eqn2-1e-4.gif"),
        cmap="PuRd",
        title=f"2D Reaction Diffusion v Residual (Threshold 1e-4)",
    )


def main():
    ### Folder to plot images ###
    makedirs(img_dir, exist_ok=True)

    # Test residuals
    reaction_diffusion_2d()


if __name__ == "__main__":
    main()
