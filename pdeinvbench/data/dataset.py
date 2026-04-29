import glob
import logging
import math

import h5py
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset

from pdeinvbench.data.transforms import collapse_time_and_channels_torch_transform
from pdeinvbench.data.utils import extract_params_from_path
from pdeinvbench.utils.types import PDE, PDE_NUM_SPATIAL, PDE_TRAJ_LEN
from typing import Callable


class PDE_MultiParam(Dataset):
    """Data Loader that loads the multiple parameter version of PDE Datasets."""

    def __init__(
        self,
        data_root: str,
        pde: PDE,
        n_past: int,
        dilation: int,
        cutoff_first_n_frames: int,
        train: bool,
        frac_param_combinations: float = 1,
        frac_ics_per_param: float = 1,
        random_sample_param: bool = True,
        downsample_factor: int = 0,
        every_nth_window: int = 1,
        window_start_percent: float = 0.0,
        window_end_percent: float = 1.0,
        degradation_operator: Callable = None,
        grid_transform: Callable = None,
    ):
        """
        Args:
            data_root: path containing the h5 files for the current data split
            pde: name of the PDE system - one of the enum values.
            n_past: number of conditioning frames
            dilation: frequency at which to subsample the ground truth trajectories in the time dimension
            cutoff_first_n_frames: number of initial frames to cutoff in each trajectory (may want to do this e.g. if initial PDE residuals are very high)
            train: if training dataloader, windows are randomly sampled from each trajecory, if non-training dataloader we loop through all non-overlapping windows
            frac_param_combinations: fraction of parameter combinations to use. 1 takes all parameters. "0.x" takes x percent of total parameters
            frac_ics_per_param: fraction of initial conditions per parameter combination to keep.
            random_sample_param: (bool) If frac_param_combinations < 1, true means we randomly sample params and false means we grab the first n_frac params. Defaults to true.
            downsample_factor: downsample a solution field spatially by the 'downsample_factor'. eg if downsample_factor=4, sol field spatial size=[128,128] --downsample--> final spatial size = [32,32]
            every_nth_window: take every nth window from the list of non-over-lapping windows
            window_start_percent: percent of the way through the trajectory to start the window after cutoff_first_n_frames
            window_end_percent: percent of the way through the trajectory to end the window
            degradation_operator: degradation operator to apply to the data
            grid_transform: grid transform to apply to the data
        """

        self.data_root = data_root
        self.pde = pde
        self.n_past = n_past
        self.dilation = dilation
        self.cutoff_first_n_frames = cutoff_first_n_frames
        self.frac_param_combinations = frac_param_combinations
        self.frac_ics_per_param = frac_ics_per_param
        self.random_sample_param = random_sample_param
        self.train = train
        self.every_nth_window = every_nth_window
        self.degradation_operator = degradation_operator
        self.grid_transform = grid_transform
        assert (
            window_start_percent < window_end_percent
        ), "window_start_percent must be less than window_end_percent"
        self.window_start_index = int(
            (PDE_TRAJ_LEN[self.pde] - self.cutoff_first_n_frames) * window_start_percent
            + self.cutoff_first_n_frames
        )
        self.window_end_index = int(
            (PDE_TRAJ_LEN[self.pde] - self.cutoff_first_n_frames) * window_end_percent
            + self.cutoff_first_n_frames
        )
        self.total_trajectory_length = self.window_end_index - self.window_start_index

        if self.train:
            self.num_windows = self.total_trajectory_length - self.n_past - 1
        else:
            self.num_windows = (self.total_trajectory_length) // (
                (self.n_past) * self.every_nth_window
            )

            if self.num_windows == 0 and self.every_nth_window > 1:
                self.every_nth_window = 1
                self.num_windows = (self.total_trajectory_length) // ((self.n_past))

        # Quick check basically force a non-AR dataloader for darcy flow
        if self.pde == PDE.DarcyFlow2D:
            self.num_windows = 1

        self.downsample_factor = downsample_factor

        if PDE_NUM_SPATIAL[pde] == 2:
            self.transforms = [collapse_time_and_channels_torch_transform]
        else:
            self.transforms = None

        # get all h5 paths in the root folder and read them
        # each h5 path represents a set of trajectories with a different PDE parameter
        self.h5_paths = glob.glob(f"{self.data_root}/*.h5")
        if len(self.h5_paths) == 0:
            self.h5_paths = glob.glob(f"{self.data_root}/*.hdf5")
        if self.pde == PDE.DarcyFlow2D:
            self.h5_files = [file for file in self.h5_paths]
        else:
            self.h5_files = [h5py.File(file, "r") for file in self.h5_paths]

        # extract the individual trajectories from each h5 file
        if self.pde == PDE.ReactionDiffusion2D or self.pde == PDE.TurbulentFlow2D:
            self.seqs = [list(h5_file.keys()) for h5_file in self.h5_files]
        elif self.pde == PDE.NavierStokes2D:
            # The individual trajectories are stored in key: 'solutions'
            self.seqs = [h5_file["solutions"] for h5_file in self.h5_files]
        elif self.pde == PDE.KortewegDeVries1D:
            self.seqs = [h5_file["tensor"] for h5_file in self.h5_files]
        elif self.pde == PDE.DarcyFlow2D:
            # There is an issue where too many files are open, os throws errno 24
            self.seqs = [file for file in self.h5_paths]
        else:
            self.seqs = [h5py.File(file, "r") for file in self.h5_paths]
        if self.frac_param_combinations < 1:
            total_params = math.ceil(len(self.seqs) * self.frac_ics_per_param)

            logging.info(
                f"trimming dataset from length {len(self.seqs)} to {total_params}"
            )
            if self.random_sample_param:
                # Just a quick sanity check to ensure that all of the variables are the same length
                # If this fails, something has gone VERY wrong
                assert len(self.seqs) == len(self.h5_paths) and len(
                    self.h5_paths
                ) == len(
                    self.h5_files
                ), f"The dataloader variables are mismatched. seqs = {len(self.seqs)}, h5_paths = {len(self.h5_paths)}, h5_files = {len(self.h5_files)}"

                # We've had issues in the past with reproducibility so this forces a seed
                # Also will keep the datasets the same regardless of the training and weight init seeds
                num_sequences: int = len(self.seqs)
                requested_dataset_size: int = int(
                    num_sequences * self.frac_param_combinations
                )
                indices = np.arange(num_sequences)
                sample_seed: int = 42
                rng_generator = np.random.default_rng(seed=sample_seed)
                sampled_indices = rng_generator.choice(
                    indices, size=requested_dataset_size, replace=False
                )
                logging.info(
                    f"Using random sampling to trim the dataset down from length {len(self.seqs)} to {requested_dataset_size}"
                )
                assert (
                    len(set(sampled_indices.tolist())) == sampled_indices.shape[0]
                ), f"Duplicate items in random sampling of PDE parameters!"
                assert (
                    sampled_indices.shape[0] == requested_dataset_size
                ), f"Mismatch between the requested dataset sample size and the new sampled dataset. frac requested = {self.frac_param_combinations}, requested size = {requested_dataset_size}, new size = {sampled_indices.shape[0]}"
                self.seqs = [self.seqs[i] for i in sampled_indices]
                self.h5_paths = [self.h5_paths[i] for i in sampled_indices]
                self.h5_files = [self.h5_files[i] for i in sampled_indices]
            else:
                self.seqs = self.seqs[:total_params]
                self.h5_paths = self.h5_paths[:total_params]
                self.h5_files = self.h5_files[:total_params]

        self.num_params = len(self.seqs)
        if self.pde == PDE.KortewegDeVries1D:
            # Since it follows the same format at 1D reaction diffusion
            self.num_ics_per_param = self.seqs[0].shape[0]
        elif self.pde == PDE.DarcyFlow2D:
            self.num_ics_per_param = 1  # Each param only has one IC
        elif self.pde != PDE.NavierStokes2D:
            self.num_ics_per_param = len(
                min([self.seqs[i] for i in range(len(self.seqs))])
            )  # to manage un-even number of ICs per param
        else:
            self.num_ics_per_param = min(
                [self.seqs[i].shape[0] for i in range(len(self.seqs))]
            )

        # Trim nmber of ICs per parameter

        self.num_ics_per_param = math.ceil(
            self.num_ics_per_param * self.frac_ics_per_param
        )
        # We also need to save the dx, dt, dy information in order to compute the PDE residual
        if pde == PDE.ReactionDiffusion2D or pde == PDE.TurbulentFlow2D:
            self.x = self.h5_files[0]["0001"]["grid"]["x"][:]
            self.y = self.h5_files[0]["0001"]["grid"]["y"][:]
            self.t = torch.Tensor(self.h5_files[0]["0001"]["grid"]["t"][:])
        elif pde == PDE.NavierStokes2D:
            self.x = self.h5_files[0]["x-coordinate"][:]
            self.y = self.h5_files[0]["y-coordinate"][:]
            self.t = torch.Tensor(self.h5_files[0]["t-coordinate"][:])
        elif pde == PDE.DarcyFlow2D:
            # Not ideal but it's fine to just hard code the current coordinates darcy flow
            domain_len = 1  # Uniform grid with 1 - same regardless of resolution
            d = h5py.File(self.seqs[0], "r")
            size, _, _ = d["sol"].shape
            d.close()
            x = np.linspace(0, domain_len, size, endpoint=False)
            self.x = torch.Tensor(x)
            self.y = torch.Tensor(x)
            self.t = (
                torch.ones(10, dtype=float) * -1
            )  # Darcy flow is non time dependent so we use -1
        else:
            # All of the 1D systems
            self.y = None  # There is no y component
            self.x = self.h5_files[0]["x-coordinate"][:]
            self.t = torch.Tensor(self.h5_files[0]["t-coordinate"][:])

        if self.downsample_factor != 0:
            self.y = (
                None
                if self.y is None
                else signal.decimate(self.y, q=self.downsample_factor, axis=0).copy()
            )
            self.x = signal.decimate(self.x, q=self.downsample_factor, axis=0).copy()
        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y) if self.y is not None else None

        logging.info(
            f"Initialized dataset with {self.num_params} parameter combinations"
        )

    def __len__(self):
        """
        Number of parameters * number of ICs = number of full trajectories.
        """
        if self.train:
            return self.num_params * self.num_ics_per_param
        else:
            return self.num_params * self.num_ics_per_param * self.num_windows

    def __getitem__(self, index: int):
        """
        Loops over all parameters and ICs, and randomly samples time windows.
        Returns:
            x: conditioning frames, shape of [n_past, spatial/channel dims]
            y: target frame(s), shape of [n_future, spatial/channel dims]
            param_dict: dictionary containing the true PDE parameter for the trajectory.
        """
        # Compute the parameter and ic index for train loader
        if self.train:
            param_index = index // self.num_ics_per_param
            ic_index = index % self.num_ics_per_param
        else:
            # Compute the parameter, ic index, and window index for validation/test loaders
            # index is assumed to be in row major format of [num_params, num_ics_per_param, num_windows] dataset matrix organization
            param_index = index // (self.num_ics_per_param * self.num_windows)
            ic_index = (index // self.num_windows) % self.num_ics_per_param
            window_index = index % self.num_windows
        # get the corresponding trajectory and parameters
        h5_file = self.h5_files[param_index]
        h5_path = self.h5_paths[param_index]
        param_dict = extract_params_from_path(h5_path, self.pde)

        if self.pde == PDE.ReactionDiffusion2D or self.pde == PDE.TurbulentFlow2D:
            # get data
            seq = self.seqs[param_index][ic_index]
            traj = torch.Tensor(
                np.array(h5_file[f"{seq}/data"], dtype="f")
            )  # dim = [seq_len, spatial_dim_1, spatial_dim_2, channels]
        elif self.pde == PDE.NavierStokes2D:
            seq = self.seqs[param_index]
            traj = torch.Tensor(seq[ic_index])
            # dim = [seq_len (t), spatial_dim_1, spatial_dim_2, channels]

        elif self.pde == PDE.DarcyFlow2D:
            # Unique since there is no time dim
            # There is also only one ic per param
            seq = h5py.File(self.seqs[param_index], "r")

            coeff = torch.from_numpy(np.asarray(seq["coeff"]))
            coeff = torch.squeeze(coeff)
            coeff = torch.unsqueeze(coeff, dim=0)  # Channel first repr
            # We treat the coeff as a binary mask
            min_val = coeff.min()
            max_val = coeff.max()
            # generate the binary mask
            coeff = coeff - min_val
            binary_mask = coeff > 0

            def wrap_scalar(x):
                return torch.Tensor([x.item()])

            param_dict["coeff"] = binary_mask.float()
            param_dict["max_val"] = wrap_scalar(max_val)
            param_dict["min_val"] = wrap_scalar(min_val)
            traj = torch.from_numpy(np.asarray(seq["sol"]))
            seq.close()
        else:
            seq = self.seqs[param_index]
            traj = torch.Tensor(np.array(h5_file["tensor"][ic_index]))
        traj = traj[:: self.dilation]  # subsample based on dilation

        # sample a random window of length [n_past] from this trajectory
        if traj.shape[0] - self.n_past == 0:
            start = 0
            # if n_past > 1, problem is well posed
            if self.n_past == 1:
                raise ValueError("Problem is ill-posed when n_past == 1. ")
        else:
            if self.train:
                start = np.random.randint(
                    self.window_start_index,
                    self.window_end_index - self.n_past,
                )
            else:
                # multiply with self.n_past to avoid overlapping in validation/test samples
                start = self.window_start_index + (
                    window_index * (self.n_past) * self.every_nth_window
                )

        if self.pde != PDE.DarcyFlow2D:
            traj = traj[start : start + self.n_past]
            time_frames = self.t[start : start + self.n_past]
        else:
            time_frames = -1 * torch.ones(self.n_past, dtype=float)
        # 2D systems
        if len(traj.shape) == 4:
            # [T, Channels, Spatial, Spatial]
            traj = traj.permute((0, 3, 1, 2))

        if self.downsample_factor != 0:
            traj = signal.decimate(traj, q=self.downsample_factor, axis=-1)
            traj = (
                torch.Tensor(
                    signal.decimate(traj, q=self.downsample_factor, axis=-2).copy()
                )
                if len(traj.shape) == 4
                else torch.Tensor(traj.copy())
            )

        # split into conditioning and target frames
        if self.pde == PDE.DarcyFlow2D:
            # Transforms to reshape the traj to the expected shape
            # nx x ny x 1 -> T, C, X, Y
            # T == C == 1
            traj = torch.squeeze(traj)
            traj = torch.unsqueeze(traj, dim=0)
            traj = torch.unsqueeze(traj, dim=0)
            x = traj
            x = x.float()
        else:
            x, _ = torch.split(traj, [self.n_past, 0], dim=0)

        # apply degradation operator to the conditioning frames
        if self.degradation_operator is not None:
            x = self.degradation_operator(x)

        if self.transforms is not None:
            # Perform any data transforms if specified
            for T in self.transforms:
                x, _ = T((x, param_dict))

        # return spatial/temporal grid, frames and parameters
        spatial_grid = (self.x, self.y) if self.y is not None else (self.x,)

        if self.grid_transform is not None:
            # applying same grid transform to target frames
            x, spatial_grid = self.grid_transform(x, spatial_grid)

        ic_index = torch.tensor([ic_index], dtype=float)
        return spatial_grid, self.t, x, time_frames, ic_index, param_dict
