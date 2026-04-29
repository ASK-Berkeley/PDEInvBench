import math
from functools import partial
from os.path import join
from typing import Callable, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neuralop.models import FNO
from torch import vmap
from torch.distributions import Categorical

from pdeinvbench.data.transforms import (
    collapse_time_and_channels,
    expand_time_and_channels,
)
from pdeinvbench.data.utils import unnormalize_params
from pdeinvbench.losses import get_pde_residual_function
from pdeinvbench.models import MLP, ConvHead
from pdeinvbench.models.encoder import DeepONetEncoder, DeepONetTrunkNet
from pdeinvbench.utils.types import (
    PDE,
    PDE_NUM_CHANNELS,
    PDE_NUM_SPATIAL,
    PDE_PARAM_VALUES,
    PDE_SPATIAL_SIZE,
)


class InverseModel(nn.Module):
    """
    Model that predicts the parameters of a PDE given a set of conditioning frames.
    :param paramnet: Model that predicts the parameters.
    """

    def __init__(
        self,
        paramnet: nn.Module,
    ):
        super(InverseModel, self).__init__()

        self.paramnet = paramnet
        self.pde = paramnet.pde
        self.param_values = paramnet.param_values
        self.pde_residual = get_pde_residual_function(self.pde)
        self.params_to_predict = self.paramnet.params_to_predict
        self.use_partials = self.paramnet.encoder.use_partials
        self.dropout = self.paramnet.dropout
        # Set up data transform and detransform functions
        self.data_transform = None
        self.data_detransform = None
        if PDE_NUM_SPATIAL[self.pde] > 1:
            self.data_transform = vmap(collapse_time_and_channels)
            self.data_detransform = vmap(
                partial(
                    expand_time_and_channels, num_channels=PDE_NUM_CHANNELS[self.pde]
                )
            )

    def forward(self, solution_field, true_params, spatial_grid, t, gumbel=False):
        # get the partial derivatives and true pde residual, and append to the input if needed
        with torch.no_grad():
            # PDE residual function expects time and channel dimensions to be uncollapsed
            sf = (
                solution_field
                if self.data_transform is None
                else self.data_detransform(solution_field)
            )
            true_residual, partials = self.pde_residual(
                sf, true_params, spatial_grid, t, return_partials=True
            )
            if self.use_partials:
                partials = (
                    partials
                    if self.data_transform is None
                    else self.data_transform(partials)
                )
                new_solution_field = torch.cat([solution_field, partials], dim=1)
            else:
                new_solution_field = solution_field

        # predict parameters
        pred_params = self.paramnet(
            new_solution_field, spatial_grid=spatial_grid, t=t, gumbel=gumbel
        )

        # compute PDE residual with predicted params (replace unpredicted params with true params)
        combined_params = {**true_params, **pred_params}

        pred_residual = self.pde_residual(sf, combined_params, spatial_grid, t)
        return combined_params, pred_residual, true_residual


class ParameterNet(nn.Module):
    """
    Model that predicts PDE parameters given conditioning frames.
    :param params_to_predict: list of strings of parameters to predict.
    :param pde: enum of the PDE to use for the residual calculation.
    :param encoder: Encoder model. Can be FNO or e.g ResNet
    :param downsampler: Convolutional downsampler model.
    :param mlp_hidden_size: Hidden size of the MLP.
    :param mlp_layers: Number of layers in the MLP.
    :param mlp_activation: Activation function in the MLP.
    :param logspace: Whether to predict the parameters in log space.
    :param normalize: Whether to normalize the parameters.
    :param condition_on_time: Whether or not to provide $t$ as input to the encoder
    :param mlp_type: Alternatively, may be "conv" which denotes pointwise convolution of darcy flow
    """

    def __init__(
        self,
        params_to_predict: List[str],
        pde: PDE,
        encoder: nn.Module,
        downsampler: nn.Module,
        mlp_hidden_size: int,
        mlp_layers: int,
        mlp_activation: str,
        mlp_dropout: float,
        logspace: bool,
        normalize: bool,
        downsample_factor: int,
        mlp_type: str = "mlp",
        condition_on_time: bool = False,
    ):
        super(ParameterNet, self).__init__()

        self.encoder = encoder
        self.condition_on_time = condition_on_time
        self.downsampler = downsampler
        self.params_to_predict = params_to_predict
        self.logspace = logspace
        self.normalize = normalize
        self.pde = pde
        self.input_size = PDE_SPATIAL_SIZE[pde]
        self.param_values = None
        self.dropout = mlp_dropout
        if self.pde != PDE.DarcyFlow2D:
            self.param_values = {
                param: torch.tensor(PDE_PARAM_VALUES[pde][param]).to(
                    torch.device(torch.cuda.current_device())
                )
                for param in params_to_predict
            }

        # Consistency checks
        assert not (
            logspace and normalize
        ), "Cannot use logspace and normalize together."

        # get the input shape into the MLP by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, self.encoder.in_channels, *self.input_size)
            dummy_time = (
                torch.ones(1, self.encoder.__dict__.get("n_past", 1))
                if self.condition_on_time
                else None
            )
            if isinstance(self.encoder, DeepONetEncoder):
                if self.pde == PDE.KortewegDeVries1D:
                    dummy_grid = (torch.zeros(1, int(self.input_size[0])),)
                else:
                    nx, ny = int(self.input_size[0]), int(self.input_size[1])
                    dummy_grid = (
                        torch.zeros(1, nx),
                        torch.zeros(1, ny),
                    )
                dummy = self.encoder(dummy, dummy_grid, t=dummy_time)
            else:
                dummy = self.encoder(dummy, t=dummy_time)
            dummy = self.downsampler(dummy)
            encoder_out_channels = dummy.shape[1]
            dummy = torch.flatten(dummy, start_dim=1)
            mlp_input_size = dummy.shape[1]

        self.heads: nn.ModuleList

        # The following calls set the heads
        if mlp_type == "mlp":
            self.generate_mlp_heads(
                mlp_input_size=mlp_input_size,
                mlp_hidden_size=mlp_hidden_size,
                mlp_dropout=mlp_dropout,
                mlp_layers=mlp_layers,
                mlp_activation=mlp_activation,
                params_to_predict=self.params_to_predict,
            )
        elif mlp_type == "conv":
            self.generate_conv_heads(
                in_dim=encoder_out_channels,
                hidden_dim=mlp_hidden_size,
                n_layers=mlp_layers,
                dropout=mlp_dropout,
                activation=mlp_activation,
                params_to_predict=self.params_to_predict,
            )

    def generate_conv_heads(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        activation: str,
        params_to_predict,
    ):
        self.heads = nn.ModuleList(
            [
                ConvHead(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=dropout,
                    activation=activation,
                )
                for param in params_to_predict
            ]
        )

    def generate_mlp_heads(
        self,
        mlp_input_size: int,
        mlp_hidden_size: int,
        mlp_dropout: float,
        mlp_layers: int,
        mlp_activation,
        params_to_predict,
    ):
        self.heads = nn.ModuleList(
            [
                MLP(
                    in_dim=int(mlp_input_size),
                    hidden_size=mlp_hidden_size,
                    dropout=mlp_dropout,
                    out_dim=1,
                    num_layers=mlp_layers,
                    activation=mlp_activation,
                )
                for param in params_to_predict
            ]
        )

    def forward(self, x, gumbel=False, t=None, spatial_grid=None):

        x = self.encoder(x, t=t, spatial_grid=spatial_grid)
        x = self.downsampler(x)
        # We follow different paths depending on the PDE
        if self.pde == PDE.DarcyFlow2D:
            preds = [head(x) for head in self.heads]
        else:
            x = torch.flatten(x, start_dim=1)
            # combine output from each head
            preds = [head(x) for head in self.heads]
        if self.logspace:
            preds = [torch.exp(pred) for pred in preds]

        # convert to a dictionary
        preds = {k: v for k, v in zip(self.params_to_predict, preds)}
        if self.normalize:
            preds = unnormalize_params(preds, self.pde)
        return preds
