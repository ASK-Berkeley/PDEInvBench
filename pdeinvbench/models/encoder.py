from math import ceil
from typing import List

import torch
import torch.nn as nn
from jaxtyping import Float
from neuralop.models import FNO
from scOT.model import ScOT, ScOTConfig, ScOTOutput
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from pdeinvbench.utils.types import (
    PDE,
    PDE_NUM_PARAMETERS,
    PDE_NUM_SPATIAL,
    PDE_PARTIALS,
    PDE_SPATIAL_SIZE,
    PDE_TRAJ_LEN,
)


def resolve_number_input_channels(
    n_past: int, data_channels: int, use_partials: bool, pde: PDE
) -> int:
    """
    Returns the number of input channels for a pde given args:
        - n_past
        - data_channels
        - use_partials
    """

    num_partials = PDE_PARTIALS[pde]

    if use_partials:
        # each timestep gets partials appended to it
        data_channels += num_partials * data_channels

    in_channels = n_past * data_channels

    return in_channels


class FNOEncoder(FNO):
    """
    Wrapper around FNO that figures out the input channels based
    on the number of past frames and partial derivatives.
    :param n_modes: Number of modes to use in the FNO.
    :param n_layers: Number of layers in the FNO.
    :param n_past: Number of past frames to use.
    :param pde: PDE to use for the partial derivatives.
    :param data_channels: Number of channels per timestep in the native input data.
    :param hidden_channels: Number of channels in the hidden layers.
    :param use_partials: Whether to use partial derivatives as input (only applicable to the inverse problem)
    """

    def __init__(
        self,
        n_modes: int,
        n_layers: int,
        n_past: int,
        n_future: int,
        pde: PDE,
        data_channels: int,
        hidden_channels: int,
        use_partials: bool,
        batch_size: int,
    ):

        if use_partials:
            # if using partials, we are in inverse model mode
            # therefore, there will be a downsampler after the encoder,
            # need to preserve the number of channels
            out_channels = hidden_channels

        else:
            out_channels = hidden_channels

        # figure out the number of input channels

        self.use_partials = use_partials

        in_channels = resolve_number_input_channels(
            n_past=n_past,
            data_channels=data_channels,
            use_partials=use_partials,
            pde=pde,
        )

        # expand modes based on dimensionality of PDE
        n_modes = [n_modes] * PDE_NUM_SPATIAL[pde]
        self.batch_size = batch_size
        super(FNOEncoder, self).__init__(
            n_modes=n_modes,
            n_layers=n_layers,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
        )

    def forward(self, x, **kwargs):
        return super().forward(x)


class Resnet(nn.Module):
    """
    Wrapper around FNO replacing FNO convolution blocks with Resnet Blocks.
    """

    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        super(Resnet, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.hidden_channels = kwargs["hidden_channels"]
        self.n_layers = kwargs["n_layers"]
        self.batch_size = kwargs["batch_size"]
        self.in_block = BasicBlock(
            inplanes=self.in_channels,
            planes=self.hidden_channels,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=nn.BatchNorm2d,
        )
        self.in_block = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.hidden_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                self.hidden_channels,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                self.hidden_channels,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
        )
        self.resnet_blocks = nn.ModuleList(
            [
                BasicBlock(
                    inplanes=self.hidden_channels,
                    planes=self.hidden_channels,
                    stride=1,
                    downsample=None,
                    groups=1,
                    base_width=64,
                    dilation=1,
                    norm_layer=nn.BatchNorm2d,
                )
                for _ in range(kwargs["n_layers"])
            ]
        )

    def forward(self, x, output_shape=None, **kwargs):
        """CN-Resnet's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """
        x = self.in_block(x)
        for layer_idx in range(self.n_layers):
            x = self.resnet_blocks[layer_idx](x)

        return x


class ResnetEncoder(Resnet):
    """
    Wrapper around Resnet that figures out the input channels based
    on the number of past frames and partial derivatives.
    :param n_layers: Number of layers in the Resnet.
    :param n_past: Number of past frames to use.
    :param pde: PDE to use for the partial derivatives.
    :param data_channels: Number of channels per timestep in the native input data.
    :param hidden_channels: Number of channels in the hidden layers.
    :param use_partials: Whether to use partial derivatives as input (only applicable to the inverse problem)
    :param mode: One of "oneshot", "autoregressive", "grid_to_soln"
    """

    def __init__(
        self,
        n_layers: int,
        n_past: int,
        n_future: int,
        pde: PDE,
        data_channels: int,
        hidden_channels: int,
        use_partials: bool,
        batch_size: int,
    ):

        # figure out the number of output channels
        if use_partials:
            # if using partials, we are in inverse model mode
            # therefore, there will be a downsampler after the encoder,
            # need to preserve the number of channels
            out_channels = hidden_channels

        else:
            out_channels = hidden_channels  # data_channels

        self.use_partials = use_partials
        self.pde = pde
        in_channels = resolve_number_input_channels(
            n_past=n_past,
            data_channels=data_channels,
            use_partials=use_partials,
            pde=pde,
        )

        super(ResnetEncoder, self).__init__(
            n_layers=n_layers,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            pde=pde,
        )

    def forward(self, x, **kwargs):
        if self.pde == PDE.KortewegDeVries1D:
            x = x.unsqueeze(2)
            x = super().forward(x)
            return x[:, :, 0, :]
        return super().forward(x)


class ScOTEncoder(nn.Module):
    config: ScOTConfig
    backbone: ScOT
    n_past: int
    in_channels: int
    use_partials: bool
    patch_size: int
    padding_mode: str = "constant"

    def __init__(
        self,
        # backbone args
        embed_dim: int,  # patch embedding
        n_layers: int,
        hidden_size: int,
        patch_size: int,
        num_heads: list[int],
        skip_connections: list[int],
        depths: list[int],
        # Our args
        use_partials: bool,
        data_channels: bool,
        n_past: int,
        pde: PDE,
        **kwargs,
    ):
        super(ScOTEncoder, self).__init__()

        self.n_past = n_past
        self.use_partials = use_partials
        self.patch_size = patch_size
        self.in_channels = resolve_number_input_channels(
            n_past=self.n_past,
            use_partials=self.use_partials,
            data_channels=data_channels,
            pde=pde,
        )

        # All pdes are on square grids
        if (
            PDE_NUM_SPATIAL[pde] == 2
            and PDE_SPATIAL_SIZE[pde][0] != PDE_SPATIAL_SIZE[pde][1]
        ):
            self.spatial_size = max(PDE_SPATIAL_SIZE[pde])
        else:
            self.spatial_size = PDE_SPATIAL_SIZE[pde][0]
        self.pde = pde
        self.config = ScOTConfig(
            num_layers=n_layers,
            num_channels=self.in_channels,
            num_out_channels=hidden_size,
            depths=depths,
            num_heads=num_heads,
            skip_connections=skip_connections,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            image_size=self.spatial_size,
            **kwargs,
        )

        self.backbone = ScOT(self.config)

    def _pad_input(
        self, x: Float[torch.Tensor, "batch channels nx ny"]
    ) -> tuple[
        Float[torch.Tensor, "batch channels nx ny"], tuple[int, int, int, int] | None
    ]:
        _, _, nx, ny = x.shape
        if nx != ny:
            if nx % self.patch_size != 0:
                pad_values = (0, 0, self.spatial_size - nx % self.spatial_size, 0)
                x = nn.functional.pad(x, pad_values)
            if ny % self.patch_size != 0:
                pad_values = (0, 0, 0, self.spatial_size - ny % self.spatial_size)
                x = nn.functional.pad(x, pad_values)
            return x, None

        else:
            total_pad: int = (
                self.patch_size - (nx % self.patch_size)
            ) % self.patch_size
            left_pad, right_pad = total_pad // 2, ceil(total_pad / 2)
            assert (
                left_pad + right_pad == total_pad
            ), f"Incorrect swin padding {left_pad} + {right_pad} = {total_pad}"
            if left_pad or right_pad:
                pad_vals = (left_pad, right_pad, left_pad, right_pad)
                return (
                    torch.nn.functional.pad(
                        x,
                        pad_vals,
                        mode=self.padding_mode,
                        value=0,
                    ),
                    pad_vals,
                )
            return x, None

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels nx ny"],
        t: Float[torch.Tensor, "batch nt"] | None = None,
        **kwargs,
    ) -> Float[torch.Tensor, "batch outdim nx ny"]:
        # Check if we need to pad the input
        if self.pde == PDE.KortewegDeVries1D:
            x = x.unsqueeze(2)
            x = x.repeat(1, 1, x.shape[-1], 1)
        x, pad_vals = self._pad_input(x)
        output: ScOTOutput = self.backbone(pixel_values=x, time=t).output
        if pad_vals:
            # undo padding
            l, r, _, _ = pad_vals
            output = output[..., l:-r, l:-r]
        if self.pde == PDE.KortewegDeVries1D:
            output = output[:, :, :1, :]
        return output


class DeepONetTrunkNet(nn.Module):
    """
    MLP trunk on collocation coordinates. Uses the same ``hidden_channels`` width as the branch ResNet.
    """

    def __init__(self, hidden_channels: int, n_layers: int, pde: PDE):
        super(DeepONetTrunkNet, self).__init__()
        self.pde = pde
        if self.pde == PDE.KortewegDeVries1D:
            self.in_channels = 1
        else:
            self.in_channels = 2
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels), nn.GELU()
        )

        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_channels, self.hidden_channels),
                    nn.GELU(),
                )
                for _ in range(self.n_layers - 1)
            ]
        )
        self.out_proj = nn.Linear(self.hidden_channels, self.hidden_channels)

    def forward(self, spatial_grid, **kwargs):
        # Dataset uses ``(x,)`` for 1D or ``(x, y)`` for 2D
        if len(spatial_grid) == 1:
            x_grid = spatial_grid[0]
            y_grid = None
        else:
            x_grid, y_grid = spatial_grid[0], spatial_grid[1]

        if self.pde == PDE.KortewegDeVries1D:
            # (batch, nx) -> (batch, nx, 1) for Linear on last dim
            spatial_grid_input = x_grid.unsqueeze(-1)

            output = self.input_proj(spatial_grid_input)
            for block in self.residual_blocks:
                output = output + block(output)
            output = self.out_proj(output)

            return output.permute(0, 2, 1)

        assert y_grid is not None

        # (batch, nx) -> (batch, nx, ny)
        x_grid_input = x_grid.unsqueeze(-1).repeat(1, 1, y_grid.shape[-1])
        # (batch, ny) -> (batch, nx, ny)
        y_grid_input = y_grid.unsqueeze(1).repeat(1, x_grid.shape[-1], 1)

        # (batch, nx, ny) -> (batch, 2, nx, ny)
        collocation_grid_input = torch.stack([x_grid_input, y_grid_input], dim=1)

        # (batch, 2, nx, ny) -> (batch, nx, ny, 2)
        collocation_grid_input_transposed = collocation_grid_input.permute(0, 2, 3, 1)

        # (batch, nx, ny, 2) -> (batch, nx, ny, hidden_channels)
        output = self.input_proj(collocation_grid_input_transposed)
        for block in self.residual_blocks:
            output = output + block(output)
        output = self.out_proj(output)

        # (batch, nx, ny, hidden_channels) -> (batch, hidden_channels, nx, ny)
        output = output.permute(0, 3, 1, 2)
        return output


class DeepONetEncoder(nn.Module):
    """
    Wrapper around DeepONet that figures out the input channels based
    on the number of past frames and partial derivatives.
    """

    def __init__(self, branch_net: nn.Module, trunk_net: nn.Module):
        super(DeepONetEncoder, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.pde = self.branch_net.pde
        self.use_partials = self.branch_net.use_partials
        self.in_channels = self.branch_net.in_channels

    def forward(self, x, spatial_grid, **kwargs):
        branch_out = self.branch_net(x)
        trunk_out = self.trunk_net(spatial_grid)
        if branch_out.shape != trunk_out.shape:
            raise ValueError(
                f"DeepONet Branch and trunk output shapes do not match: {branch_out.shape} != {trunk_out.shape}"
            )
        return branch_out * trunk_out
