from types import NoneType
from typing import Any, List, Mapping

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as plotly_express
import plotly.graph_objects as go
import torch
import wandb
from functorch.dim import tree_map
from lightning.pytorch.callbacks import Callback

import pdeinvbench.utils.pytorch_utils as ptu
from pdeinvbench.utils.logging_utils import (
    collect_loss_dicts,
)
from pdeinvbench.utils.types import PDE, TypeBatch1D, TypeBatch2D


class PDEParamErrorPlottingCallback(Callback):
    """
    Logs a set of errors stratified based on PDE parameter value.
    """

    def __init__(self):
        self.validation_step_loss_dicts = []
        """
        Loss dicts for validation and autoregressive validation. Each element is a tuple of (losses, pde_params).
        PDE_params comes directly from the dataloader.
        Loss: Dict[str, torch.Tensor] with keys (shape) 
        'data_loss' (B), 'residual_loss' (B), 'loss' (), 'data_loss_per_batch_element' (B), 'residual_per_batch_element' (B).
        """
        self.pde = None  # type: ignore
        self.params_to_predict = []

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # Clear memory of loss dicts
        self.validation_step_loss_dicts = []
        self.pde = pl_module.pde
        if self.pde == PDE.DarcyFlow2D:
            self.params_to_predict = ["index"]
        else:
            self.params_to_predict = pl_module.params_to_predict

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: TypeBatch1D | TypeBatch2D,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # Convert to numpy arrays
        collect_loss_dicts(
            outputs,
            batch,
            "residual_per_batch_element",
            self.validation_step_loss_dicts,
        )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: TypeBatch1D | TypeBatch2D,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx)

    def generate_pde_parameter_histogram(self, loss_tuples):
        """
        Generates a histogram of PDE parameter values vs loss
        """
        if len(loss_tuples) == 0:
            return None
        if self.pde == PDE.DarcyFlow2D:
            parameter_keys = ["index"]
        else:
            parameter_keys = loss_tuples[0][1].keys()
        plots = {}
        for parameter in parameter_keys:
            # Num batches x Batch size
            parameter_values = [
                ptu.torch_to_numpy(loss_tuple[1][parameter].ravel())
                for loss_tuple in loss_tuples
            ]
            parameter_values = np.concatenate(parameter_values, axis=0)

            residuals = [
                ptu.torch_to_numpy(loss_tuple[0]["residual_per_batch_element"])
                for loss_tuple in loss_tuples
            ]
            residuals = np.concatenate(residuals, axis=0)

            key_name = "param_loss_per_batch_element"

            data_or_param_loss = [loss_tuple[0][key_name]
                                  for loss_tuple in loss_tuples]

            if len(data_or_param_loss[0].shape) != 0:
                data_or_param_loss = np.concatenate(data_or_param_loss, axis=0)

            residual_fig = plotly_express.density_heatmap(
                x=parameter_values,
                y=residuals,
                nbinsx=20,
                nbinsy=20,
                title=f"Residual vs. {parameter}",
            )
            residual_fig.update_layout(
                xaxis_title=f"{parameter} Values",
                yaxis_title="Residual",
                title_x=0.5,
                margin_t=40,
            )
            title = "Parameter"
            data_or_param_loss_fig = plotly_express.density_heatmap(
                x=parameter_values,
                y=data_or_param_loss,
                nbinsx=20,
                nbinsy=20,
                title=f"{title} Loss vs. {parameter}",
            )
            data_or_param_loss_fig.update_layout(
                xaxis_title=f"{title} Loss (MSE)",
                yaxis_title=f"{parameter} Values",
                title_x=0.5,
                margin_t=40,
            )
            plots[parameter] = [residual_fig, data_or_param_loss_fig]
        return plots

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.on_validation_epoch_start(trainer, pl_module)

    def log_plots(self, prefix: str):
        plots = self.generate_pde_parameter_histogram(
            self.validation_step_loss_dicts)
        if plots is not None:
            for parameter, plots in plots.items():
                residual_fig, data_loss_fig = plots
                wandb.log(
                    {
                        f"{prefix}/residual_vs_{parameter}": residual_fig,
                        f"{prefix}/data_loss_vs_{parameter}": data_loss_fig,
                    }
                )
        plt.close()

    def log_parameter_predictions_table(self, loss_tuples, prefix: str = "test"):
        if len(loss_tuples) == 0:
            return None
        if self.pde == PDE.DarcyFlow2D:
            parameter_keys = ["index"]
        else:
            parameter_keys = loss_tuples[0][1].keys()
        plots = {}
        columns = ["ic_index", "true_parameters",
                   "predicted_parameters", "param_loss"]
        for parameter in parameter_keys:
            # Num batches x Batch size
            true_parameters = [
                ptu.torch_to_numpy(loss_tuple[1][parameter].ravel())
                for loss_tuple in loss_tuples
            ]
            true_parameters = np.concatenate(true_parameters, axis=0)

            residuals = [
                ptu.torch_to_numpy(loss_tuple[0]["residual_per_batch_element"])
                for loss_tuple in loss_tuples
            ]
            residuals = np.concatenate(residuals, axis=0)

            key_name = "param_loss_per_batch_element"

            data_or_param_loss = [
                ptu.torch_to_numpy(loss_tuple[0][key_name])
                for loss_tuple in loss_tuples
            ]

            if len(data_or_param_loss[0].shape) != 0:
                data_or_param_loss = np.concatenate(data_or_param_loss, axis=0)

            ic_index = [
                ptu.torch_to_numpy(loss_tuple[2]).ravel() for loss_tuple in loss_tuples
            ]
            timestamps = [
                ptu.torch_to_numpy(loss_tuple[3]).ravel() for loss_tuple in loss_tuples
            ]

            ic_index = np.concatenate(ic_index, axis=0)
            timestamps = np.concatenate(timestamps, axis=0)

            predicted_parameters = [
                ptu.torch_to_numpy(
                    loss_tuple[0]["predictions"][parameter]).ravel()
                for loss_tuple in loss_tuples
            ]

            predicted_parameters = np.concatenate(predicted_parameters, axis=0)

            data = [
                [
                    ic_index[i],
                    timestamps[i],
                    true_parameters[i],
                    predicted_parameters[i],
                    data_or_param_loss[i],
                ]
                for i in range(len(ic_index))
            ]
            table = wandb.Table(
                data=data,
                columns=[
                    "ic_index",
                    "timestamps",
                    "true_parameters",
                    "predicted_parameters",
                    "param_loss",
                ],
            )
            wandb.log(
                {f"{prefix}/parameter_predictions_table_{parameter}": table})

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # Plot error stratified by PDE parameter value
        self.log_plots("validation")
        # Clear caches
        self.validation_step_loss_dicts = []

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.on_validation_epoch_start(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.log_plots("test")
        self.log_parameter_predictions_table(
            self.validation_step_loss_dicts, "test")
        # Clear caches
        self.validation_step_loss_dicts = []


class PDEParamErrorTestTimeTailoringCallback(PDEParamErrorPlottingCallback):
    """
    Logs errors before and after tailoring, stratified by PDE parameter value.
    """

    def __init__(self):
        super().__init__()
        """
        Loss dicts for test time tailoring. Each element is a tuple of (losses, pde_params).
        PDE_params comes directly from the dataloader.
        Loss: Dict[str, torch.Tensor] with keys (shape) 
        'data_loss' (B), 'residual_loss' (B), 'loss' (), 'data_loss_per_batch_element' (B), 'residual_per_batch_element' (B).
        """
        self.pre_tailored_loss_dicts = []
        self.post_tailored_loss_dicts = []
        self.params_to_predict = []
        self.pde = None  # type: ignore

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.pde = pl_module.pde
        if self.pde == PDE.DarcyFlow2D:
            # 'Coeff' is a 2D parameter field, index corresponds to the filename of the parameter in the 2D field
            self.params_to_predict = ["index"]
        else:
            self.params_to_predict = pl_module.params_to_predict

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: TypeBatch1D | TypeBatch2D,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if (
            not hasattr(pl_module, "num_tailoring_steps")
            or pl_module.tailoring_optimizer is None
            or pl_module.num_tailoring_steps == 0
        ):
            return
        if "pre_tailored_metrics" in outputs:

            collect_loss_dicts(
                outputs["pre_tailored_metrics"],
                batch,
                "param_loss_per_batch_element",
                self.pre_tailored_loss_dicts,
            )
        collect_loss_dicts(
            {key: v for key, v in outputs.items() if key != "pre_tailored_metrics"},
            batch,
            "param_loss_per_batch_element",
            self.post_tailored_loss_dicts,
        )

    def log_tables(self):
        # take the param_loss_per_batch_element for the pre and post tailored metrics

        pre_tailored_data_loss = [
            loss_tuple[0]["param_loss_per_batch_element"]
            for loss_tuple in self.pre_tailored_loss_dicts
        ]
        post_tailored_data_loss = [
            loss_tuple[0]["param_loss_per_batch_element"]
            for loss_tuple in self.post_tailored_loss_dicts
        ]

        parameter_values = [
            ptu.torch_to_numpy(
                loss_tuple[1][self.params_to_predict[0]].ravel())
            for loss_tuple in self.pre_tailored_loss_dicts
        ]

        pre_tailored_parameter_values = [
            ptu.torch_to_numpy(
                loss_tuple[0]["predictions"][self.params_to_predict[0]].ravel()
            )
            for loss_tuple in self.pre_tailored_loss_dicts
        ]
        post_tailored_parameter_values = [
            ptu.torch_to_numpy(
                loss_tuple[0]["predictions"][self.params_to_predict[0]].ravel()
            )
            for loss_tuple in self.post_tailored_loss_dicts
        ]

        parameter_values = np.concatenate(parameter_values, axis=0)
        pre_tailored_parameter_values = np.concatenate(
            pre_tailored_parameter_values, axis=0
        )
        post_tailored_parameter_values = np.concatenate(
            post_tailored_parameter_values, axis=0
        )

        pre_tailored_data_loss = np.concatenate(pre_tailored_data_loss, axis=0)
        post_tailored_data_loss = np.concatenate(
            post_tailored_data_loss, axis=0)

        # log table containing paramter value, pre tailored data loss, post tailored data loss
        data = [
            [
                parameter_values[i],
                pre_tailored_data_loss[i],
                post_tailored_data_loss[i],
                pre_tailored_parameter_values[i],
                post_tailored_parameter_values[i],
            ]
            for i in range(len(parameter_values))
        ]
        table = wandb.Table(
            data=data,
            columns=[
                "parameter_value",
                "pre_tailored_data_loss",
                "post_tailored_data_loss",
                "pre_tailored_parameter_value",
                "post_tailored_parameter_value",
            ],
        )
        wandb.log({"tailoring_data_loss_table": table})

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (
            not hasattr(pl_module, "num_tailoring_steps")
            or pl_module.tailoring_optimizer is None
            or pl_module.num_tailoring_steps == 0
        ):
            return
        self.log_tables()
        self.pre_tailored_loss_dicts = []
        self.post_tailored_loss_dicts = []


class TailoringTimingMetricsCallback(Callback):
    """
    Logs the timing metrics for the tailoring step.
    """

    def __init__(self):
        self.tailoring_timing_metrics = {}

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: TypeBatch1D | TypeBatch2D,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if (
            not hasattr(pl_module, "tailoring_optimizer")
            or pl_module.tailoring_optimizer is None
            or pl_module.num_tailoring_steps == 0
        ):
            return

        if "tailoring_timing_metrics" in outputs:
            self.tailoring_timing_metrics[dataloader_idx] = outputs[
                "tailoring_timing_metrics"
            ]

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for dataloader_idx in self.tailoring_timing_metrics:
            wandb.log(
                {
                    f"tailoring_timing_metrics_dataloader_{dataloader_idx}": self.tailoring_timing_metrics[
                        dataloader_idx
                    ]
                }
            )


class InverseErrorByTailoringStepCallback(Callback):
    """
    Helper callback that plots the error by tailoring step. On the Y-axis is the metric and the X-axis is the tailoring step.
    Uses plotly to generate the plot and plots to W&B.
    This is specifically for PINO and tailoring.
    """

    def __init__(self):
        # Required class variables - reset on val epoch start
        self.errors_by_tailor_step = {}

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (
            not hasattr(pl_module, "tailoring_optimizer")
            or pl_module.tailoring_optimizer is None
            or pl_module.num_tailoring_steps == 0
        ):
            return
        self.errors_by_tailor_step = {}

    def generate_plots(self, pl_module, loader_idx=0):
        """
        Generates the plots for the data and residual error by tailoring step.
        """
        num_tailoring_steps = pl_module.num_tailoring_steps
        metric_plots = {}

        for error_metric in self.errors_by_tailor_step[loader_idx]:
            metric_data_by_tailor_step = np.asarray(
                self.errors_by_tailor_step[loader_idx][error_metric]
            )

            # Calculate mean across tailoring steps
            mean_metric_data_by_tailor_step = np.mean(
                metric_data_by_tailor_step, axis=0
            )

            # Calculate y-axis bounds with some padding (e.g., 5% of the range)
            y_min = np.min(metric_data_by_tailor_step)
            y_max = np.max(metric_data_by_tailor_step)
            y_range = y_max - y_min
            y_axis_min = y_min - 0.05 * y_range  # Add 5% padding below min
            y_axis_max = y_max + 0.05 * y_range  # Add 5% padding above max

            # Create data for the table
            data = [
                [x, y]
                for (x, y) in zip(
                    list(range(num_tailoring_steps)
                         ), mean_metric_data_by_tailor_step
                )
            ]
            table = wandb.Table(
                data=data, columns=["tailor_steps", f"mean_{error_metric}"]
            )

            # Create a Plotly figure for custom y-axis bounds
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(num_tailoring_steps)),
                    y=mean_metric_data_by_tailor_step,
                    mode="lines",
                    name=f"Mean {error_metric}",
                )
            )
            fig.update_layout(
                title=f"Tailoring Steps vs Mean {error_metric}",
                xaxis_title="Tailoring Steps",
                yaxis_title=f"Mean {error_metric}",
                # Set y-axis bounds
                yaxis=dict(range=[y_axis_min, y_axis_max]),
            )

            # Log the Plotly figure to WandB
            metric_plots[error_metric] = wandb.Plotly(fig)
        return metric_plots

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: TypeBatch1D | TypeBatch2D,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """
        After each batch, we accumulate the metric for each tailoring step.
        """
        if (
            not hasattr(pl_module, "tailoring_optimizer")
            or pl_module.tailoring_optimizer is None
            or pl_module.num_tailoring_steps == 0
        ):
            return

        tailoring_metrics = outputs["tailoring_metrics"]
        if dataloader_idx not in self.errors_by_tailor_step:
            self.errors_by_tailor_step[dataloader_idx] = {}
        current_dataloader = trainer.test_dataloaders[dataloader_idx]

        for metric, metric_data in tailoring_metrics.items():
            if metric not in self.errors_by_tailor_step[dataloader_idx]:
                self.errors_by_tailor_step[dataloader_idx][metric] = []
            if "per_batch_element" in metric and pl_module.tailor_per_batch:
                current_batch_size = metric_data[0].shape[0]
                elements_to_add = pl_module.batch_size - current_batch_size
                if current_batch_size != pl_module.batch_size:
                    for tailoring_step in range(pl_module.num_tailoring_steps):
                        step_ouput = metric_data[tailoring_step]

                        # Get the last element of the tensor
                        last_element = step_ouput[-1]

                        # Create a tensor with repeated last elements
                        repeated_elements = (
                            last_element.repeat(elements_to_add, 1)
                            if len(step_ouput.shape) > 1
                            else last_element.repeat(elements_to_add)
                        )

                        # Concatenate the original tensor with the repeated elements
                        metric_data[tailoring_step] = torch.cat(
                            [step_ouput, repeated_elements], dim=0
                        )

            self.errors_by_tailor_step[dataloader_idx][metric].append(
                metric_data)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (
            not hasattr(pl_module, "tailoring_optimizer")
            or pl_module.tailoring_optimizer is None
            or pl_module.num_tailoring_steps == 0
        ):
            return

        """
        After each epoch, we plot the metric by tailoring step.
        """

        for dataloader_idx in self.errors_by_tailor_step:
            tailoring_figures = self.generate_plots(pl_module, dataloader_idx)
            to_log = {}
            for tailoring_metric, err_fig in tailoring_figures.items():
                to_log[
                    f"tailoring_step_plots_dataloader_{dataloader_idx}/{tailoring_metric}"
                ] = err_fig

            wandb.log(to_log)
            plt.close()
