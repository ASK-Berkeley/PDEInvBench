import os
from typing import Callable, Dict, List, Tuple, Union
import time
import lightning as L
import torch
import typeguard
from functorch.dim import tree_map
from jaxtyping import Float, jaxtyped
from lightning.pytorch.utilities import grad_norm
from pdeinvbench.utils.pytorch_utils import compute_grad_norm
from lightning.pytorch.loggers import WandbLogger
from pdeinvbench.data.utils import unnormalize_params
from pdeinvbench.utils.logging_utils import get_best_model_weights
from copy import deepcopy

from pdeinvbench.losses import (
    get_param_metric,
    get_pde_residual_function,
    pde_residual_reduction,
)
from pdeinvbench.losses.metrics import (
    classification_metrics_darcy_flow,
    param_relative_loss,
)
from pdeinvbench.models.inverse_model import InverseModel
from pdeinvbench.utils import pytorch_utils as ptu
from pdeinvbench.utils.types import (
    PDE,
    DataMetrics,
    ParamMetrics,
    TypeAutoRegressiveInitFrames,
    TypeAutoRegressivePredFrames,
    TypeBatch,
    TypeBatch1D,
    TypeBatch2D,
    TypeLossDict,
    TypeParam,
    TypePredict1D,
    TypePredict2D,
)
from pdeinvbench.lightning_modules.inversemodule import InverseModule


class InverseTestTimeTailoringModule(InverseModule):
    """
    InverseModule with test time tailoring
    :param num_tailoring_steps: Number of tailoring steps
    :param tailoring_optimizer: Optimizer for tailoring steps
    :param tailor_per_batch: Whether to tailor per batch or per element
    :param tailor_anchor_loss_weight: Weight for the anchor loss
    :param tailor_residual_loss_weight: Weight for the residual loss
    """

    def __init__(
        self,
        *args,
        num_tailoring_steps: int = 0,
        tailoring_optimizer: torch.optim.Optimizer = None,
        tailor_per_batch: bool = False,
        tailor_anchor_loss_weight: float = 0,
        tailor_residual_loss_weight: float = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_tailoring_steps = num_tailoring_steps
        self.tailoring_optimizer = tailoring_optimizer
        self.tailor_per_batch = tailor_per_batch
        self.tailor_anchor_loss_weight = tailor_anchor_loss_weight
        self.tailor_residual_loss_weight = tailor_residual_loss_weight

    @torch.enable_grad()
    def tailor(
        self,
        batch: Union[TypeBatch1D, TypeBatch2D],
        stage: str,
        dataloader_idx: int = 0,
        true_params: any = None,
    ) -> Union[
        TypePredict1D,
        TypePredict2D,
    ]:
        """
        As is, pytorch lightning doesn't support test time tailoring out of the box. As a result, we use a separate optimizer for the tailoring step.
        The tailoring step is performed on the model using :self.tailoring_optimizer: and the loss is computed using :self.pde_residual:, calls super().predict_with_loss to get the loss.
        :param batch: Batch of data.
        :param stage: Which of "train", "validation", "test","tailoring" is the current stage.
        :param dataloader_idx: Index of the dataloader.
        Returns the appended prediction, appended target and losses, same as :super().predict_with_loss:.
        """
        timing_metrics = {"tailoring_step_time": [], "tailoring_total_time": None}
        total_tailoring_start_time = time.time()
        # Each tailoring step is isolated so we make a copy of the model
        base_model = self.model
        model = deepcopy(base_model).to(next(base_model.parameters()).device)
        # We remap self.model to the tailored model to enable calling super().predict_with_loss
        self.model = model
        self.model.train()
        optimizer = self.tailoring_optimizer(model.parameters())
        aux_metrics = {"grad_norm": []}

        for tailoring_step in range(self.num_tailoring_steps):
            tailoring_step_start_time = time.time()
            with torch.enable_grad():
                optimizer.zero_grad()
                predicted_pde_params, _, loss = self.predict_with_loss(
                    batch, "tailoring"
                )

                if "timing_metrics" in loss:
                    del loss["timing_metrics"]

                for key in predicted_pde_params:
                    loss[f"tailored_vs_init_predicted_relative_param_loss_{key}"] = (
                        loss[f"relative_param_loss_{key}"]
                    )
                if self.pde != PDE.DarcyFlow2D:
                    tailored_vs_true_params_relative_error = param_relative_loss(
                        predicted_pde_params,
                        true_params,
                        reduction="mean",
                    )
                else:
                    tailored_vs_true_params_relative_error = param_relative_loss(
                        {"coeff": predicted_pde_params["coeff"]},
                        {"coeff": true_params["coeff"]},
                        reduction="mean",
                    )

                if self.tailor_anchor_loss_weight == 0:
                    # Only backprop based on the residual loss
                    total_loss = (
                        self.tailor_residual_loss_weight * loss["residual_loss"]
                    )
                else:
                    # sum anchor param loss and residual loss and backprop instead
                    total_loss = (
                        self.tailor_residual_loss_weight * loss["residual_loss"]
                    ) + (self.tailor_anchor_loss_weight * loss["param_loss"])
                total_loss.backward()
                optimizer.step()
                # Aux metrics for debugging
                grad_norm = compute_grad_norm(self.model, None)
                for param in tailored_vs_true_params_relative_error:
                    if (
                        f"tailored_vs_true_params_relative_error_{param}"
                        not in aux_metrics
                    ):
                        aux_metrics[
                            f"tailored_vs_true_params_relative_error_{param}"
                        ] = []
                    if self.tailor_per_batch:
                        metric_value = tailored_vs_true_params_relative_error[param]
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.detach().cpu()
                        else:
                            metric_value = metric_value
                        aux_metrics[
                            f"tailored_vs_true_params_relative_error_{param}"
                        ].append(metric_value)
                    else:
                        aux_metrics[
                            f"tailored_vs_true_params_relative_error_{param}"
                        ].append(tailored_vs_true_params_relative_error[param].item())
                aux_metrics["grad_norm"].append(grad_norm)
                for metric, value in loss.items():
                    if metric not in aux_metrics:
                        aux_metrics[metric] = []
                    if self.tailor_per_batch:
                        metric_value = value
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.detach().cpu()
                        else:
                            metric_value = metric_value
                        aux_metrics[metric].append(metric_value)
                    else:
                        aux_metrics[metric].append(value.item())
                if "total_loss" not in aux_metrics:
                    aux_metrics["total_loss"] = []
                if "optimizing_residual_loss" not in aux_metrics:
                    aux_metrics["optimizing_residual_loss"] = []
                if "optimizing_anchor_loss" not in aux_metrics:
                    aux_metrics["optimizing_anchor_loss"] = []
                aux_metrics["total_loss"].append(total_loss.item())
                aux_metrics["optimizing_residual_loss"].append(
                    loss["residual_loss"].item()
                )
                aux_metrics["optimizing_anchor_loss"].append(loss["param_loss"].item())
            torch.cuda.empty_cache()
            tailoring_step_end_time = time.time()
            if "tailoring_step_time" not in aux_metrics:
                aux_metrics["tailoring_step_time"] = []
            aux_metrics["tailoring_step_time"].append(
                torch.tensor(tailoring_step_end_time - tailoring_step_start_time)
            )
        with torch.no_grad():
            batch[-1] = true_params
            final_appended_prediction, final_appended_target, final_losses = (
                self.predict_with_loss(batch, stage)
            )

        # We delete the model and optimizer to prevent memory leaks
        optimizer.zero_grad(set_to_none=True)
        del model, optimizer
        torch.cuda.empty_cache()
        # Reset the model to the base model
        self.model = base_model
        total_tailoring_end_time = time.time()
        total_tailoring_time = total_tailoring_end_time - total_tailoring_start_time
        final_losses["total_tailoring_time"] = torch.tensor(total_tailoring_time)
        return (
            final_appended_prediction,
            final_appended_target,
            {
                "final_losses": final_losses,
                "aux_metrics": aux_metrics,
            },
        )

    @jaxtyped(typechecker=typeguard.typechecked)
    def test_step(
        self,
        batch: Union[TypeBatch1D, TypeBatch2D],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> TypeLossDict:
        stage = "test"
        self.model.train()
        test_losses = self.validation_step(batch, batch_idx, dataloader_idx, stage)
        # return test_losses if not tailoring
        if self.num_tailoring_steps == 0 or self.tailoring_optimizer is None:
            return test_losses
        else:
            self.log_metrics(stage, test_losses, prefix_dir="pre-tailored")
            # replace the true PDE params with the predicted PDE params for tailoring anchor loss
            true_pde_params = batch[-1].copy()
            for param, predicted_param in test_losses["predictions"].items():
                if self.pde != PDE.DarcyFlow2D:
                    batch[-1][param] = predicted_param.detach()
                else:
                    batch[-1][param] = torch.tensor(predicted_param).to(
                        next(self.model.parameters()).device
                    )
                    true_pde_params = tree_map(
                        lambda x: torch.tensor(x).to(
                            next(self.model.parameters()).device
                        ),
                        true_pde_params,
                    )

        if self.tailor_per_batch:
            predictions, targets, loss_dict = self.tailor(
                batch, stage, dataloader_idx, true_pde_params
            )

            losses, aux_metrics = (
                loss_dict["final_losses"],
                loss_dict["aux_metrics"],
            )
            losses["predictions"] = predictions
            losses["targets"] = targets
        else:
            # Add a second "batch dimension" to the batch. As a result, we manually loop over the batch dimension.
            # Ideally, we would use vmap over the real batch dimension, but there are inplace ops in the forward pass of FNO that is not vectorizable.
            # Related github issue with a minimal example: https://github.com/pytorch/pytorch/issues/103329
            # Specifically, inplace ops such as: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/spectral_convolution.py#L459 are not vectorizable.

            # PDE param is a dict so we have to handle it separately. Each parameter is a tensor with shape (B, 1). We unsqueeze it such that the tensor is now (B, 1, 1)
            # When we loop over and index the real batch dimension, the PDE parameter will be a tensor of shape (1, 1), reprsenting a batch of size 1. This ensure PDE param also has a second batch dim.
            batch = tree_map(lambda x: x.unsqueeze(1), batch)
            batch_size = batch[0][0].shape[0]
            losses = []
            aux_metrics = {"grad_norm": []}
            appended_predictions = []
            appended_targets = []
            for idx in range(batch_size):
                single_input = tree_map(lambda x: x[idx], batch)
                single_true_params = tree_map(
                    lambda x: x[idx].unsqueeze(1), true_pde_params
                )

                predictions, targets, curr_loss_dict = self.tailor(
                    single_input, stage, dataloader_idx, single_true_params
                )
                loss, single_aux_metrics = (
                    curr_loss_dict["final_losses"],
                    curr_loss_dict["aux_metrics"],
                )

                appended_predictions.append(predictions)
                appended_targets.append(targets)
                losses.append(loss)
                # Append the aux metrics
                for k in single_aux_metrics.keys():
                    if k not in aux_metrics:
                        aux_metrics[k] = []
                    aux_metrics[k].append(single_aux_metrics[k])

            # Stack the results back into a tensor
            appended_predictions = {
                key: torch.stack(
                    [prediction[key] for prediction in appended_predictions], dim=0
                )
                for key in appended_predictions[0].keys()
            }
            appended_targets = {
                key: torch.stack([target[key] for target in appended_targets], dim=0)
                for key in appended_targets[0].keys()
            }

            # Stack the losses back into a tensor
            losses = {
                key: torch.stack([loss[key] for loss in losses], dim=0)
                for key in losses[0].keys() if key != "timing_metrics"
            }
            # Only certain keys need to be squeezed
            losses = tree_map(lambda x: x if x.dim() < 2 else x.squeeze(1), losses)

            # Loss contains the following keys from super().predict_with_loss
            # 'data_loss' (B), 'residual_loss' (B), 'loss' (B), 'data_loss_per_batch_element' (B), 'residual_per_batch_element' (B), 'data_loss_per_frame' (B, T)
            # As a result of looping, we have to reduce the batch dimension for some keys.
            # Average the losses
            for key in losses.keys():
                if "per_batch_element" not in key:
                    losses[key] = losses[key].mean(dim=0)

            # Average the aux metrics and prep for logging
            for k in aux_metrics.keys():
                aux_metrics[k] = torch.tensor(aux_metrics[k]).mean(dim=0)
            losses["predictions"] = appended_predictions
            losses["targets"] = appended_targets
        self.log_metrics(stage, losses, prefix_dir="post-tailored")
        losses["tailoring_metrics"] = aux_metrics
        losses["pre_tailored_metrics"] = test_losses

        return losses

    def log_metrics(self, stage, losses, prefix_dir=""):
        super().log_metrics(stage, losses, prefix_dir)
        if "total_tailoring_time" in losses:
            self.log(
                os.path.join(stage, prefix_dir, "total_tailoring_time"),
                losses["total_tailoring_time"],
                on_step=False,
                on_epoch=True,
            )
