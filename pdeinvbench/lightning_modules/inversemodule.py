import os
from typing import Callable, Dict, List, Tuple, Union

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
import time

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
    TypeBatch,
    TypeBatch1D,
    TypeBatch2D,
    TypeLossDict,
    TypeParam,
    TypePredict1D,
    TypePredict2D,
)


class InverseModule(L.LightningModule):
    """
    Inverse Problem Module. Takes a set of conditioning frames from a PDE trajectory and predicts the value of the PDE parameter(s).
    :param model: Model.
    :param optimizer: Optimizer.
    :param lr_scheduler: Learning rate scheduler.
    :param pde: enum of the PDE to use for the residual calculation.
    :param param_loss_metric: Metric to use for the parameter loss.
    :param inverse_residual_loss_weight: Weight for the PDE residual loss obtained from the predicted parameters.
    :param n_past: Number of past frames to condition on.
    :param use_partials: whether to append the partial derivatives to the input.
    :param params_to_predict: list of strings of parameters to predict.
    :param residual_filter: Whether to use residual filtering.
    :param batch_size: Batch size.
    """

    def __init__(
        self,
        model: InverseModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        pde: PDE,
        param_loss_metric: ParamMetrics,
        inverse_residual_loss_weight: float,
        inverse_param_loss_weight: float,
        n_past: int,
        use_partials: bool,
        params_to_predict: List[str],
        residual_filter: bool,
        batch_size: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.old_inverse_model = None
        self.new_inverse_model = None
        pde = PDE(pde.value)
        self.pde_residual = get_pde_residual_function(pde)
        self.pde = pde
        param_loss_metric = ParamMetrics(param_loss_metric.value)
        self.param_loss_metric = get_param_metric(param_loss_metric)
        # self.param_loss_metric = get_param_metric(param_loss_metric)
        self.inverse_residual_loss_weight = inverse_residual_loss_weight
        self.inverse_param_loss_weight = inverse_param_loss_weight

        self.validation_step_outputs = []
        self.validation_step_targets = []
        """
        Loss dicts for validation. Each element is a tuple of (losses, pde_params).
        PDE_params comes directly from the dataloader.
        Loss: Dict[str, torch.Tensor] with keys (shape) 
        'data_loss' (B), 'residual_loss' (B), 'loss' (), 'data_loss_per_batch_element' (B), 'residual_per_batch_element' (B).
        """
        self.validation_step_loss_dicts = []
        self.n_past = n_past
        self.use_partials = use_partials
        self.params_to_predict = params_to_predict
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.residual_filter = residual_filter
        self.batch_size = batch_size

    @jaxtyped(typechecker=typeguard.typechecked)
    def predict_with_loss(
        self,
        batch: Union[TypeBatch1D, TypeBatch2D],
        stage: str,
        gumbel: bool = False,
    ) -> Tuple[TypeParam, TypeParam, Dict[str, Union[torch.tensor, TypeBatch]]]:
        """
        Common method which computes a prediction of the PDE parameter from the conditioning frames and returns the loss.
        :param batch: Batch of data.
        :param stage: Which of "train", "validation", "test","tailoring" is the current stage.
        """
        _, _, initial_frames, _, _, _ = batch
        start_time = time.time()
        # To fix batching issue. Sometimes the input batch_size is less than the defined batch_size.
        # This causes logging issues downstream during logging. This snippet repeats the final batch element
        # to match the specified input batch_size.
        num_repeat_elements = 0
        if initial_frames.shape[0] < self.batch_size:
            num_repeat_elements = self.batch_size - initial_frames.shape[0]
            batch = tree_map(
                lambda x: torch.cat(
                    [x] + [x[-1].unsqueeze(0)
                           for _ in range(num_repeat_elements)]
                ),
                batch,
            )
        spatial_grid, t, initial_frames, _, _, pde_params = batch

        forward_pass_start_time = time.time()
        predicted_pde_params, residual, true_residual = self.model(
            initial_frames, pde_params, spatial_grid, t, gumbel=gumbel
        )
        forward_pass_end_time = time.time()
        forward_pass_time = forward_pass_end_time - forward_pass_start_time
        if self.residual_filter:
            ids = [
                true_residual.reshape(
                    true_residual.shape[0], -1).max(dim=1)[0] < 100
            ]
            residual = residual[ids]
            true_residual = true_residual[ids]

            for key in pde_params:
                predicted_pde_params[key] = predicted_pde_params[key][ids]
                pde_params[key] = pde_params[key][ids]

        residual_loss = pde_residual_reduction(
            residual
        )  # Convert residual tensor to float loss
        true_residual_loss = pde_residual_reduction(true_residual)
        param_losses = self.param_loss_metric(
            predicted_pde_params,
            pde_params,
            reduction="mean",
        )

        relative_param_losses = param_relative_loss(
            {param: prediction for param, prediction in predicted_pde_params.items()},
            pde_params,
            reduction="mean",
        )

        # To avoid scenario where the averaged loss is computed with zero errors corresponding to params
        # that are not predicted for darcy flow.
        backprop_losses: list[torch.Tensor]
        if self.pde == PDE.DarcyFlow2D and stage != "tailoring":
            backprop_losses = [
                v for v in param_losses.values() if v.item() != 0]
        else:
            backprop_losses = list(param_losses.values())

        param_loss = torch.stack(backprop_losses).mean()

        individual_param_losses = {
            f"param_loss_{k}": v for k, v in param_losses.items()
        }
        individual_relative_param_losses = {
            f"relative_param_loss_{k}": v for k, v in relative_param_losses.items()
        }

        weighted_param_loss = self.inverse_param_loss_weight * param_loss

        weighted_residual_loss = self.inverse_residual_loss_weight * residual_loss

        loss = weighted_param_loss + weighted_residual_loss
        losses = {
            "param_loss": param_loss,
            "residual_loss": residual_loss,
            "true_residual_loss": true_residual_loss,
            "loss": loss,
            **individual_param_losses,
            **individual_relative_param_losses,
        }

        if self.pde == PDE.DarcyFlow2D:
            # Additional metrics for darcy flow to be consistent with PINO
            darcy_losses: dict[str, float] = classification_metrics_darcy_flow(
                predicted_coeff=predicted_pde_params["coeff"],
                true_coeff=pde_params["coeff"],
            )
            losses.update(darcy_losses)

        # In the case of validation, we want to handle some additional metrics
        if "validation" in stage or "test" in stage or "tailoring" in stage:
            # we want to stratify the losses based on PDE parameter
            # Stratification happens in the plotting callback - this performs bookkeeping.
            param_loss_per_batch_element, residual_per_batch_element = (
                self.stratify_losses(
                    predicted_pde_params,
                    pde_params,
                    residual,
                )
            )
            losses["param_loss_per_batch_element"] = param_loss_per_batch_element
            losses["residual_per_batch_element"] = residual_per_batch_element

        # reset batch to original size by removing the repeated last element in orginal batch
        if num_repeat_elements > 0:
            predicted_pde_params = tree_map(
                lambda x: x[: -1 * num_repeat_elements], predicted_pde_params
            )
            pde_params = tree_map(
                lambda x: x[: -1 * num_repeat_elements], pde_params)
            losses = tree_map(
                lambda x: x[: -1 *
                            num_repeat_elements] if x.numel() > 1 else x, losses
            )
        end_time = time.time()
        losses["timing_metrics"] = {
            "predict_with_loss_time": end_time - start_time,
            "forward_pass_time": forward_pass_time,
        }
        return predicted_pde_params, pde_params, losses

    @jaxtyped(typechecker=typeguard.typechecked)
    def training_step(
        self,
        batch: Union[TypeBatch1D, TypeBatch2D],
        batch_idx: int,
    ) -> TypeLossDict:
        stage = "train"
        _, _, losses = self.predict_with_loss(batch, stage="train")
        self.log_metrics(stage, losses)
        return losses

    @jaxtyped(typechecker=typeguard.typechecked)
    def validation_step(
        self,
        batch: Union[TypeBatch1D, TypeBatch2D],
        batch_idx: int,
        dataloader_idx: int = 0,
        stage: str = None,
    ) -> Union[None, TypeLossDict]:
        stage = "validation" if stage == None else stage
        prediction, target, losses = self.predict_with_loss(batch, stage=stage)

        #### Bookkeeping for plotting. See logging_callbacks.py for plotting. #####

        # NOTE: If we dont convert to numpy, there will be an OOM exception
        if self.pde == PDE.DarcyFlow2D:
            for k in prediction.keys():
                prediction[k] = prediction[k].cpu().numpy()
                target[k] = target[k].cpu().numpy()
                self.validation_step_outputs.append(prediction)
                self.validation_step_targets.append(target)
                self.validation_step_loss_dicts.append(
                    (
                        ptu.torch_dict_to_numpy(losses),
                        ptu.torch_dict_to_numpy(batch[-1]),
                    )
                )
        else:
            self.validation_step_outputs.append(prediction)
            self.validation_step_targets.append(target)
            self.validation_step_loss_dicts.append((losses, batch[-1]))
        losses["predictions"] = prediction
        losses["targets"] = target
        self.log_metrics(stage, losses)
        return losses

    @jaxtyped(typechecker=typeguard.typechecked)
    def test_step(
        self,
        batch: Union[TypeBatch1D, TypeBatch2D],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> TypeLossDict:
        stage = "test"
        self.model.train()
        test_losses = self.validation_step(
            batch, batch_idx, dataloader_idx, stage)
        return test_losses

    def log_metrics(self, stage, losses, prefix_dir=""):
        if "validation" in stage or "test" in stage:
            on_step = False
            on_epoch = True
        else:
            on_step = True
            on_epoch = False

        main_loss_type = "param_loss"

        self.log(
            os.path.join(stage, prefix_dir, main_loss_type),
            losses["param_loss"],
            prog_bar=True,
            on_step=on_step,
            on_epoch=on_epoch,
        )

        for param in self.params_to_predict:
            self.log(
                os.path.join(stage, prefix_dir, f"{main_loss_type}_{param}"),
                losses[f"param_loss_{param}"],
                prog_bar=True,
                on_step=on_step,
                on_epoch=on_epoch,
            )

            self.log(
                os.path.join(stage, prefix_dir,
                             f"relative_param_loss_{param}"),
                losses[f"relative_param_loss_{param}"],
                prog_bar=True,
                on_step=on_step,
                on_epoch=on_epoch,
            )

        self.log(
            os.path.join(stage, prefix_dir, "residual_loss"),
            losses["residual_loss"],
            prog_bar=True,
            on_step=on_step,
            on_epoch=on_epoch,
        )

        self.log(
            os.path.join(stage, prefix_dir, "true_residual_loss"),
            losses["true_residual_loss"],
            prog_bar=True,
            on_step=on_step,
            on_epoch=on_epoch,
        )

        if isinstance(self.model, InverseModel):

            # Check for darcy inverse model
            if self.pde == PDE.DarcyFlow2D:
                self.log(
                    os.path.join(
                        stage, prefix_dir, "darcy_flow_classification_accuracy"
                    ),
                    losses["darcy_flow_classification_accuracy"],
                    prog_bar=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    os.path.join(stage, prefix_dir,
                                 "darcy_flow_classification_iou"),
                    losses["darcy_flow_classification_iou"],
                    prog_bar=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )

        self.log(
            os.path.join(stage, prefix_dir, "loss"),
            losses["loss"],
            on_step=on_step,
            on_epoch=on_epoch,
        )

        if "timing_metrics" in losses:
            self.log(
                os.path.join(stage, prefix_dir, "predict_with_loss_time"),
                losses["timing_metrics"]["predict_with_loss_time"],
                on_step=on_step,
                on_epoch=on_epoch,
            )

            self.log(
                os.path.join(stage, prefix_dir, "forward_pass_time"),
                losses["timing_metrics"]["forward_pass_time"],
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def configure_optimizers(self):
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        return [optimizer], [lr_scheduler]

    def on_before_optimizer_step(self, optimizer):
        # logging the sum of the l2 norms of both the forward and inverse model

        self.old_inverse_model = torch.cat(
            [p.flatten() for p in self.model.parameters()]
        )
        inverse_norms = grad_norm(self.model, 2)
        inverse_total_norm = sum(inverse_norms.values())
        self.log(
            f"outer_loss_grad_norms/inverse_model",
            inverse_total_norm,
            on_step=True,
        )

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        self.new_inverse_model = torch.cat(
            [p.flatten() for p in self.model.parameters()]
        )
        param_diff_norm = torch.norm(
            self.new_inverse_model - self.old_inverse_model)
        original_param_norm = torch.norm(self.old_inverse_model)
        rel_diff = param_diff_norm / original_param_norm
        self.log(
            "outer_loss_grad_norms/callback_inverse_param_rel_diff",
            rel_diff,
            on_step=True,
        )

    def stratify_losses(self, predicted_pde_params, pde_params, residual):
        # Compute the per trajectory param loss & residual
        param_loss_per_batch_element = self.param_loss_metric(
            predicted_pde_params, pde_params, reduction="none"
        )

        # Edge case where we instead sum the error for darcy flow
        if self.pde == PDE.DarcyFlow2D:
            coeff = param_loss_per_batch_element["coeff"]
            # In the case we use a relative error, we don't need to do this
            if len(coeff.shape) == 4:
                coeff = torch.flatten(coeff, start_dim=2).sum(axis=2)

            param_loss_per_batch_element["coeff"] = coeff

        param_loss_per_batch_element = (
            torch.stack(list(param_loss_per_batch_element.values()))
            .squeeze(-1)
            .mean(dim=0)
        )
        residual_per_batch_element = pde_residual_reduction(
            residual, dim=tuple(range(1, residual.dim()))
        )

        return param_loss_per_batch_element, residual_per_batch_element

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
