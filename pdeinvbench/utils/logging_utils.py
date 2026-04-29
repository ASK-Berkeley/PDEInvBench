import numpy as np
import pdeinvbench.utils.pytorch_utils as ptu
from pdeinvbench.utils.pytorch_utils import is_numpy
from jaxtyping import jaxtyped
import typeguard
import wandb
from pdeinvbench.utils.types import (
    TypeScaleInputField2D,
    TypeScaleInputField1D,
    TypeScaledField1D,
    TypeLoggingField1D,
    TypeLoggingField2D,
)
from typing import Dict, Tuple
import torch

from lightning.pytorch.loggers import WandbLogger

import time
import warnings


class CustomWandbLogger(WandbLogger):
    def after_save_checkpoint(self, checkpoint_callback):

        # Logic to delete intermediate checkpoints
        # You can use the W&B API to delete specific checkpoints if needed
        # Example: delete the checkpoints that are not the best
        super().after_save_checkpoint(checkpoint_callback)
        if self._log_model:
            api = wandb.Api()
            entity = self._wandb_init["entity"]
            project = self._project
            model_ckpt = self._checkpoint_name
            try:
                model_artifacts = api.artifacts(
                    "model", f"{entity}/{project}/{model_ckpt}"
                )
                for model_ckpt in model_artifacts:
                    if (
                        model_ckpt.state != "DELETED"
                        and "latest" not in model_ckpt.aliases
                        and "best" not in model_ckpt.aliases
                    ):
                        model_ckpt.delete()
            except Exception as e:
                # adhoc way to do deal with latency in wandb
                warnings.warn(f"W&B error {e}")


@jaxtyped(typechecker=typeguard.typechecked)
def scale_2d_field_for_wandb(
    predictions: TypeScaleInputField2D,
    targets: TypeScaleInputField2D,
) -> TypeLoggingField2D:
    """
    Scales the predicted trajectory and the ground truth trajectory between [0,255] for wandb video logging for 2d systems, per channel.
    Computes the absolute difference between the predicted and ground truth trajectories and scales the difference
    between [0,255] for wandb video logging.
    For the predictions and target fields:
        The 0 value corresponds to the lowest solution field value across both ground truth and predicted trajectories.
        The 255 value corresponds to the largest solution field value across both ground truth and predicted trajectories.
    The absolute difference field:
        The 0 value corresponds to the lowest absolute difference value across both ground truth and predicted trajectories.
        The 255 value corresponds to the largest absolute difference value across both ground truth and predicted trajectories.
    The scaled fields repeated 3 times as RGB channel dimensions for logging.
    :param predictions: numpy.Array - predicted solution (T x X_spatial x y_spatial)
    :param target: numpy.Array - target solution (T x X_spatial x y_spatial)
    :return: numpy.Array - scaled target field (T x 3 x X_spatial x y_spatial), scaled predicted fields (T x 3 x X_spatial x y_spatial), scaled absolute difference field (T x 3 x X_spatial x y_spatial)
    """
    if is_numpy(predictions):
        predictions = np.expand_dims(predictions, axis=1)
    else:
        predictions = ptu.torch_to_numpy(predictions.unsqueeze(1))
    if is_numpy(targets):
        targets = np.expand_dims(targets, axis=1)
    else:
        targets = ptu.torch_to_numpy(targets.unsqueeze(1))

    pred_min, pred_max = np.min(predictions), np.max(predictions)
    target_min, target_max = np.min(targets), np.max(targets)
    scale_min, scale_max = min(pred_min, target_min), max(pred_max, target_max)
    scaled_target = 255 * (
        (
            (
                np.repeat(
                    targets,
                    repeats=3,
                    axis=1,
                )
            )
            - scale_min
        )
        / (scale_max - scale_min)
    )
    scaled_pred = 255 * (
        (
            (
                np.repeat(
                    predictions,
                    repeats=3,
                    axis=1,
                )
            )
            - scale_min
        )
        / (scale_max - scale_min)
    )

    diff_min, diff_max = np.min(np.abs(predictions - targets)), np.max(
        np.abs(predictions - targets)
    )
    scaled_diff = 255 * (
        (
            np.repeat(
                np.abs(predictions - targets),
                repeats=3,
                axis=1,
            )
            - diff_min
        )
        / (diff_max - diff_min)
    )
    return (
        scaled_target.astype(np.uint8),
        scaled_pred.astype(np.uint8),
        scaled_diff.astype(np.uint8),
    )


@jaxtyped(typechecker=typeguard.typechecked)
def scale_1d_field(field: TypeScaleInputField1D) -> TypeScaledField1D:
    """
    Scales 1D trajectory between 0 and 255 for wandb image logging.
    The 0 value corresponds to the lowest solution field value across both ground truth and predicted trajectories.
    The 255 value corresponds to the largest solution field value across both ground truth and predicted trajectories.
    :param field: numpy.Array - predicted solution (T x X_spatial)
    :return: numpy.Array - scaled field

    """
    scale_min = np.min(field)
    scale_max = np.max(field)
    scaled_field = 255 * ((field - scale_min) / scale_max - scale_min)
    return scaled_field.astype(np.uint8)


@jaxtyped(typechecker=typeguard.typechecked)
def scale_1d_field_for_wandb(
    predictions: TypeScaleInputField1D,
    target: TypeScaleInputField1D,
) -> TypeLoggingField1D:
    """
    Scales the predicted trajectory and the ground truth trajectory between [0,255] for wandb video logging for 1d systems.
    Computes the absolute difference between the predicted and ground truth trajectories and scales the difference
    between [0,255] for wandb video logging.
    For the predictions and target fields:
        The 0 value corresponds to the lowest solution field value across both ground truth and predicted trajectories.
        The 255 value corresponds to the largest solution field value across both ground truth and predicted trajectories.
    The absolute difference field:
        The 0 value corresponds to the lowest absolute difference value across both ground truth and predicted trajectories.
        The 255 value corresponds to the largest absolute difference value across both ground truth and predicted trajectories.
    :param predictions: numpy.Array - predicted solution (T x X_spatial)
    :param target: numpy.Array - target solution (T x X_spatial)
    :return: numpy.Array - scaled target field, scaled predicted fields, scaled absolute difference field
    """
    scale_min = min(np.min(target), np.min(predictions))
    scale_max = max(np.max(target), np.max(predictions))
    scaled_target = 255 * ((target - scale_min) / scale_max - scale_min)
    scaled_predictions = 255 * ((predictions - scale_min) / scale_max - scale_min)

    difference = np.absolute(target - predictions)
    difference_min, difference_max = np.min(difference), np.max(difference)
    scaled_diff = 255 * (
        (difference - difference_min) / (difference_max - difference_min)
    )
    return (
        scaled_target.astype(np.uint8),
        scaled_predictions.astype(np.uint8),
        scaled_diff.astype(np.uint8),
    )


def get_best_model_weights(
    entity: str,
    project: str,
    metric: str = "validation/param_loss",
    filters: Dict = None,
) -> Dict[str, torch.Tensor]:
    """Get the weights from the best performing model."""
    api = wandb.Api()

    # Build filters query
    filter_str = " ".join([f"{k}={v}" for k, v in (filters or {}).items()])

    # Get all runs
    runs = api.runs(f"{entity}/{project}", filters=filter_str)

    best_value = float("inf")
    best_run = None

    for run in runs:
        if run.state != "finished":
            continue

        try:
            current_value = run.summary.get(metric)
            if current_value is None:
                continue

            if current_value < best_value:
                best_value = current_value
                best_run = run

        except Exception as e:
            print(f"Error processing run {run.id}: {e}")
            continue

    if best_run is None:
        raise ValueError("No valid runs found!")

    # Download the checkpoint
    checkpoint_file = None
    for file in best_run.files():
        if file.name.endswith(".ckpt"):
            checkpoint_file = file
            break

    if checkpoint_file is None:
        raise ValueError(f"No checkpoint found in best run {best_run.id}")

    # Download and load checkpoint
    checkpoint_file.download(replace=True)
    checkpoint = torch.load(checkpoint_file.name)

    return checkpoint["state_dict"]


def collect_loss_dicts(outputs, batch, metric_name, metrics_array):
    outputs = ptu.torch_dict_to_numpy(outputs)
    pde_params_np = ptu.torch_dict_to_numpy(batch[-1])
    ic_index = batch[-2]
    if type(ic_index) == torch.Tensor:
        # to cpu and then numpy
        ic_index = ic_index.cpu().numpy()
    timestamps = batch[-3]
    if type(timestamps) == torch.Tensor:
        timestamps = timestamps.cpu().numpy()
    required_batch_size = outputs[metric_name].shape[0]
    param_key = list(pde_params_np.keys())[0]
    if required_batch_size > pde_params_np[param_key].shape[0]:
        num_repeat_elements = required_batch_size - pde_params_np[param_key].shape[0]
        batch = tree_map(
            lambda x: torch.cat(
                [x] + [x[-1].unsqueeze(0) for _ in range(num_repeat_elements)]
            ),
            batch,
        )
        pde_params_np = ptu.torch_dict_to_numpy(batch[-1])

    metrics_array.append((outputs, pde_params_np, ic_index, timestamps))
