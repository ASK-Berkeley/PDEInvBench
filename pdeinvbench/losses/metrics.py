from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import typeguard
from jaxtyping import Float, jaxtyped

from pdeinvbench.utils.types import DataMetrics, ParamMetrics


@jaxtyped(typechecker=typeguard.typechecked)
def get_data_metric(metric: DataMetrics) -> Callable:
    if metric == DataMetrics.MSE:
        return torch.nn.functional.mse_loss
    elif metric == DataMetrics.Relative_Error:
        return relative_error


@jaxtyped(typechecker=typeguard.typechecked)
def get_param_metric(metric: ParamMetrics) -> Callable:
    if metric == ParamMetrics.MSE:
        return param_mse_loss
    elif metric == ParamMetrics.Relative_Error:
        return param_relative_loss

    else:
        raise ValueError("Parameter metric not recognized.")


@jaxtyped(typechecker=typeguard.typechecked)
def relative_error(
    input: Float[torch.Tensor, "batch ..."],
    target: Float[torch.Tensor, "batch ..."],
    reduction: str = "mean",
    setting: str = "per_trajectory",
) -> torch.Tensor:
    """
    Compute the relative loss between two solution fields.
    :param input: torch.Tensor - predicted solution (B x ...)
    :param target: torch.Tensor - target solution (B x ...)
    :param reduction: str - reduction method for the loss. None returns a tensor of batch size
    :return: torch.Tensor - relative data loss
    """
    difference = input - target
    batch_size = difference.shape[0]
    if setting == "per_trajectory":
        difference = difference.view(batch_size, -1)
        loss = torch.linalg.norm(difference, dim=1) / torch.linalg.norm(
            target.view(batch_size, -1), dim=1
        )
    elif setting == "per_frame":
        time_size = difference.shape[1]
        difference = difference.view(batch_size, time_size, -1)
        loss = torch.linalg.norm(difference, dim=2) / torch.linalg.norm(
            target.view(batch_size, time_size, -1), dim=2
        )
    else:
        raise ValueError(
            "relative data error setting not recognized. Select between 'per_trajectory' and 'per_frame'"
        )
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


@jaxtyped(typechecker=typeguard.typechecked)
def param_mse_loss(
    predicted_params: Dict[
        str, Float[torch.Tensor, "batch 1"] | Float[torch.Tensor, "batch 1 nx ny"]
    ],
    true_params: Dict[
        str, Float[torch.Tensor, "batch 1"] | Float[torch.Tensor, "batch 1 nx ny"]
    ],
    reduction: str,
) -> Dict[str, torch.Tensor]:
    """
    Compute the MSE loss between predicted and true parameters.
    :param predicted_params: torch.Tensor - dictionary of predicted parameters.
    :param true_params: torch.Tensor - dictionary of true parameters.
    :return: dictionary of parameter MSE losses
    """
    # assert that dictionaries both have the same keys.
    assert (
        predicted_params.keys() == true_params.keys()
    ), "Keys of predicted and true parameters do not match."

    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "Reduction must be either 'mean' 'sum'. or 'none'"

    loss: dict[str, torch.Tensor] = dict()

    for param in predicted_params.keys():
        loss[param] = torch.nn.functional.mse_loss(
            predicted_params[param], true_params[param], reduction=reduction
        )
    return loss


@jaxtyped(typechecker=typeguard.typechecked)
def param_relative_loss(
    predicted_params: Dict[
        str, Float[torch.Tensor, "batch 1"] | Float[torch.Tensor, "batch 1 nx ny"]
    ],
    true_params: Dict[
        str, Float[torch.Tensor, "batch 1"] | Float[torch.Tensor, "batch 1 nx ny"]
    ],
    reduction: str,
) -> Dict[str, torch.Tensor]:
    """
    Compute the relative loss between predicted and true parameters.
    :param predicted_params: torch.Tensor - dictionary of predicted parameters.
    :param true_params: torch.Tensor - dictionary of true parameters.
    :return: dictionary of parameter relative losses
    """
    # assert that dictionaries both have the same keys.
    assert (
        predicted_params.keys() == true_params.keys()
    ), "Keys of predicted and true parameters do not match."
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "Reduction must be either 'mean', 'sum' or 'none'."

    loss: dict[str, torch.Tensor] = dict()

    for param in predicted_params.keys():
        # In the case that the parameter is a vector field, we have to use the relative l2 loss from the forward problem
        # Only relevant in the case of darcy flow
        if isinstance(true_params[param], Float[torch.Tensor, "batch 1 nx ny"]):
            predicted = predicted_params[param]
            true = true_params[param]
            error = relative_error(predicted, true, reduction="none")
            loss[param] = error
        else:
            difference = predicted_params[param] - true_params[param]

            loss[param] = torch.linalg.norm(difference, dim=1) / torch.linalg.norm(
                true_params[param], dim=1
            )

        if reduction == "mean":
            loss[param] = loss[param].mean()
        elif reduction == "sum":
            loss[param] = loss[param].sum()

    return loss


@jaxtyped(typechecker=typeguard.typechecked)
def pde_residual_reduction(
    pde_residual: Float[torch.Tensor, "batch ..."],
    reduction: str = "mean",
    dim: Any = None,
) -> torch.Tensor:
    """
    Given a tensor of PDE residuals (B x ...), compute the reduction of the residuals into a single metric.
    :param pde_residual: torch.Tensor containing the PDE residual field.
    :param reduction: How to reduce different batch elements. Must be one of "sum" or "mean". Default: "mean"
    :param dim: dimension(s) along which to reduce. Default: None
    """
    target = torch.zeros_like(pde_residual)
    sq_diff = (pde_residual - target).square()
    if reduction == "mean":
        return sq_diff.mean(dim)
    elif reduction == "sum":
        return sq_diff.sum(dim)
    else:
        AssertionError("Reduction method not recognized.")


@jaxtyped(typechecker=typeguard.typechecked)
def classification_metrics_darcy_flow(
    predicted_coeff: Float[torch.Tensor, "batch 1 nx ny"],
    true_coeff: Float[torch.Tensor, "batch 1 nx ny"],
    reduction: str = "mean",
) -> dict[str, Float[torch.Tensor, ""] | Float[torch.Tensor, "batch"]]:
    """
    Stand in for classification metrics to compute on darcy flow.
    Reduction represents the batch-wise reduction.
    Returns a dict with two keys: "accuracy" and "IoU".
    """
    metrics: dict[str, float] = {}

    # Predicted coeff is a field that goes from 0 -> 1, so we presume anything > 0.5 is binned to true
    binarized_predicted = predicted_coeff > 0.5
    binarized_true = true_coeff.bool()
    pointwise_correctness = binarized_predicted == binarized_true
    flat_pointwise = pointwise_correctness.flatten(start_dim=1)
    num_points = flat_pointwise.shape[1]
    per_batch_elem_accuracy = flat_pointwise.sum(dim=1).float() / num_points
    metrics["darcy_flow_classification_accuracy"] = per_batch_elem_accuracy

    intersection = binarized_predicted & binarized_true
    intersection = intersection.flatten(start_dim=1)
    union = binarized_predicted | binarized_true
    union = union.flatten(start_dim=1)
    IoU = intersection.sum(dim=1) / union.sum(dim=1)
    metrics["darcy_flow_classification_iou"] = IoU

    for k in metrics.keys():
        if reduction == "mean":
            metrics[k] = metrics[k].mean()
        elif reduction == "sum":
            metrics[k] = metrics[k].sum()

    return metrics
