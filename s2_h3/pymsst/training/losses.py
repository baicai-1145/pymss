from __future__ import annotations

import torch
import torch.nn.functional as F


def align_prediction_and_target(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim == 3 and target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    elif prediction.ndim == 4 and target.ndim == 3 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]

    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}")
    return prediction, target


def masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    q: float = 0.95,
    coarse: bool = True,
    original_shape_compat: bool = False,
) -> torch.Tensor:
    if original_shape_compat:
        prediction, target = torch.broadcast_tensors(prediction, target)
    else:
        prediction, target = align_prediction_and_target(prediction, target)
    if prediction.ndim == 3:
        prediction = prediction[:, None]
        target = target[:, None]
    loss = (prediction - target).pow(2) if original_shape_compat else F.mse_loss(prediction, target, reduction="none")
    if loss.ndim >= 3:
        loss = loss.transpose(0, 1)
    if coarse and loss.ndim >= 2:
        loss = loss.mean(dim=tuple(range(max(1, loss.ndim - 2), loss.ndim)))
    loss = loss.reshape(loss.shape[0], -1)
    quantile = torch.quantile(loss.detach(), q, interpolation="linear", dim=1, keepdim=True)
    mask = loss < quantile
    return (loss * mask).mean()


def signal_to_distortion_ratio(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    prediction, target = align_prediction_and_target(prediction, target)
    dims = tuple(range(1, target.ndim))
    signal = target.pow(2).sum(dim=dims)
    noise = (target - prediction).pow(2).sum(dim=dims)
    return 10.0 * torch.log10((signal + eps) / (noise + eps))


def compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    q: float = 0.95,
    coarse: bool = True,
    original_shape_compat: bool = False,
) -> torch.Tensor:
    if loss_name == "masked_loss":
        return masked_loss(prediction, target, q=q, coarse=coarse, original_shape_compat=original_shape_compat)
    prediction, target = align_prediction_and_target(prediction, target)
    if loss_name == "mse_loss":
        return F.mse_loss(prediction, target)
    if loss_name == "l1_loss":
        return F.l1_loss(prediction, target)
    raise ValueError(f"unsupported loss: {loss_name}")
