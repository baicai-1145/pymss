from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class LossCurveRecorder(Callback):
    fieldnames = ("event", "global_step", "epoch", "batch_idx", "loss", "val_loss", "val_sdr")

    def __init__(self, path: str | Path, interval: int = 1) -> None:
        self.path = Path(path)
        self.interval = int(interval)
        if self.interval < 0:
            raise ValueError("loss curve recording interval must be >= 0")

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if not self.interval or (batch_idx + 1) % self.interval != 0:
            return
        loss = _to_float(trainer.callback_metrics.get("train/loss"))
        if loss is None:
            loss = _extract_loss(outputs)
        if loss is None:
            return
        self._append(
            {
                "event": "train",
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "batch_idx": int(batch_idx),
                "loss": loss,
                "val_loss": "",
                "val_sdr": "",
            }
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        val_loss = _to_float(trainer.callback_metrics.get("val/loss"))
        val_sdr = _to_float(trainer.callback_metrics.get("val_sdr"))
        if val_loss is None and val_sdr is None:
            return
        self._append(
            {
                "event": "validation",
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "batch_idx": "",
                "loss": "",
                "val_loss": "" if val_loss is None else val_loss,
                "val_sdr": "" if val_sdr is None else val_sdr,
            }
        )

    def _append(self, row: dict[str, object]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)


def _extract_loss(outputs: Any) -> float | None:
    if isinstance(outputs, torch.Tensor):
        return _to_float(outputs)
    if isinstance(outputs, dict):
        return _to_float(outputs.get("loss"))
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ExponentialMovingAverage(Callback):
    """Maintain EMA model weights and use them for validation, matching the original trainer."""

    def __init__(self, decay: float) -> None:
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.ema_model: AveragedModel | None = None
        self._stored_state: dict[str, torch.Tensor] | None = None
        self._pending_ema_state: dict[str, torch.Tensor] | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.ema_model = AveragedModel(pl_module.model, multi_avg_fn=get_ema_multi_avg_fn(self.decay))
        if self._pending_ema_state is not None:
            self.ema_model.module.load_state_dict(self._pending_ema_state)
            self._pending_ema_state = None
        pl_module.ema_model = self.ema_model
        pl_module.ema_decay = self.decay

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.ema_model is not None:
            self.ema_model.update_parameters(pl_module.model)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_model is None:
            return
        self._stored_state = {key: value.detach().clone() for key, value in pl_module.model.state_dict().items()}
        pl_module.model.load_state_dict(self.ema_model.module.state_dict())
        pl_module.using_ema_weights = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._restore_training_weights(pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._restore_training_weights(pl_module)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "ema_state_dict": None if self.ema_model is None else self.ema_model.module.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ema_state = state_dict.get("ema_state_dict")
        if ema_state is None:
            return
        if self.ema_model is None:
            self._pending_ema_state = ema_state
        else:
            self.ema_model.module.load_state_dict(ema_state)

    def _restore_training_weights(self, pl_module: pl.LightningModule) -> None:
        if self._stored_state is None:
            return
        pl_module.model.load_state_dict(self._stored_state)
        pl_module.using_ema_weights = False
        self._stored_state = None
