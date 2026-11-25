from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from pymss_core import to_plain

from pymsst import __version__
from pymsst.data.datamodule import MusdbDataModule
from pymsst.training.cuda_graph import RoformerMaskCoreCudaGraph
from pymsst.training import SeparationLightningModule, compute_loss


LOSS_CURVE_FIELDS = ("event", "global_step", "epoch", "batch_idx", "loss", "val_loss", "val_sdr")


def run_direct_loss_curve(
    *,
    implementation: str,
    model_type: str,
    config_path: str | Path,
    data_root: str | Path,
    results_path: str | Path,
    steps: int,
    seed: int = 0,
    start_checkpoint: str | Path | None = None,
    original_root: str | Path = "/root/Music-Source-Separation-Training",
    batch_size: int | None = None,
    chunk_size: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    loss_record_interval: int = 1,
    optimizer_backend: str | None = None,
    cuda_attention_backend: str | None = None,
    fused_kernel_backend: str | None = None,
    fused_rms_norm_mode: str | None = None,
    cuda_graph_mode: str | None = None,
    runtime_optimization_preset: str | None = None,
) -> dict[str, object]:
    if implementation == "pymsst":
        return _run_pymsst_curve(
            model_type=model_type,
            config_path=Path(config_path),
            data_root=Path(data_root),
            results_path=Path(results_path),
            steps=steps,
            seed=seed,
            start_checkpoint=Path(start_checkpoint) if start_checkpoint else None,
            batch_size=batch_size,
            chunk_size=chunk_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            loss_record_interval=loss_record_interval,
            optimizer_backend=optimizer_backend,
            cuda_attention_backend=cuda_attention_backend,
            fused_kernel_backend=fused_kernel_backend,
            fused_rms_norm_mode=fused_rms_norm_mode,
            cuda_graph_mode=cuda_graph_mode,
            runtime_optimization_preset=runtime_optimization_preset,
        )
    if implementation == "original":
        if optimizer_backend not in {None, "default"}:
            raise ValueError("--optimizer-backend is only supported for --implementation pymsst")
        if cuda_attention_backend is not None:
            raise ValueError("--cuda-attention-backend is only supported for --implementation pymsst")
        if fused_kernel_backend not in {None, "off"}:
            raise ValueError("--fused-kernel-backend is only supported for --implementation pymsst")
        if fused_rms_norm_mode not in {None, "off"}:
            raise ValueError("--fused-rms-norm-mode is only supported for --implementation pymsst")
        if cuda_graph_mode not in {None, "off"}:
            raise ValueError("--cuda-graph-mode is only supported for --implementation pymsst")
        if runtime_optimization_preset not in {None, "off"}:
            raise ValueError("--runtime-optimization-preset is only supported for --implementation pymsst")
        if num_workers or pin_memory or persistent_workers:
            raise ValueError("--num-workers/--pin-memory/--persistent-workers are only supported for --implementation pymsst")
        if loss_record_interval != 1:
            raise ValueError("--loss-record-interval is only supported for --implementation pymsst")
        return _run_original_curve(
            model_type=model_type,
            config_path=Path(config_path),
            data_root=Path(data_root),
            results_path=Path(results_path),
            steps=steps,
            seed=seed,
            start_checkpoint=Path(start_checkpoint) if start_checkpoint else None,
            original_root=Path(original_root),
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
    raise ValueError("implementation must be 'pymsst' or 'original'")


def _run_pymsst_curve(
    *,
    model_type: str,
    config_path: Path,
    data_root: Path,
    results_path: Path,
    steps: int,
    seed: int,
    start_checkpoint: Path | None,
    batch_size: int | None,
    chunk_size: int | None,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    loss_record_interval: int,
    optimizer_backend: str | None,
    cuda_attention_backend: str | None,
    fused_kernel_backend: str | None,
    fused_rms_norm_mode: str | None,
    cuda_graph_mode: str | None,
    runtime_optimization_preset: str | None,
) -> dict[str, object]:
    _seed_everything(seed)
    results_path.mkdir(parents=True, exist_ok=True)
    module = SeparationLightningModule(
        model_type=model_type,
        config_path=config_path,
        original_loss_compat=True,
        start_checkpoint=start_checkpoint,
        strict_checkpoint=True,
        optimizer_backend=optimizer_backend,
        cuda_attention_backend=cuda_attention_backend,
        fused_kernel_backend=fused_kernel_backend,
        fused_rms_norm_mode=fused_rms_norm_mode,
        cuda_graph_mode=cuda_graph_mode,
        runtime_optimization_preset=runtime_optimization_preset,
    )
    _consume_original_roformer_python_rng(model_type)
    config = module.config
    _apply_overrides(config, steps=steps, batch_size=batch_size, chunk_size=chunk_size)
    config.training.num_workers = int(num_workers)
    config.training.pin_memory = bool(pin_memory)
    config.training.persistent_workers = bool(persistent_workers)
    config.training.loss_record_interval = int(loss_record_interval)
    config.training.runtime_optimization_preset = module.runtime_optimization_preset
    config.training.fused_kernel_backend = module.fused_kernel_backend
    config.training.fused_rms_norm_mode = module.fused_rms_norm_mode or "off"
    config.training.fused_kernel_active = module.fused_training_status.active
    config.training.fused_kernel_patched_band_splits = module.fused_training_status.patched_band_splits
    config.training.fused_kernel_patched_band_split_bwd_v2 = module.fused_training_status.patched_band_split_bwd_v2
    config.training.fused_kernel_patched_band_split_bwd_v3 = module.fused_training_status.patched_band_split_bwd_v3
    config.training.fused_kernel_patched_band_split_ragged_forward = (
        module.fused_training_status.patched_band_split_ragged_forward
    )
    config.training.fused_kernel_patched_band_split_ragged_backward = (
        module.fused_training_status.patched_band_split_ragged_backward
    )
    config.training.fused_kernel_patched_rotary = module.fused_training_status.patched_rotary
    config.training.fused_kernel_patched_rms_norm = module.fused_training_status.patched_rms_norm
    config.training.fused_kernel_rms_norm_mode = module.fused_training_status.rms_norm_mode
    config.training.fused_kernel_patched_rms_norm_bwd_v2 = module.fused_training_status.patched_rms_norm_bwd_v2
    config.training.fused_kernel_patched_single_stem_masks = module.fused_training_status.patched_single_stem_masks
    config.training.fused_kernel_patched_attention_gates = module.fused_training_status.patched_attention_gates
    config.training.fused_kernel_attention_gate_mode = module.fused_training_status.attention_gate_mode
    config.training.fused_kernel_patched_layout_reduce_copy = module.fused_training_status.patched_layout_reduce_copy
    config.training.fused_kernel_patched_train_param_pack_autograd = (
        module.fused_training_status.patched_train_param_pack_autograd
    )
    config.training.fused_kernel_patched_mask_final_autograd = (
        module.fused_training_status.patched_mask_final_autograd
    )
    config.training.fused_kernel_patched_mask_estimator_final = module.fused_training_status.patched_mask_estimator_final
    config.training.fused_kernel_mask_estimator_final_mode = module.fused_training_status.mask_estimator_final_mode
    if module.fused_training_status.reason:
        config.training.fused_kernel_reason = module.fused_training_status.reason
    normalized_cuda_graph_mode = module.cuda_graph_mode
    config.training.cuda_graph_mode = normalized_cuda_graph_mode
    batch_size = int(config.training.batch_size)

    data = MusdbDataModule(
        data_root,
        instruments=tuple(config.training.instruments),
        target_instrument=config.training.get("target_instrument"),
        chunk_size=int(config.audio.chunk_size),
        sample_rate=int(config.audio.sample_rate),
        channels=int(config.audio.num_channels),
        batch_size=batch_size,
        steps_per_epoch=steps,
        val_tracks=0,
        seed=seed,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        min_mean_abs=float(config.audio.get("min_mean_abs", 0.0)),
        mix_from_stems=True,
        loudness_augmentation=bool(config.get("augmentations", {}).get("enable", False))
        and bool(config.get("augmentations", {}).get("loudness", False)),
        loudness_min=float(config.get("augmentations", {}).get("loudness_min", 0.5)),
        loudness_max=float(config.get("augmentations", {}).get("loudness_max", 1.5)),
        keep_target_stem_dim=True,
    )
    data.setup("fit")

    device = _device()
    module.to(device)
    module.train()
    optimizer = _extract_optimizer(module.configure_optimizers())
    cuda_graph_runner = (
        RoformerMaskCoreCudaGraph(module.model, normalized_cuda_graph_mode)
        if normalized_cuda_graph_mode != "off" and device.type == "cuda"
        else None
    )

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return compute_loss(
            module._prediction_for_loss(prediction, target),
            target,
            loss_name="masked_loss",
            q=float(config.training.get("q", 0.95)),
            coarse=bool(config.training.get("coarse_loss_clip", True)),
            original_shape_compat=True,
        )

    losses = _run_manual_steps(
        model=module,
        dataloader=data.train_dataloader(),
        optimizer=optimizer,
        config=config,
        device=device,
        steps=steps,
        loss_fn=loss_fn,
        loss_record_interval=loss_record_interval,
        cuda_graph_runner=cuda_graph_runner,
    )
    if cuda_graph_runner is not None:
        config.training.cuda_graph_active = cuda_graph_runner.status.active
        if cuda_graph_runner.status.reason:
            config.training.cuda_graph_reason = cuda_graph_runner.status.reason
    return _write_direct_result(
        implementation="pymsst",
        model_type=model_type,
        config_path=config_path,
        data_root=data_root,
        results_path=results_path,
        steps=steps,
        seed=seed,
        start_checkpoint=start_checkpoint,
        losses=losses,
        config=config,
    )


def _run_original_curve(
    *,
    model_type: str,
    config_path: Path,
    data_root: Path,
    results_path: Path,
    steps: int,
    seed: int,
    start_checkpoint: Path | None,
    original_root: Path,
    batch_size: int | None,
    chunk_size: int | None,
) -> dict[str, object]:
    _seed_everything(seed)
    results_path.mkdir(parents=True, exist_ok=True)
    if str(original_root) not in sys.path:
        sys.path.insert(0, str(original_root))

    from torch.utils.data import DataLoader
    from utils.dataset import MSSDataset
    from utils.losses import masked_loss as original_masked_loss
    from utils.model_utils import get_optimizer, load_start_checkpoint
    from utils.settings import get_model_from_config

    model, config = get_model_from_config(model_type, str(config_path))
    _apply_overrides(config, steps=steps, batch_size=batch_size, chunk_size=chunk_size)
    batch_size = int(config.training.batch_size)

    metadata_path = results_path / "metadata_1.pkl"
    if metadata_path.is_file():
        metadata_path.unlink()
    trainset = MSSDataset(
        config,
        str(data_root / "train"),
        batch_size=batch_size,
        metadata_path=str(metadata_path),
        dataset_type=1,
        verbose=False,
    )
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    if start_checkpoint:
        checkpoint = torch.load(start_checkpoint, weights_only=False, map_location="cpu")
        args = _OriginalArgs(model_type=model_type, start_check_point=str(start_checkpoint))
        load_start_checkpoint(args, model, checkpoint, type_="train")

    device = _device()
    model.to(device)
    model.train()
    optimizer = get_optimizer(config, model)

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return original_masked_loss(
            prediction,
            target,
            q=float(config.training.get("q", 0.95)),
            coarse=bool(config.training.get("coarse_loss_clip", True)),
        )

    losses = _run_manual_steps(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        config=config,
        device=device,
        steps=steps,
        loss_fn=loss_fn,
    )
    return _write_direct_result(
        implementation="original",
        model_type=model_type,
        config_path=config_path,
        data_root=data_root,
        results_path=results_path,
        steps=steps,
        seed=seed,
        start_checkpoint=start_checkpoint,
        losses=losses,
        config=config,
    )


def _run_manual_steps(
    *,
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    config,
    device: torch.device,
    steps: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_record_interval: int = 1,
    cuda_graph_runner: RoformerMaskCoreCudaGraph | None = None,
) -> list[tuple[int, float]]:
    loss_record_interval = int(loss_record_interval)
    if loss_record_interval < 0:
        raise ValueError("loss_record_interval must be >= 0")
    use_amp = bool(config.training.get("use_amp", False)) and device.type == "cuda"
    grad_accum = int(config.training.get("gradient_accumulation_steps", 1))
    grad_clip = float(config.training.get("grad_clip", 0) or 0)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    losses: list[tuple[int, float]] = []
    completed_steps = 0

    for batch_idx, (target, mixture) in enumerate(dataloader):
        if batch_idx >= steps:
            break
        step = batch_idx + 1
        target = target.to(device, non_blocking=True)
        mixture = mixture.to(device, non_blocking=True)

        if cuda_graph_runner is not None and cuda_graph_runner.status.active == "off" and cuda_graph_runner.status.reason is None:
            cuda_graph_runner.capture(mixture, use_amp=use_amp)
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = cuda_graph_runner.forward(mixture, use_amp=use_amp) if cuda_graph_runner is not None else model(mixture)
            loss = loss_fn(prediction, target)

        if _should_record_loss(step, steps, loss_record_interval):
            losses.append((step, float(loss.detach().cpu())))
        scaled_loss = loss / grad_accum
        scaler.scale(scaled_loss).backward()

        is_update_step = ((batch_idx + 1) % grad_accum == 0) or (batch_idx + 1 == steps)
        if is_update_step:
            scaler.unscale_(optimizer)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        completed_steps += 1

    if completed_steps != steps:
        raise RuntimeError(f"expected {steps} train steps, got {completed_steps}")
    return losses


def _extract_optimizer(configured):
    if isinstance(configured, torch.optim.Optimizer):
        return configured
    if isinstance(configured, dict) and isinstance(configured.get("optimizer"), torch.optim.Optimizer):
        return configured["optimizer"]
    if isinstance(configured, (list, tuple)) and configured:
        first = configured[0]
        if isinstance(first, torch.optim.Optimizer):
            return first
        if isinstance(first, dict) and isinstance(first.get("optimizer"), torch.optim.Optimizer):
            return first["optimizer"]
    raise TypeError(f"unsupported optimizer configuration: {type(configured).__name__}")


def _should_record_loss(step: int, total_steps: int, interval: int) -> bool:
    return interval > 0 and (step == 1 or step == total_steps or step % interval == 0)


def _write_direct_result(
    *,
    implementation: str,
    model_type: str,
    config_path: Path,
    data_root: Path,
    results_path: Path,
    steps: int,
    seed: int,
    start_checkpoint: Path | None,
    losses: list[tuple[int, float]],
    config,
) -> dict[str, object]:
    results_path.mkdir(parents=True, exist_ok=True)
    loss_curve_path = results_path / "loss_curve.csv"
    with loss_curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOSS_CURVE_FIELDS)
        writer.writeheader()
        for global_step, loss in losses:
            writer.writerow(
                {
                    "event": "train",
                    "global_step": global_step,
                    "epoch": 0,
                    "batch_idx": global_step - 1,
                    "loss": loss,
                    "val_loss": "",
                    "val_sdr": "",
                }
            )

    metadata = {
        "schema_version": 1,
        "package": "pymsst",
        "version": __version__,
        "implementation": implementation,
        "model_type": model_type,
        "config_path": str(config_path),
        "data_root": str(data_root),
        "results_path": str(results_path),
        "start_checkpoint": str(start_checkpoint) if start_checkpoint else None,
        "seed": seed,
        "steps": steps,
        "loss_curve_path": str(loss_curve_path),
        "config": _plain_config(config),
        "environment": _environment_metadata(),
    }
    (results_path / "run_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    return metadata


def _apply_overrides(config, *, steps: int, batch_size: int | None, chunk_size: int | None) -> None:
    config.training.num_steps = int(steps)
    config.training.num_epochs = 1
    config.training.read_metadata_procs = 1
    if batch_size is not None:
        config.training.batch_size = int(batch_size)
    if chunk_size is not None:
        config.audio.chunk_size = int(chunk_size)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _consume_original_roformer_python_rng(model_type: str) -> None:
    if model_type in {"bs_roformer", "mel_band_roformer"}:
        random.getrandbits(96)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _environment_metadata() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _plain_config(config):
    plain = to_plain(config)
    if hasattr(plain, "to_dict"):
        return plain.to_dict()
    if hasattr(plain, "items"):
        return {key: _plain_config(item) for key, item in plain.items()}
    if isinstance(plain, (list, tuple)):
        return [_plain_config(item) for item in plain]
    return plain


class _OriginalArgs:
    def __init__(self, *, model_type: str, start_check_point: str) -> None:
        self.model_type = model_type
        self.start_check_point = start_check_point
        self.load_only_compatible_weights = False
        self.lora_checkpoint_loralib = None
