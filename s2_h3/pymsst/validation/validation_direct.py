from __future__ import annotations

import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pymsst import __version__
from pymsst.training import SeparationLightningModule

from .direct_compare import (
    _apply_overrides,
    _consume_original_roformer_python_rng,
    _environment_metadata,
    _jsonable,
    _plain_config,
    _seed_everything,
)


def run_direct_validation_metrics(
    *,
    implementation: str,
    model_type: str,
    config_path: str | Path,
    data_root: str | Path,
    results_path: str | Path,
    seed: int = 0,
    val_tracks: int = 1,
    start_checkpoint: str | Path | None = None,
    original_root: str | Path = "/root/Music-Source-Separation-Training",
    batch_size: int | None = None,
    chunk_size: int | None = None,
) -> dict[str, Any]:
    if implementation == "pymsst":
        return _run_pymsst_validation(
            model_type=model_type,
            config_path=Path(config_path),
            data_root=Path(data_root),
            results_path=Path(results_path),
            seed=seed,
            val_tracks=val_tracks,
            start_checkpoint=Path(start_checkpoint) if start_checkpoint else None,
            original_root=Path(original_root),
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
    if implementation == "original":
        return _run_original_validation(
            model_type=model_type,
            config_path=Path(config_path),
            data_root=Path(data_root),
            results_path=Path(results_path),
            seed=seed,
            val_tracks=val_tracks,
            start_checkpoint=Path(start_checkpoint) if start_checkpoint else None,
            original_root=Path(original_root),
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
    raise ValueError("implementation must be 'pymsst' or 'original'")


def compare_validation_metrics(
    pymsst_metrics_path: str | Path,
    original_metrics_path: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    pymsst_payload = json.loads(Path(pymsst_metrics_path).read_text(encoding="utf-8"))
    original_payload = json.loads(Path(original_metrics_path).read_text(encoding="utf-8"))
    pymsst_metrics = pymsst_payload["metric_avg"]
    original_metrics = original_payload["metric_avg"]
    common = sorted(set(pymsst_metrics) & set(original_metrics))
    diffs = {
        name: abs(float(pymsst_metrics[name]) - float(original_metrics[name]))
        for name in common
    }
    payload = {
        "schema_version": 1,
        "common_metrics": common,
        "diffs": diffs,
        "max_abs_diff": max(diffs.values()) if diffs else None,
        "pymsst_metric_avg": pymsst_metrics,
        "original_metric_avg": original_metrics,
    }
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def _run_pymsst_validation(
    *,
    model_type: str,
    config_path: Path,
    data_root: Path,
    results_path: Path,
    seed: int,
    val_tracks: int,
    start_checkpoint: Path | None,
    original_root: Path,
    batch_size: int | None,
    chunk_size: int | None,
) -> dict[str, Any]:
    _seed_everything(seed)
    module = SeparationLightningModule(
        model_type=model_type,
        config_path=config_path,
        original_loss_compat=True,
        start_checkpoint=start_checkpoint,
        strict_checkpoint=True,
        optimizer_backend="default",
    )
    _consume_original_roformer_python_rng(model_type)
    config = module.config
    _apply_overrides(config, steps=1, batch_size=batch_size, chunk_size=chunk_size)
    return _run_validation(
        implementation="pymsst",
        model=module.model,
        config=config,
        model_type=model_type,
        config_path=config_path,
        data_root=data_root,
        results_path=results_path,
        seed=seed,
        val_tracks=val_tracks,
        start_checkpoint=start_checkpoint,
        original_root=original_root,
    )


def _run_original_validation(
    *,
    model_type: str,
    config_path: Path,
    data_root: Path,
    results_path: Path,
    seed: int,
    val_tracks: int,
    start_checkpoint: Path | None,
    original_root: Path,
    batch_size: int | None,
    chunk_size: int | None,
) -> dict[str, Any]:
    _seed_everything(seed)
    if str(original_root) not in sys.path:
        sys.path.insert(0, str(original_root))
    from utils.settings import get_model_from_config
    from utils.model_utils import load_start_checkpoint

    model, config = get_model_from_config(model_type, str(config_path))
    _apply_overrides(config, steps=1, batch_size=batch_size, chunk_size=chunk_size)
    if start_checkpoint is not None:
        load_start_checkpoint(
            Namespace(model_type=model_type, start_check_point=str(start_checkpoint)),
            model,
            type_="valid",
        )
    return _run_validation(
        implementation="original",
        model=model,
        config=config,
        model_type=model_type,
        config_path=config_path,
        data_root=data_root,
        results_path=results_path,
        seed=seed,
        val_tracks=val_tracks,
        start_checkpoint=start_checkpoint,
        original_root=original_root,
    )


def _run_validation(
    *,
    implementation: str,
    model: torch.nn.Module,
    config: Any,
    model_type: str,
    config_path: Path,
    data_root: Path,
    results_path: Path,
    seed: int,
    val_tracks: int,
    start_checkpoint: Path | None,
    original_root: Path,
) -> dict[str, Any]:
    if str(original_root) not in sys.path:
        sys.path.insert(0, str(original_root))
    from valid import valid
    import utils.metrics as original_metrics

    original_sdr = original_metrics.sdr

    def scalar_sdr(references: np.ndarray, estimates: np.ndarray) -> float:
        return float(np.asarray(original_sdr(references, estimates)).mean())

    original_metrics.sdr = scalar_sdr

    results_path.mkdir(parents=True, exist_ok=True)
    valid_path = _prepare_validation_subset(results_path / "validation_subset", data_root, val_tracks)
    args = Namespace(
        valid_path=str(valid_path),
        metrics=["sdr"],
        model_type=model_type,
        store_dir="",
        extension="wav",
        use_tta=False,
    )
    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode():
        metric_avg, all_metrics = valid(model, args, config, device, verbose=False)
    payload = {
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
        "val_tracks": val_tracks,
        "valid_path": str(valid_path),
        "metric_avg": metric_avg,
        "all_metrics": all_metrics,
        "config": _plain_config(config),
        "environment": _environment_metadata(),
    }
    output_path = results_path / "validation_metrics.json"
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def _prepare_validation_subset(path: Path, data_root: Path, val_tracks: int) -> Path:
    split_path = data_root / "train"
    tracks = sorted(track for track in split_path.iterdir() if (track / "mixture.wav").exists())
    if val_tracks <= 0:
        raise ValueError("val_tracks must be positive")
    selected = tracks[:val_tracks]
    if len(selected) < val_tracks:
        raise ValueError(f"requested {val_tracks} validation tracks, found {len(selected)} in {split_path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for track in selected:
        track_path = path / track.name
        track_path.mkdir()
        for stem in ("mixture", "vocals", "drums", "bass", "other"):
            source = track / f"{stem}.wav"
            if source.exists():
                (track_path / source.name).symlink_to(source)
    return path
