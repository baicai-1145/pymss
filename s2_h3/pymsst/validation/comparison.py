from __future__ import annotations

import csv
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pymss_core import load_config, to_plain


DEFAULT_ORIGINAL_ROOT = Path("/root/Music-Source-Separation-Training")
DEFAULT_DATA_ROOT = Path("/root/pymsst/musdb18hq")
DEFAULT_SCRATCH_CONFIGS = {
    "bs_roformer": DEFAULT_ORIGINAL_ROOT / "configs/config_musdb18_bs_roformer.yaml",
    "mel_band_roformer": DEFAULT_ORIGINAL_ROOT / "configs/config_musdb18_mel_band_roformer.yaml",
}
DEFAULT_PRETRAINED_CONFIGS = {
    "bs_roformer": DEFAULT_ORIGINAL_ROOT / "configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
    "mel_band_roformer": DEFAULT_ORIGINAL_ROOT / "configs/viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml",
}
DEFAULT_PRETRAINED_CHECKPOINTS = {
    "bs_roformer": Path("/root/pymsst/pretrained/model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
    "mel_band_roformer": Path("/root/pymsst/pretrained/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"),
}
ORIGINAL_TRAINING_REQUIREMENTS = (
    "PyYAML",
    "librosa",
    "matplotlib",
    "ml-collections",
    "omegaconf",
    "pandas",
    "requests",
    "soundfile",
    "tqdm",
    "audiomentations",
    "auraloss",
    "pedalboard",
    "torch-l1-snr>=0.1.2",
    "torch-log-wmse>=0.3.1",
    "wandb",
    "beartype",
    "einops",
    "packaging",
    "rotary-embedding-torch",
    "PoPE-pytorch",
)


@dataclass(frozen=True)
class ComparisonRun:
    run_id: str
    implementation: str
    model_type: str
    mode: str
    seed: int
    cwd: str
    config_path: str
    results_path: str
    start_checkpoint: str | None
    loss_curve_path: str | None
    command: list[str]


@dataclass(frozen=True)
class CurveComparison:
    pymsst_points: int
    original_points: int
    compared_points: int
    mean_abs_diff: float
    max_abs_diff: float
    first_pymsst_loss: float
    first_original_loss: float
    last_pymsst_loss: float
    last_original_loss: float


def summarize_comparisons(
    comparison_paths: list[str | Path],
    *,
    min_points: int | None = None,
    max_mean_abs_diff: float | None = None,
    max_max_abs_diff: float | None = None,
) -> dict[str, Any]:
    if not comparison_paths:
        raise ValueError("at least one comparison path is required")

    records = [
        _comparison_summary_record(
            Path(path),
            min_points=min_points,
            max_mean_abs_diff=max_mean_abs_diff,
            max_max_abs_diff=max_max_abs_diff,
        )
        for path in comparison_paths
    ]
    mean_values = [record["mean_abs_diff"] for record in records]
    max_values = [record["max_abs_diff"] for record in records]
    point_values = [record["compared_points"] for record in records]
    failures = [
        {"path": record["path"], "failures": record["failures"]}
        for record in records
        if record["failures"]
    ]
    return {
        "schema_version": 1,
        "thresholds": {
            "min_points": min_points,
            "max_mean_abs_diff": max_mean_abs_diff,
            "max_max_abs_diff": max_max_abs_diff,
        },
        "aggregate": {
            "count": len(records),
            "min_compared_points": min(point_values),
            "max_mean_abs_diff": max(mean_values),
            "max_max_abs_diff": max(max_values),
            "passed": not failures,
        },
        "comparisons": records,
        "failures": failures,
    }


def write_comparison_plan(
    output_root: str | Path,
    *,
    pymsst_root: str | Path,
    original_root: str | Path = DEFAULT_ORIGINAL_ROOT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    steps: int = 1000,
    repeats: int = 3,
    base_seed: int = 0,
    include_pretrained: bool = True,
    batch_size: int | None = None,
    chunk_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    use_amp: bool | None = None,
    val_tracks: int = 0,
    accelerator: str = "gpu",
    devices: str = "1",
) -> dict[str, Any]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    runs: list[ComparisonRun] = []
    requirements_path = output_path / "requirements" / "original_training.txt"
    constraints_path = output_path / "requirements" / "constraints.txt"
    _write_original_requirements(requirements_path)
    _write_constraints(constraints_path)
    setup_script_path = output_path / "setup_original_env.sh"
    _write_setup_script(
        setup_script_path,
        pymsst_root=Path(pymsst_root),
        requirements_path=requirements_path,
        constraints_path=constraints_path,
    )
    original_valid_path = _prepare_validation_subset(output_path, Path(data_root), val_tracks)
    for mode, model_type, source_config, start_checkpoint in _iter_cases(include_pretrained):
        for repeat in range(repeats):
            seed = base_seed + repeat
            run_base = f"{model_type}_{mode}_seed{seed}"
            config_path = output_path / "configs" / f"{run_base}.yaml"
            _write_config_copy(
                source_config,
                config_path,
                steps=steps,
                batch_size=batch_size,
                chunk_size=chunk_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                use_amp=use_amp,
            )
            runs.append(
                _pymsst_run(
                    output_path,
                    pymsst_root=Path(pymsst_root),
                    data_root=Path(data_root),
                    model_type=model_type,
                    mode=mode,
                    seed=seed,
                    config_path=config_path,
                    start_checkpoint=start_checkpoint,
                    val_tracks=val_tracks,
                    accelerator=accelerator,
                    devices=devices,
                    steps=steps,
                )
            )
            runs.append(
                _original_run(
                    output_path,
                    pymsst_root=Path(pymsst_root),
                    original_root=Path(original_root),
                    data_root=Path(data_root),
                    valid_path=original_valid_path,
                    model_type=model_type,
                    mode=mode,
                    seed=seed,
                    config_path=config_path,
                    start_checkpoint=start_checkpoint,
                    steps=steps,
                )
            )

    plan = {
        "schema_version": 1,
        "steps": steps,
        "repeats": repeats,
        "base_seed": base_seed,
        "data_root": str(Path(data_root)),
        "setup_script": str(setup_script_path),
        "runs": [asdict(run) for run in runs],
    }
    (output_path / "comparison_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    _write_shell_script(output_path / "run_all.sh", runs, setup_script_path=setup_script_path)
    return plan


def compare_loss_curves(pymsst_curve: str | Path, original_curve_or_checkpoint: str | Path) -> CurveComparison:
    pymsst_losses = read_pymsst_loss_curve(pymsst_curve)
    original_losses = read_original_loss_curve(original_curve_or_checkpoint)
    compared = min(len(pymsst_losses), len(original_losses))
    if compared == 0:
        raise ValueError("no overlapping loss points to compare")

    diffs = [abs(pymsst_losses[index] - original_losses[index]) for index in range(compared)]
    return CurveComparison(
        pymsst_points=len(pymsst_losses),
        original_points=len(original_losses),
        compared_points=compared,
        mean_abs_diff=sum(diffs) / compared,
        max_abs_diff=max(diffs),
        first_pymsst_loss=pymsst_losses[0],
        first_original_loss=original_losses[0],
        last_pymsst_loss=pymsst_losses[compared - 1],
        last_original_loss=original_losses[compared - 1],
    )


def read_pymsst_loss_curve(path: str | Path) -> list[float]:
    losses: list[float] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("event") == "train" and row.get("loss"):
                losses.append(float(row["loss"]))
    return losses


def read_original_loss_curve(path: str | Path) -> list[float]:
    if Path(path).suffix.lower() == ".csv":
        return read_pymsst_loss_curve(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    all_losses = checkpoint.get("all_losses", {}) if isinstance(checkpoint, dict) else {}
    losses: list[float] = []
    for key in sorted(all_losses, key=_epoch_sort_key):
        losses.extend(float(value) for value in all_losses[key])
    return losses


def _comparison_summary_record(
    path: Path,
    *,
    min_points: int | None,
    max_mean_abs_diff: float | None,
    max_max_abs_diff: float | None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "path": str(path),
        "pymsst_points": int(payload["pymsst_points"]),
        "original_points": int(payload["original_points"]),
        "compared_points": int(payload["compared_points"]),
        "mean_abs_diff": float(payload["mean_abs_diff"]),
        "max_abs_diff": float(payload["max_abs_diff"]),
        "first_pymsst_loss": float(payload["first_pymsst_loss"]),
        "first_original_loss": float(payload["first_original_loss"]),
        "last_pymsst_loss": float(payload["last_pymsst_loss"]),
        "last_original_loss": float(payload["last_original_loss"]),
        "failures": [],
    }
    if min_points is not None and record["compared_points"] < min_points:
        record["failures"].append(f"compared_points {record['compared_points']} < {min_points}")
    if max_mean_abs_diff is not None and record["mean_abs_diff"] > max_mean_abs_diff:
        record["failures"].append(f"mean_abs_diff {record['mean_abs_diff']} > {max_mean_abs_diff}")
    if max_max_abs_diff is not None and record["max_abs_diff"] > max_max_abs_diff:
        record["failures"].append(f"max_abs_diff {record['max_abs_diff']} > {max_max_abs_diff}")
    record["passed"] = not record["failures"]
    return record


def _iter_cases(include_pretrained: bool):
    for model_type, config_path in DEFAULT_SCRATCH_CONFIGS.items():
        yield "scratch", model_type, config_path, None
    if include_pretrained:
        for model_type, config_path in DEFAULT_PRETRAINED_CONFIGS.items():
            yield "pretrained", model_type, config_path, DEFAULT_PRETRAINED_CHECKPOINTS[model_type]


def _write_config_copy(
    source_config: Path,
    output_config: Path,
    *,
    steps: int,
    batch_size: int | None,
    chunk_size: int | None,
    gradient_accumulation_steps: int | None,
    use_amp: bool | None,
) -> None:
    config = to_plain(load_config(source_config))
    config.setdefault("audio", {})
    config.setdefault("training", {})
    if chunk_size is not None:
        config["audio"]["chunk_size"] = int(chunk_size)
    config["training"]["num_epochs"] = 1
    config["training"]["num_steps"] = int(steps)
    config["training"]["read_metadata_procs"] = 1
    if batch_size is not None:
        config["training"]["batch_size"] = int(batch_size)
    if gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    if use_amp is not None:
        config["training"]["use_amp"] = bool(use_amp)

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")


def _pymsst_run(
    output_root: Path,
    *,
    pymsst_root: Path,
    data_root: Path,
    model_type: str,
    mode: str,
    seed: int,
    config_path: Path,
    start_checkpoint: Path | None,
    val_tracks: int,
    accelerator: str,
    devices: str,
    steps: int,
) -> ComparisonRun:
    run_id = f"pymsst_{model_type}_{mode}_seed{seed}"
    results_path = output_root / "pymsst" / run_id
    command = [
        "uv",
        "run",
        "pymsst",
        "train",
        "--model-type",
        model_type,
        "--config-path",
        str(config_path),
        "--data-root",
        str(data_root),
        "--results-path",
        str(results_path),
        "--seed",
        str(seed),
        "--steps-per-epoch",
        str(steps),
        "--max-epochs",
        "1",
        "--val-tracks",
        str(val_tracks),
        "--accelerator",
        accelerator,
        "--devices",
        devices,
        "--original-loss-compat",
        "--export-core-checkpoint",
        str(results_path / f"{run_id}_core.ckpt"),
    ]
    if start_checkpoint:
        command.extend(["--start-checkpoint", str(start_checkpoint)])
    return ComparisonRun(
        run_id=run_id,
        implementation="pymsst",
        model_type=model_type,
        mode=mode,
        seed=seed,
        cwd=str(pymsst_root),
        config_path=str(config_path),
        results_path=str(results_path),
        start_checkpoint=str(start_checkpoint) if start_checkpoint else None,
        loss_curve_path=str(results_path / model_type / "loss_curve.csv"),
        command=command,
    )


def _original_run(
    output_root: Path,
    *,
    pymsst_root: Path,
    original_root: Path,
    data_root: Path,
    valid_path: Path,
    model_type: str,
    mode: str,
    seed: int,
    config_path: Path,
    start_checkpoint: Path | None,
    steps: int,
) -> ComparisonRun:
    run_id = f"original_{model_type}_{mode}_seed{seed}"
    results_path = output_root / "original" / run_id
    command = [
        str(pymsst_root / ".venv" / "bin" / "python"),
        "train.py",
        "--model_type",
        model_type,
        "--config_path",
        str(config_path),
        "--data_path",
        str(data_root / "train"),
        "--valid_path",
        str(valid_path),
        "--results_path",
        str(results_path),
        "--dataset_type",
        "1",
        "--num_workers",
        "0",
        "--seed",
        str(seed),
        "--device_ids",
        "0",
        "--use_standard_loss",
        "--loss",
        "masked_loss",
        "--save_weights_every_epoch",
    ]
    if start_checkpoint:
        command.extend(["--start_check_point", str(start_checkpoint)])
    return ComparisonRun(
        run_id=run_id,
        implementation="original",
        model_type=model_type,
        mode=mode,
        seed=seed,
        cwd=str(original_root),
        config_path=str(config_path),
        results_path=str(results_path),
        start_checkpoint=str(start_checkpoint) if start_checkpoint else None,
        loss_curve_path=None,
        command=command,
    )


def _write_original_requirements(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ORIGINAL_TRAINING_REQUIREMENTS) + "\n", encoding="utf-8")


def _prepare_validation_subset(output_root: Path, data_root: Path, val_tracks: int) -> Path:
    train_root = data_root / "train"
    subset_root = output_root / "validation" / "train_subset"
    subset_root.mkdir(parents=True, exist_ok=True)
    for existing in subset_root.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            for child in existing.iterdir():
                child.unlink()
            existing.rmdir()
    if not train_root.is_dir() or val_tracks <= 0:
        return subset_root

    for track in sorted(item for item in train_root.iterdir() if item.is_dir())[:val_tracks]:
        subset_track = subset_root / track.name
        subset_track.mkdir()
        for audio_file in sorted(track.glob("*.wav")):
            (subset_track / audio_file.name).symlink_to(audio_file)
    return subset_root


def _write_constraints(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio_version = torch.__version__
    path.write_text(
        f"torch=={torch.__version__}\n"
        f"torchaudio=={torchaudio_version}\n"
        f"numpy=={np.__version__}\n",
        encoding="utf-8",
    )


def _write_setup_script(path: Path, *, pymsst_root: Path, requirements_path: Path, constraints_path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "uv pip install "
        f"--python {shlex.quote(str(pymsst_root / '.venv' / 'bin' / 'python'))} "
        "--index https://download.pytorch.org/whl/cu128 "
        f"-r {shlex.quote(str(requirements_path))} "
        f"-c {shlex.quote(str(constraints_path))}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _write_shell_script(path: Path, runs: list[ComparisonRun], *, setup_script_path: Path) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append(str(setup_script_path))
    lines.append("")
    for run in runs:
        lines.append(f"mkdir -p {shlex.quote(run.results_path)}")
        lines.append(f"cd {shlex.quote(run.cwd)}")
        lines.append(" ".join(shlex.quote(part) for part in run.command))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _epoch_sort_key(key: str) -> tuple[int, str]:
    if key.startswith("epoch_"):
        try:
            return int(key.split("_", 1)[1]), key
        except ValueError:
            pass
    return sys.maxsize, key


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
