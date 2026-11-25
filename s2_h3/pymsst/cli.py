from __future__ import annotations

import argparse
import json
import os
import platform
import random
from glob import glob
from pathlib import Path

from . import __version__
from .data.musdb import DEFAULT_STEMS, scan_musdb_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pymsst", description="Training tools for pymss-core models.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    check_musdb = subparsers.add_parser("check-musdb", help="Check a MUSDB18-HQ style dataset layout.")
    check_musdb.add_argument("root", type=Path, help="Dataset root containing train/test split directories.")
    check_musdb.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="Split to check. Can be passed multiple times. Defaults to train and test.",
    )
    check_musdb.set_defaults(func=_check_musdb)

    train = subparsers.add_parser("train", help="Train a pymss-core BSR/MBR model with PyTorch Lightning.")
    train.add_argument("--model-type", choices=("bs_roformer", "mel_band_roformer"), required=True)
    train.add_argument("--config-path", type=Path, required=True)
    train.add_argument("--data-root", type=Path, default=Path("/root/pymsst/musdb18hq"))
    train.add_argument("--results-path", type=Path, default=Path("runs"))
    train.add_argument("--start-checkpoint", type=Path, help="Initial model weights to load before training.")
    train.add_argument("--resume-checkpoint", type=Path, help="Lightning checkpoint path used to resume training.")
    train.add_argument("--strict-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--loss", choices=("masked_loss", "mse_loss", "l1_loss"), default="masked_loss")
    train.add_argument(
        "--original-loss-compat",
        action="store_true",
        help="Match the original training repo's single-stem target shape and masked_loss broadcasting behavior.",
    )
    train.add_argument("--batch-size", type=int)
    train.add_argument("--chunk-size", type=int)
    train.add_argument("--steps-per-epoch", type=int)
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--pin-memory", action="store_true")
    train.add_argument("--persistent-workers", action="store_true")
    train.add_argument("--val-fraction", type=float, default=0.1)
    train.add_argument("--val-tracks", type=int)
    train.add_argument("--accelerator", default="auto")
    train.add_argument("--devices", default="auto", help="Lightning devices value: auto, an integer, or comma-separated ids.")
    train.add_argument("--strategy", default="auto")
    train.add_argument("--precision", default="auto")
    train.add_argument("--optimizer-backend", choices=("default", "foreach", "fused", "fused_capturable"), default=None)
    train.add_argument("--cuda-attention-backend", choices=("auto", "default", "cudnn", "efficient", "math", "flash"), default=None)
    train.add_argument("--fused-kernel-backend", choices=("off", "cuda_ext", "g4"), default=None)
    train.add_argument("--fused-rms-norm-mode", choices=("off", "fast", "mixed"), default=None)
    train.add_argument("--cuda-graph-mode", choices=("off", "mask_core", "stft_mask_core"), default=None)
    train.add_argument("--runtime-optimization-preset", choices=("off", "train_fragmentation", "high_util"), default=None)
    train.add_argument("--train-log-interval", type=int, default=1)
    train.add_argument("--loss-history-interval", type=int, default=1)
    train.add_argument("--log-every-n-steps", type=int, default=1)
    train.add_argument("--disable-progress-bar", action="store_true")
    train.add_argument("--accumulate-grad-batches", type=int)
    train.add_argument("--gradient-clip-val", type=float)
    train.add_argument("--export-core-checkpoint", type=Path)
    train.add_argument("--loss-history-path", type=Path)
    train.set_defaults(func=_train)

    compare_plan = subparsers.add_parser(
        "compare-plan",
        help="Create a reproducible pymsst/original loss-curve comparison plan.",
    )
    compare_plan.add_argument("--output-root", type=Path, required=True)
    compare_plan.add_argument("--pymsst-root", type=Path, default=Path("/root/pymsst"))
    compare_plan.add_argument("--original-root", type=Path, default=Path("/root/Music-Source-Separation-Training"))
    compare_plan.add_argument("--data-root", type=Path, default=Path("/root/pymsst/musdb18hq"))
    compare_plan.add_argument("--steps", type=int, default=1000)
    compare_plan.add_argument("--repeats", type=int, default=3)
    compare_plan.add_argument("--base-seed", type=int, default=0)
    compare_plan.add_argument("--scratch-only", action="store_true")
    compare_plan.add_argument("--batch-size", type=int)
    compare_plan.add_argument("--chunk-size", type=int)
    compare_plan.add_argument("--gradient-accumulation-steps", type=int)
    compare_plan.add_argument("--use-amp", action=argparse.BooleanOptionalAction)
    compare_plan.add_argument("--val-tracks", type=int, default=0)
    compare_plan.add_argument("--accelerator", default="gpu")
    compare_plan.add_argument("--devices", default="1")
    compare_plan.set_defaults(func=_compare_plan)

    compare_loss = subparsers.add_parser("compare-loss", help="Compare a pymsst loss CSV with an original checkpoint.")
    compare_loss.add_argument("--pymsst-loss-curve", type=Path, required=True)
    compare_loss.add_argument("--original-checkpoint", type=Path, required=True)
    compare_loss.add_argument("--output", type=Path)
    compare_loss.set_defaults(func=_compare_loss)

    compare_summary = subparsers.add_parser(
        "compare-summary",
        help="Summarize one or more loss comparison JSON files and optionally enforce thresholds.",
    )
    compare_summary.add_argument("comparison", nargs="*", type=Path, help="comparison.json file to include.")
    compare_summary.add_argument("--glob", action="append", default=[], help="Glob pattern for comparison.json files.")
    compare_summary.add_argument("--min-points", type=int)
    compare_summary.add_argument("--max-mean-abs-diff", type=float)
    compare_summary.add_argument("--max-max-abs-diff", type=float)
    compare_summary.add_argument("--output", type=Path)
    compare_summary.set_defaults(func=_compare_summary)

    compare_direct = subparsers.add_parser(
        "compare-direct",
        help="Run one lightweight pymsst/original training loss curve without saving checkpoints.",
    )
    compare_direct.add_argument("--implementation", choices=("pymsst", "original"), required=True)
    compare_direct.add_argument("--model-type", choices=("bs_roformer", "mel_band_roformer"), required=True)
    compare_direct.add_argument("--config-path", type=Path, required=True)
    compare_direct.add_argument("--data-root", type=Path, default=Path("/root/pymsst/musdb18hq"))
    compare_direct.add_argument("--results-path", type=Path, required=True)
    compare_direct.add_argument("--steps", type=int, required=True)
    compare_direct.add_argument("--seed", type=int, default=0)
    compare_direct.add_argument("--start-checkpoint", type=Path)
    compare_direct.add_argument("--original-root", type=Path, default=Path("/root/Music-Source-Separation-Training"))
    compare_direct.add_argument("--batch-size", type=int)
    compare_direct.add_argument("--chunk-size", type=int)
    compare_direct.add_argument("--num-workers", type=int, default=0)
    compare_direct.add_argument("--pin-memory", action="store_true")
    compare_direct.add_argument("--persistent-workers", action="store_true")
    compare_direct.add_argument("--loss-record-interval", type=int, default=1)
    compare_direct.add_argument("--optimizer-backend", choices=("default", "foreach", "fused", "fused_capturable"), default=None)
    compare_direct.add_argument("--fused-kernel-backend", choices=("off", "cuda_ext", "g4"), default=None)
    compare_direct.add_argument("--fused-rms-norm-mode", choices=("off", "fast", "mixed"), default=None)
    compare_direct.add_argument("--cuda-graph-mode", choices=("off", "mask_core", "stft_mask_core"), default=None)
    compare_direct.add_argument("--runtime-optimization-preset", choices=("off", "train_fragmentation", "high_util"), default=None)
    compare_direct.add_argument(
        "--cuda-attention-backend",
        choices=("auto", "default", "cudnn", "efficient", "math", "flash"),
        default=None,
    )
    compare_direct.set_defaults(func=_compare_direct)

    compare_validation = subparsers.add_parser(
        "compare-validation-direct",
        help="Run original-style whole-track validation metrics for pymsst or original.",
    )
    compare_validation.add_argument("--implementation", choices=("pymsst", "original"), required=True)
    compare_validation.add_argument("--model-type", choices=("bs_roformer", "mel_band_roformer"), required=True)
    compare_validation.add_argument("--config-path", type=Path, required=True)
    compare_validation.add_argument("--data-root", type=Path, default=Path("/root/pymsst/musdb18hq"))
    compare_validation.add_argument("--results-path", type=Path, required=True)
    compare_validation.add_argument("--seed", type=int, default=0)
    compare_validation.add_argument("--val-tracks", type=int, default=1)
    compare_validation.add_argument("--start-checkpoint", type=Path)
    compare_validation.add_argument("--original-root", type=Path, default=Path("/root/Music-Source-Separation-Training"))
    compare_validation.add_argument("--batch-size", type=int)
    compare_validation.add_argument("--chunk-size", type=int)
    compare_validation.set_defaults(func=_compare_validation_direct)

    compare_validation_metrics = subparsers.add_parser(
        "compare-validation-metrics",
        help="Compare pymsst and original validation_metrics.json files.",
    )
    compare_validation_metrics.add_argument("--pymsst-metrics", type=Path, required=True)
    compare_validation_metrics.add_argument("--original-metrics", type=Path, required=True)
    compare_validation_metrics.add_argument("--output", type=Path)
    compare_validation_metrics.set_defaults(func=_compare_validation_metrics)

    return parser


def _check_musdb(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    splits = tuple(args.splits or ("train", "test"))
    scans = scan_musdb_root(root, splits=splits, stems=DEFAULT_STEMS)

    ok = True
    for split, scan in scans.items():
        print(f"{split}: {len(scan.tracks)} complete tracks")
        if scan.incomplete:
            ok = False
            print(f"{split}: {len(scan.incomplete)} incomplete tracks")
            for item in scan.incomplete[:10]:
                missing = ", ".join(item.missing_files)
                print(f"  - {item.name}: missing {missing}")
            if len(scan.incomplete) > 10:
                print(f"  ... {len(scan.incomplete) - 10} more")

    return 0 if ok else 1


def _train(args: argparse.Namespace) -> int:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    CHECKPOINT_EVERY_N_STEPS = 500
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pymss_core import to_plain

    from .data.datamodule import MusdbDataModule
    from .export import export_core_checkpoint
    from .training import ExponentialMovingAverage, LossCurveRecorder, SeparationLightningModule

    pl.seed_everything(args.seed, workers=True)

    module = SeparationLightningModule(
        model_type=args.model_type,
        config_path=args.config_path,
        loss_name=args.loss,
        original_loss_compat=args.original_loss_compat,
        start_checkpoint=args.start_checkpoint,
        strict_checkpoint=args.strict_checkpoint,
        optimizer_backend=args.optimizer_backend,
        cuda_attention_backend=args.cuda_attention_backend,
        fused_kernel_backend=args.fused_kernel_backend,
        fused_rms_norm_mode=args.fused_rms_norm_mode,
        cuda_graph_mode=args.cuda_graph_mode,
        runtime_optimization_preset=args.runtime_optimization_preset,
        train_log_interval=args.train_log_interval,
    )
    _consume_original_roformer_python_rng(args.model_type)
    config = module.config
    if args.chunk_size is not None:
        config.audio.chunk_size = int(args.chunk_size)
    batch_size = args.batch_size or int(config.training.batch_size)
    steps_per_epoch = args.steps_per_epoch or int(config.training.num_steps)
    max_epochs = args.max_epochs or int(config.training.num_epochs)
    accumulate_grad_batches = args.accumulate_grad_batches or int(config.training.get("gradient_accumulation_steps", 1))
    gradient_clip_val = args.gradient_clip_val
    if gradient_clip_val is None:
        gradient_clip_val = float(config.training.get("grad_clip", 0) or 0)
    precision = _resolve_precision(args.precision, config, torch.cuda.is_available())

    data = MusdbDataModule(
        args.data_root,
        instruments=tuple(config.training.instruments),
        target_instrument=config.training.get("target_instrument"),
        chunk_size=int(config.audio.chunk_size),
        sample_rate=int(config.audio.sample_rate),
        channels=int(config.audio.num_channels),
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        val_fraction=args.val_fraction,
        val_tracks=args.val_tracks,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        min_mean_abs=float(config.audio.get("min_mean_abs", 0.0)),
        mix_from_stems=True,
        loudness_augmentation=bool(config.get("augmentations", {}).get("enable", False))
        and bool(config.get("augmentations", {}).get("loudness", False)),
        loudness_min=float(config.get("augmentations", {}).get("loudness_min", 0.5)),
        loudness_max=float(config.get("augmentations", {}).get("loudness_max", 1.5)),
        keep_target_stem_dim=args.original_loss_compat,
    )

    run_dir = args.results_path / args.model_type
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_history_path = args.loss_history_path or run_dir / "loss_curve.csv"
    if _is_rank_zero_process():
        _write_json(
            run_dir / "run_metadata.json",
            {
                "schema_version": 1,
                "package": "pymsst",
                "version": __version__,
                "model_type": args.model_type,
                "config_path": str(args.config_path),
                "data_root": str(args.data_root),
                "results_path": str(args.results_path),
                "start_checkpoint": str(args.start_checkpoint) if args.start_checkpoint else None,
                "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
                "seed": args.seed,
                "batch_size": batch_size,
                "chunk_size": int(config.audio.chunk_size),
                "steps_per_epoch": steps_per_epoch,
                "max_epochs": max_epochs,
                "max_steps": args.max_steps,
                "accumulate_grad_batches": accumulate_grad_batches,
                "gradient_clip_val": gradient_clip_val,
                "precision": precision,
                "optimizer_backend": module.optimizer_backend,
                "runtime_optimization_preset": module.runtime_optimization_preset,
                "cuda_attention_backend": module.cuda_attention_backend,
                "fused_kernel_backend": module.fused_kernel_backend,
                "fused_rms_norm_mode": module.fused_rms_norm_mode,
                "fused_kernel_active": module.fused_training_status.active,
                "fused_kernel_reason": module.fused_training_status.reason,
                "fused_kernel_patched_band_splits": module.fused_training_status.patched_band_splits,
                "fused_kernel_patched_band_split_bwd_v2": module.fused_training_status.patched_band_split_bwd_v2,
                "fused_kernel_patched_band_split_bwd_v3": module.fused_training_status.patched_band_split_bwd_v3,
                "fused_kernel_patched_band_split_ragged_forward": module.fused_training_status.patched_band_split_ragged_forward,
                "fused_kernel_patched_band_split_ragged_backward": (
                    module.fused_training_status.patched_band_split_ragged_backward
                ),
                "fused_kernel_patched_rotary": module.fused_training_status.patched_rotary,
                "fused_kernel_patched_rms_norm": module.fused_training_status.patched_rms_norm,
                "fused_kernel_rms_norm_mode": module.fused_training_status.rms_norm_mode,
                "fused_kernel_patched_rms_norm_bwd_v2": module.fused_training_status.patched_rms_norm_bwd_v2,
                "fused_kernel_patched_single_stem_masks": module.fused_training_status.patched_single_stem_masks,
                "fused_kernel_patched_attention_gates": module.fused_training_status.patched_attention_gates,
                "fused_kernel_attention_gate_mode": module.fused_training_status.attention_gate_mode,
                "fused_kernel_patched_layout_reduce_copy": module.fused_training_status.patched_layout_reduce_copy,
                "fused_kernel_patched_train_param_pack_autograd": (
                    module.fused_training_status.patched_train_param_pack_autograd
                ),
                "fused_kernel_patched_mask_final_autograd": (
                    module.fused_training_status.patched_mask_final_autograd
                ),
                "fused_kernel_patched_mask_estimator_final": module.fused_training_status.patched_mask_estimator_final,
                "fused_kernel_mask_estimator_final_mode": module.fused_training_status.mask_estimator_final_mode,
                "cuda_graph_mode": module.cuda_graph_mode,
                "cuda_graph_active": module.cuda_graph_status.active,
                "cuda_graph_reason": module.cuda_graph_status.reason,
                "train_log_interval": args.train_log_interval,
                "loss_history_interval": args.loss_history_interval,
                "log_every_n_steps": args.log_every_n_steps,
                "progress_bar": not args.disable_progress_bar,
                "loss": args.loss,
                "original_loss_compat": args.original_loss_compat,
                "loss_history_path": str(loss_history_path),
                "config": to_plain(config),
                "environment": _environment_metadata(torch),
            },
        )
    no_validation = args.val_tracks is not None and args.val_tracks <= 0
    if no_validation and str(config.training.get("scheduler", "ReduceLROnPlateau")) == "ReduceLROnPlateau":
        config.training.scheduler = "off"
    if no_validation:
        checkpoint = ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="{epoch:03d}-{step:06d}",
            save_last=True,
            every_n_train_steps=CHECKPOINT_EVERY_N_STEPS,
            save_top_k=0,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        )
    else:
        checkpoint = ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="{epoch:03d}-{step:06d}-{val_sdr:.4f}",
            monitor="val_sdr",
            mode="max",
            save_last=True,
            every_n_train_steps=CHECKPOINT_EVERY_N_STEPS,
            save_top_k=1,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
    callbacks = [checkpoint, LossCurveRecorder(loss_history_path, interval=args.loss_history_interval)]
    ema_momentum = float(config.training.get("ema_momentum", 0) or 0)
    if ema_momentum > 0:
        callbacks.append(ExponentialMovingAverage(ema_momentum))
    trainer_kwargs = {"num_sanity_val_steps": 0} if no_validation else {}
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        strategy=args.strategy,
        precision=precision,
        max_epochs=max_epochs,
        max_steps=args.max_steps or -1,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        default_root_dir=run_dir,
        callbacks=callbacks,
        logger=_training_loggers(run_dir, CSVLogger, TensorBoardLogger),
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=not args.disable_progress_bar,
        **trainer_kwargs,
    )
    if no_validation:
        data.setup("fit")
        trainer.fit(module, train_dataloaders=data.train_dataloader(), ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(module, datamodule=data, ckpt_path=args.resume_checkpoint)

    if trainer.is_global_zero:
        _update_json(
            run_dir / "run_metadata.json",
            {
                "cuda_graph_active": module.cuda_graph_status.active,
                "cuda_graph_reason": module.cuda_graph_status.reason,
                "ema_active": module.ema_decay is not None,
                "ema_decay": module.ema_decay,
                "scheduler": str(config.training.get("scheduler", "ReduceLROnPlateau")),
                "config": to_plain(config),
            },
        )

    if args.export_core_checkpoint and trainer.is_global_zero:
        export_model = module.ema_model.module if module.ema_model is not None else module.model
        export_core_checkpoint(
            export_model,
            args.export_core_checkpoint,
            model_type=args.model_type,
            config=config,
            metadata={
                "source": "pymsst",
                "lightning_checkpoint": str(checkpoint.last_model_path or ""),
                "global_step": int(trainer.global_step),
                "ema_active": module.ema_model is not None,
                "ema_decay": module.ema_decay,
            },
        )
        print(f"Exported pymss-core checkpoint: {args.export_core_checkpoint}")

    return 0


def _compare_plan(args: argparse.Namespace) -> int:
    from .validation import write_comparison_plan

    plan = write_comparison_plan(
        args.output_root,
        pymsst_root=args.pymsst_root,
        original_root=args.original_root,
        data_root=args.data_root,
        steps=args.steps,
        repeats=args.repeats,
        base_seed=args.base_seed,
        include_pretrained=not args.scratch_only,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        val_tracks=args.val_tracks,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    print(f"Wrote comparison plan: {args.output_root / 'comparison_plan.json'}")
    print(f"Wrote run script: {args.output_root / 'run_all.sh'}")
    print(f"Runs: {len(plan['runs'])}")
    return 0


def _compare_loss(args: argparse.Namespace) -> int:
    from .validation import compare_loss_curves

    result = compare_loss_curves(args.pymsst_loss_curve, args.original_checkpoint)
    payload = result.__dict__
    if args.output:
        _write_json(args.output, payload)
        print(f"Wrote loss comparison: {args.output}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _compare_summary(args: argparse.Namespace) -> int:
    from .validation import summarize_comparisons

    comparison_paths = [*args.comparison]
    for pattern in args.glob:
        comparison_paths.extend(Path(path) for path in sorted(glob(pattern)))
    payload = summarize_comparisons(
        comparison_paths,
        min_points=args.min_points,
        max_mean_abs_diff=args.max_mean_abs_diff,
        max_max_abs_diff=args.max_max_abs_diff,
    )
    if args.output:
        _write_json(args.output, payload)
        print(f"Wrote comparison summary: {args.output}")
    else:
        print(json.dumps(payload, indent=2))
    return 0 if payload["aggregate"]["passed"] else 1


def _compare_direct(args: argparse.Namespace) -> int:
    from .validation import run_direct_loss_curve

    metadata = run_direct_loss_curve(
        implementation=args.implementation,
        model_type=args.model_type,
        config_path=args.config_path,
        data_root=args.data_root,
        results_path=args.results_path,
        steps=args.steps,
        seed=args.seed,
        start_checkpoint=args.start_checkpoint,
        original_root=args.original_root,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        loss_record_interval=args.loss_record_interval,
        optimizer_backend=args.optimizer_backend,
        cuda_attention_backend=args.cuda_attention_backend,
        fused_kernel_backend=args.fused_kernel_backend,
        fused_rms_norm_mode=args.fused_rms_norm_mode,
        cuda_graph_mode=args.cuda_graph_mode,
        runtime_optimization_preset=args.runtime_optimization_preset,
    )
    print(f"Wrote direct loss curve: {metadata['loss_curve_path']}")
    return 0


def _compare_validation_direct(args: argparse.Namespace) -> int:
    from .validation import run_direct_validation_metrics

    metadata = run_direct_validation_metrics(
        implementation=args.implementation,
        model_type=args.model_type,
        config_path=args.config_path,
        data_root=args.data_root,
        results_path=args.results_path,
        seed=args.seed,
        val_tracks=args.val_tracks,
        start_checkpoint=args.start_checkpoint,
        original_root=args.original_root,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    print(f"Wrote validation metrics: {Path(metadata['results_path']) / 'validation_metrics.json'}")
    return 0


def _compare_validation_metrics(args: argparse.Namespace) -> int:
    from .validation import compare_validation_metrics

    payload = compare_validation_metrics(
        args.pymsst_metrics,
        args.original_metrics,
        output=args.output,
    )
    if args.output:
        print(f"Wrote validation comparison: {args.output}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _consume_original_roformer_python_rng(model_type: str) -> None:
    if model_type in {"bs_roformer", "mel_band_roformer"}:
        # The original training process lazily imports RoFormer modules after
        # seeding, and that import consumes three Mersenne Twister words before
        # data sampling starts. Preserve seed-compatible MUSDB stem selection.
        random.getrandbits(96)


def _parse_devices(value: str) -> str | int | list[int]:
    if value == "auto":
        return value
    if "," in value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        return value


def _training_loggers(run_dir: Path, csv_logger_cls, tensorboard_logger_cls):
    loggers = [csv_logger_cls(save_dir=run_dir, name="logs")]
    try:
        loggers.append(tensorboard_logger_cls(save_dir=run_dir, name="tensorboard"))
    except (ImportError, ModuleNotFoundError):
        pass
    return loggers


def _is_rank_zero_process() -> bool:
    rank = os.environ.get("RANK")
    if rank is not None:
        return rank == "0"
    local_rank = os.environ.get("LOCAL_RANK")
    return local_rank in {None, "0"}


def _resolve_precision(value: str, config, cuda_available: bool) -> str:
    if value != "auto":
        return value
    use_amp = bool(config.training.get("use_amp", False))
    return "16-mixed" if use_amp and cuda_available else "32-true"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _update_json(path: Path, updates: dict) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(_jsonable(updates))
    _write_json(path, payload)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _environment_metadata(torch_module) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_device_count": int(torch_module.cuda.device_count()) if torch_module.cuda.is_available() else 0,
        "cuda_device_name": torch_module.cuda.get_device_name(0) if torch_module.cuda.is_available() else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return int(args.func(args))
