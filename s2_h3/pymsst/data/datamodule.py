from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import read_wav_chunk, read_wav_info
from .musdb import MusdbTrack, scan_musdb_split


@dataclass(frozen=True)
class MusdbTrackInfo:
    track: MusdbTrack
    frames: int


class MusdbChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        tracks: list[MusdbTrackInfo],
        *,
        instruments: tuple[str, ...],
        chunk_size: int,
        sample_rate: int,
        channels: int,
        target_instrument: str | None = None,
        examples: int | None = None,
        min_mean_abs: float = 0.0,
        random_chunks: bool = True,
        mix_from_stems: bool = False,
        loudness_augmentation: bool = False,
        loudness_min: float = 0.5,
        loudness_max: float = 1.5,
        keep_target_stem_dim: bool = False,
        max_attempts: int = 10,
    ) -> None:
        if not tracks:
            raise ValueError("tracks must not be empty")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if target_instrument is not None and target_instrument not in instruments:
            raise ValueError(f"target_instrument {target_instrument!r} is not in instruments")

        self.tracks = tracks
        self.instruments = instruments
        self.chunk_size = int(chunk_size)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.target_instrument = target_instrument
        self.examples = int(examples) if examples is not None else len(tracks)
        self.min_mean_abs = float(min_mean_abs)
        self.random_chunks = random_chunks
        self.mix_from_stems = mix_from_stems
        self.loudness_augmentation = loudness_augmentation
        self.loudness_min = float(loudness_min)
        self.loudness_max = float(loudness_max)
        self.keep_target_stem_dim = keep_target_stem_dim
        self.max_attempts = max(1, int(max_attempts))

    def __len__(self) -> int:
        return self.examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mix_from_stems:
            return self._load_random_stem_mix()

        for _ in range(self.max_attempts):
            track_info = self._select_track(index)
            offset = self._select_offset(track_info, index)
            target, mixture = self._load_example(track_info.track, offset)
            if self.min_mean_abs <= 0 or target.abs().mean().item() >= self.min_mean_abs:
                return target, mixture

        return target, mixture

    def _select_track(self, index: int) -> MusdbTrackInfo:
        if self.random_chunks:
            selected = torch.randint(0, len(self.tracks), ()).item()
            return self.tracks[selected]
        return self.tracks[index % len(self.tracks)]

    def _select_offset(self, track_info: MusdbTrackInfo, index: int) -> int:
        max_offset = max(0, track_info.frames - self.chunk_size)
        if max_offset == 0:
            return 0
        if self.random_chunks:
            return int(torch.randint(0, max_offset + 1, ()).item())
        return int((index * self.chunk_size) % (max_offset + 1))

    def _load_example(self, track: MusdbTrack, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        mixture = torch.from_numpy(
            read_wav_chunk(
                track.files["mixture"],
                start=offset,
                frames=self.chunk_size,
                target_channels=self.channels,
            )
        )

        if self.target_instrument is not None:
            target = torch.from_numpy(
                read_wav_chunk(
                    track.files[self.target_instrument],
                    start=offset,
                    frames=self.chunk_size,
                    target_channels=self.channels,
                )
            )
            if self.keep_target_stem_dim:
                target = target.unsqueeze(0)
            return target, mixture

        sources = [
            torch.from_numpy(
                read_wav_chunk(
                    track.files[instrument],
                    start=offset,
                    frames=self.chunk_size,
                    target_channels=self.channels,
                )
            )
            for instrument in self.instruments
        ]
        return torch.stack(sources, dim=0), mixture

    def _load_random_stem_mix(self) -> tuple[torch.Tensor, torch.Tensor]:
        sources = [self._load_random_source(instrument) for instrument in self.instruments]
        stacked = torch.stack(sources, dim=0)
        if self.loudness_augmentation:
            gains = np.random.uniform(self.loudness_min, self.loudness_max, size=(len(sources),))
            stacked = stacked * torch.tensor(gains, dtype=torch.float32)[:, None, None]
        mixture = stacked.sum(dim=0)
        if self.target_instrument is not None:
            index = self.instruments.index(self.target_instrument)
            target = stacked[index : index + 1] if self.keep_target_stem_dim else stacked[index]
            return target, mixture
        return stacked, mixture

    def _load_random_source(self, instrument: str) -> torch.Tensor:
        for _ in range(self.max_attempts):
            track_info = random.choice(self.tracks)
            offset = _random_offset(track_info.frames, self.chunk_size)
            source = torch.from_numpy(
                read_wav_chunk(
                    track_info.track.files[instrument],
                    start=offset,
                    frames=self.chunk_size,
                    target_channels=self.channels,
                )
            )
            if self.min_mean_abs <= 0 or source.abs().mean().item() >= self.min_mean_abs:
                return source
        return source


class MusdbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        *,
        instruments: tuple[str, ...],
        target_instrument: str | None,
        chunk_size: int,
        sample_rate: int,
        channels: int,
        batch_size: int,
        steps_per_epoch: int,
        val_fraction: float = 0.1,
        val_tracks: int | None = None,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        min_mean_abs: float = 0.0,
        train_split: str = "train",
        mix_from_stems: bool = True,
        loudness_augmentation: bool = False,
        loudness_min: float = 0.5,
        loudness_max: float = 1.5,
        keep_target_stem_dim: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.instruments = instruments
        self.target_instrument = target_instrument
        self.chunk_size = int(chunk_size)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.val_fraction = float(val_fraction)
        self.val_tracks = val_tracks
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and self.num_workers > 0
        self.min_mean_abs = float(min_mean_abs)
        self.train_split = train_split
        self.mix_from_stems = mix_from_stems
        self.loudness_augmentation = loudness_augmentation
        self.loudness_min = float(loudness_min)
        self.loudness_max = float(loudness_max)
        self.keep_target_stem_dim = keep_target_stem_dim

        self.train_dataset: MusdbChunkDataset | None = None
        self.val_dataset: MusdbChunkDataset | None = None
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return

        required_stems = ("mixture", *self.instruments)
        scan = scan_musdb_split(self.root, self.train_split, stems=required_stems, audio_extensions=(".wav",))
        if not scan.tracks:
            raise ValueError(f"no complete MUSDB tracks found in {scan.path}")
        track_infos = [_track_info(track, self.sample_rate) for track in scan.tracks]
        train_tracks, val_tracks = _split_tracks(track_infos, val_fraction=self.val_fraction, val_tracks=self.val_tracks, seed=self.seed)

        examples = max(1, self.steps_per_epoch * self.batch_size)
        self.train_dataset = MusdbChunkDataset(
            train_tracks,
            instruments=self.instruments,
            target_instrument=self.target_instrument,
            chunk_size=self.chunk_size,
            sample_rate=self.sample_rate,
            channels=self.channels,
            examples=examples,
            min_mean_abs=self.min_mean_abs,
            random_chunks=True,
            mix_from_stems=self.mix_from_stems,
            loudness_augmentation=self.loudness_augmentation,
            loudness_min=self.loudness_min,
            loudness_max=self.loudness_max,
            keep_target_stem_dim=self.keep_target_stem_dim,
        )
        if val_tracks:
            self.val_dataset = MusdbChunkDataset(
                val_tracks,
                instruments=self.instruments,
                target_instrument=self.target_instrument,
                chunk_size=self.chunk_size,
                sample_rate=self.sample_rate,
                channels=self.channels,
                examples=len(val_tracks),
                min_mean_abs=0.0,
                random_chunks=False,
                keep_target_stem_dim=self.keep_target_stem_dim,
            )
        self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        if self.val_dataset is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )


def _track_info(track: MusdbTrack, sample_rate: int) -> MusdbTrackInfo:
    mixture_info = read_wav_info(track.files["mixture"])
    if mixture_info.sample_rate != sample_rate:
        raise ValueError(f"{mixture_info.path} has sample_rate={mixture_info.sample_rate}, expected {sample_rate}")
    return MusdbTrackInfo(track=track, frames=mixture_info.frames)


def _split_tracks(
    tracks: list[MusdbTrackInfo],
    *,
    val_fraction: float,
    val_tracks: int | None,
    seed: int,
) -> tuple[list[MusdbTrackInfo], list[MusdbTrackInfo]]:
    indices = list(range(len(tracks)))
    random.Random(seed).shuffle(indices)
    if val_tracks is not None and val_tracks <= 0:
        return tracks, []
    if len(tracks) < 2:
        raise ValueError("at least two complete tracks are required for validation; set val_tracks=0 to disable it")
    if val_tracks is None:
        val_count = max(1, round(len(tracks) * val_fraction))
    else:
        val_count = int(val_tracks)
    val_count = min(max(1, val_count), len(tracks) - 1)

    val_index_set = set(indices[:val_count])
    train = [track for index, track in enumerate(tracks) if index not in val_index_set]
    val = [track for index, track in enumerate(tracks) if index in val_index_set]
    return train, val


def _random_offset(frames: int, chunk_size: int) -> int:
    max_offset = max(0, frames - chunk_size)
    if max_offset == 0:
        return 0
    return int(np.random.randint(max_offset + 1))
