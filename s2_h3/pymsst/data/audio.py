from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    sample_width: int


def read_wav_info(path: str | Path) -> WavInfo:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as handle:
        return WavInfo(
            path=wav_path,
            sample_rate=handle.getframerate(),
            channels=handle.getnchannels(),
            frames=handle.getnframes(),
            sample_width=handle.getsampwidth(),
        )


def read_wav_chunk(
    path: str | Path,
    *,
    start: int,
    frames: int,
    target_channels: int,
) -> np.ndarray:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if target_channels <= 0:
        raise ValueError("target_channels must be positive")

    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        total_frames = handle.getnframes()
        start = max(0, min(int(start), total_frames))
        handle.setpos(start)
        raw = handle.readframes(frames)

    audio = _decode_pcm(raw, sample_width, channels)
    audio = _match_channels(audio, target_channels)

    if audio.shape[0] < frames:
        padding = np.zeros((frames - audio.shape[0], target_channels), dtype=np.float32)
        audio = np.concatenate([audio, padding], axis=0)
    elif audio.shape[0] > frames:
        audio = audio[:frames]

    return audio.T.astype(np.float32, copy=False)


def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        data = _decode_int24(raw).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")

    if data.size == 0:
        return np.zeros((0, channels), dtype=np.float32)
    if data.size % channels != 0:
        raise ValueError("wav frame data is not aligned to the channel count")
    return data.reshape(-1, channels)


def _decode_int24(raw: bytes) -> np.ndarray:
    bytes_ = np.frombuffer(raw, dtype=np.uint8)
    if bytes_.size % 3 != 0:
        raise ValueError("24-bit PCM data is not aligned to 3-byte samples")
    triples = bytes_.reshape(-1, 3).astype(np.int32)
    values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    sign_bit = 1 << 23
    return (values ^ sign_bit) - sign_bit


def _match_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    channels = audio.shape[1]
    if channels == target_channels:
        return audio
    if channels > target_channels:
        return audio[:, :target_channels]
    if channels == 1:
        return np.repeat(audio, target_channels, axis=1)
    raise ValueError(f"cannot convert {channels} channels to {target_channels} channels")
