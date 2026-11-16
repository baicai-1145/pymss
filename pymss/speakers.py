"""Full-clip, variable-count speaker separation without fixed instrument stems."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpeakerSeparation:
    tracks: np.ndarray                 # [speakers, channels, samples], at model sample rate
    sample_rate: int
    limit_reached: bool

    @property
    def count(self):
        return self.tracks.shape[0]


class SpeakerSeparator:
    def __init__(self, checkpoint, config, *, device="auto"):
        import torch
        from pymss_core import get_model_from_config
        from pymss_core.modules.bs_roformer.adaptive import AdaptiveRoformer

        self.model, self.config = get_model_from_config("auto", config)
        if not isinstance(self.model, AdaptiveRoformer):
            raise TypeError("speaker separation requires an adaptive BS/Mel-band RoFormer")
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved_config = package.get("config")
        if saved_config and (saved_config["model"] != dict(self.config.model) or saved_config["audio"] != dict(self.config.audio)):
            raise ValueError("checkpoint model/audio configuration does not match the supplied config")
        self.model.load_state_dict(package["state_dict"], strict=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.sample_rate = int(self.config.audio.sample_rate)

    def separate(self, audio, *, speaker_limit=None, threshold=0.5):
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None]
        if audio.ndim != 2 or audio.shape[0] != self.model.audio_channels or audio.shape[1] == 0:
            raise ValueError(f"audio must be [{self.model.audio_channels}, samples] at {self.sample_rate} Hz")
        if not np.isfinite(audio).all():
            raise ValueError("audio contains nonfinite samples")
        with torch.inference_mode():
            output = self.model(torch.from_numpy(np.ascontiguousarray(audio))[None].to(self.device),
                                stop_early=True, speaker_limit=speaker_limit, threshold=threshold)
        count = int(output.counts(threshold)[0])
        return SpeakerSeparation(output.audio[0, :count].cpu().numpy(), self.sample_rate,
                                 bool(output.at_limit(threshold)[0]))

    def separate_file(self, path, output_dir, *, speaker_limit=None, threshold=0.5):
        from .audio_io import load_audio, save_audio

        output_dir = Path(output_dir)
        if (output_dir / "separation.json").exists() or any(output_dir.glob("speaker_*.wav")):
            raise FileExistsError("output directory already contains a speaker separation; choose a new directory")
        audio, _ = load_audio(path, sr=self.sample_rate, mono=self.model.audio_channels == 1)
        result = self.separate(audio, speaker_limit=speaker_limit, threshold=threshold)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, track in enumerate(result.tracks, 1):
            destination = output_dir / f"speaker_{index:03d}.wav"
            save_audio(destination, track.T, self.sample_rate, "wav", {"wav_bit_depth": "FLOAT"})
            paths.append(str(destination))
        metadata = {"input": str(path), "sample_rate": self.sample_rate, "speakers": result.count,
                        "samples": result.tracks.shape[-1], "limit_reached": result.limit_reached, "tracks": paths}
        (output_dir / "separation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata


def add_arguments(parser):
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--speaker-limit", type=int, help="Safety limit; exceeding the trained range needs evaluation.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.set_defaults(func=run)


def run(args):
    separator = SpeakerSeparator(args.checkpoint, args.config, device=args.device)
    result = separator.separate_file(args.input, args.output_dir, speaker_limit=args.speaker_limit, threshold=args.threshold)
    print(json.dumps(result, indent=2))
    if result["limit_reached"]:
        import sys
        print("Speaker decoding reached its safety limit while still predicting another speaker.", file=sys.stderr)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
