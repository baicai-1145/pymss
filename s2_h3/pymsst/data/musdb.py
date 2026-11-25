from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_STEMS = ("mixture", "vocals", "drums", "bass", "other")
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac")


@dataclass(frozen=True)
class MusdbTrack:
    name: str
    path: Path
    files: dict[str, Path]


@dataclass(frozen=True)
class IncompleteTrack:
    name: str
    path: Path
    missing_files: tuple[str, ...]


@dataclass(frozen=True)
class MusdbSplitScan:
    split: str
    path: Path
    tracks: tuple[MusdbTrack, ...]
    incomplete: tuple[IncompleteTrack, ...]

    @property
    def ok(self) -> bool:
        return not self.incomplete


def scan_musdb_root(
    root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "test"),
    stems: tuple[str, ...] = DEFAULT_STEMS,
    audio_extensions: tuple[str, ...] = DEFAULT_AUDIO_EXTENSIONS,
) -> dict[str, MusdbSplitScan]:
    root_path = Path(root)
    return {
        split: scan_musdb_split(root_path, split, stems=stems, audio_extensions=audio_extensions)
        for split in splits
    }


def scan_musdb_split(
    root: str | Path,
    split: str,
    *,
    stems: tuple[str, ...] = DEFAULT_STEMS,
    audio_extensions: tuple[str, ...] = DEFAULT_AUDIO_EXTENSIONS,
) -> MusdbSplitScan:
    split_path = Path(root) / split
    if not split_path.exists():
        return MusdbSplitScan(split=split, path=split_path, tracks=(), incomplete=())

    tracks: list[MusdbTrack] = []
    incomplete: list[IncompleteTrack] = []

    for track_path in sorted(item for item in split_path.iterdir() if item.is_dir()):
        files: dict[str, Path] = {}
        missing: list[str] = []

        for stem in stems:
            stem_file = _find_stem_file(track_path, stem, audio_extensions)
            if stem_file is None:
                missing.append(_format_stem_name(stem, audio_extensions))
            else:
                files[stem] = stem_file

        if missing:
            incomplete.append(IncompleteTrack(name=track_path.name, path=track_path, missing_files=tuple(missing)))
        else:
            tracks.append(MusdbTrack(name=track_path.name, path=track_path, files=files))

    return MusdbSplitScan(split=split, path=split_path, tracks=tuple(tracks), incomplete=tuple(incomplete))


def _find_stem_file(track_path: Path, stem: str, audio_extensions: tuple[str, ...]) -> Path | None:
    for extension in audio_extensions:
        candidate = track_path / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _format_stem_name(stem: str, audio_extensions: tuple[str, ...]) -> str:
    if len(audio_extensions) == 1:
        return f"{stem}{audio_extensions[0]}"
    joined = "|".join(audio_extensions)
    return f"{stem}({joined})"
