from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import os
import platform
import sys
from typing import Callable, Iterator, Any


def python_identity(executable: str | Path) -> str:
    """Normalize path spelling while retaining POSIX virtual environment links."""
    if not str(executable).strip():
        raise ValueError("Environment Python path must not be empty")
    value = os.path.abspath(os.path.expanduser(str(executable)))
    if os.name == "nt":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return os.path.normcase(value)


def environment_identity() -> dict:
    return {"python": python_identity(sys.executable), "prefix": python_identity(sys.prefix),
            "pythonVersion": list(sys.version_info[:2]), "platform": sys.platform,
            "architecture": platform.machine().lower()}


@dataclass(frozen=True)
class PluginContext:
    """An explicit runtime binding; sibling bindings share installation intent."""

    root: Path
    registry_url: str | None = None
    local_source: Path | None = None
    bundled_root: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.root, str) and not self.root.strip():
            raise ValueError("Plugin binding path must not be empty")
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if self.registry_url is not None:
            if not isinstance(self.registry_url, str):
                raise ValueError("Plugin registry URL must be a string")
            object.__setattr__(self, "registry_url", self.registry_url.strip() or None)
        from .inventory import valid_binding
        if not valid_binding(self.root.name):
            raise ValueError("Invalid plugin environment binding name")
        for name in ("local_source", "bundled_root"):
            if (value := getattr(self, name)) is not None:
                object.__setattr__(self, name, Path(value).absolute())


@dataclass
class PluginPolicy:
    """Host environment facts and adapters, independent of a graphical host."""

    event: Callable[[str, dict, str | None], None] = lambda event, payload, task_id: None
    cancelled: Callable[[], bool] = lambda: False
    validate: Callable[[Path], None] = lambda plugin: None
    probe: Callable[[Path, list[Path] | None], list[dict]] | None = None
    constraints: Callable[[Path], list[str]] = lambda plugin: []
    index_args: Callable[[dict], list[str]] = lambda payload: []
    deferred_operations: Callable[[Path], set[str]] = lambda plugin: set()
    primary_operation: Callable[[Path], str | None] = lambda plugin: None
    validate_requirements: Callable[..., None] | None = None
    component_paths: Callable[[Path], list[Path]] = lambda plugin: []


_CURRENT: ContextVar[tuple[PluginContext, PluginPolicy] | None] = ContextVar("pymss_plugin_management", default=None)


def current() -> tuple[PluginContext, PluginPolicy]:
    value = _CURRENT.get()
    if value is None:
        raise RuntimeError("Plugin management requires an explicit context")
    return value


@contextmanager
def use(context: PluginContext, policy: PluginPolicy) -> Iterator[None]:
    token = _CURRENT.set((context, policy))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def check_cancelled() -> None:
    if current()[1].cancelled():
        raise InterruptedError("Plugin management was cancelled")


def emit(event: str, payload: dict[str, Any], *, task_id: str | None = None) -> None:
    current()[1].event(event, payload, task_id)
