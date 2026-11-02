"""Per-interpreter CLI defaults registered by graphical and other hosts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .context import PluginContext, environment_identity, python_identity


@dataclass(frozen=True)
class EnvironmentBinding:
    python_path: Path
    binding_dir: Path
    core_path: Path | None = None
    import_paths: tuple[Path, ...] = ()


def _record_path(executable: str | Path) -> Path:
    configured = os.environ.get("PYMSS_PLUGIN_ENVIRONMENTS_DIR")
    root = Path(configured).expanduser().absolute() if configured else Path.home() / ".pymss" / "environments"
    key = hashlib.sha256(python_identity(executable).encode()).hexdigest()
    return root / f"{key}.json"


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError(f"Invalid environment {name}")
    return Path(value)


def _decode(value: object, executable: str | Path) -> EnvironmentBinding:
    if not isinstance(value, dict) or type(value.get("schemaVersion")) is not int or value["schemaVersion"] != 1:
        raise ValueError("Unsupported environment binding record")
    python = _absolute_path(value.get("pythonPath"), "Python path")
    if python_identity(python) != python_identity(executable):
        raise ValueError("Environment binding belongs to a different interpreter")
    if python_identity(executable) == python_identity(sys.executable) and value.get("environment") != environment_identity():
        raise ValueError("The Python environment changed; refresh it in its host before using its CLI defaults")
    binding = PluginContext(_absolute_path(value.get("bindingDir"), "plugin binding")).root
    core = value.get("corePath")
    if core is not None:
        core = _absolute_path(core, "core path").resolve()
        if not all((core / package / "__init__.py").is_file() for package in ("pymss", "pymss_plugin_manager")):
            raise ValueError("The registered pymss core is unavailable; refresh the environment in its host")
    imports = value.get("importPaths", [])
    if not isinstance(imports, list) or len(imports) > 32:
        raise ValueError("Environment import paths must be a bounded list")
    paths = tuple(dict.fromkeys(_absolute_path(path, "import path") for path in imports))
    return EnvironmentBinding(python, binding, core, tuple(path for path in paths if path.is_dir()))


def environment_binding(executable: str | Path | None = None) -> EnvironmentBinding | None:
    executable = sys.executable if executable is None else executable
    path = _record_path(executable)
    try:
        if path.is_symlink():
            raise ValueError("Environment binding record must be a regular file")
        with path.open("rb") as stream:
            content = stream.read(16385)
    except FileNotFoundError:
        return None
    if len(content) > 16384:
        raise ValueError("Environment binding record exceeds its size limit")
    return _decode(json.loads(content), executable)


def register_environment_binding(
    binding_dir: str | Path, *, core_path: str | Path | None = None,
    import_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> EnvironmentBinding:
    """Atomically register defaults for the interpreter running this function."""
    executable = sys.executable
    binding = PluginContext(binding_dir).root
    if core_path is not None and not str(core_path).strip():
        raise ValueError("Environment core path must not be empty")
    if not isinstance(import_paths, (list, tuple)) or len(import_paths) > 32:
        raise ValueError("Environment import paths must be a bounded list")
    if any(not isinstance(path, (str, Path)) or not str(path).strip() for path in import_paths):
        raise ValueError("Environment import paths must not be empty")
    value = {
        "schemaVersion": 1,
        "pythonPath": python_identity(executable),
        "bindingDir": str(binding),
        "corePath": str(Path(core_path).expanduser().resolve()) if core_path is not None else None,
        "environment": environment_identity(),
        "importPaths": [str(Path(path).expanduser().resolve()) for path in import_paths],
    }
    result = _decode(value, executable)
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if len(content) > 16384:
        raise ValueError("Environment binding record exceeds its size limit")
    path = _record_path(executable)
    if path.is_file() and not path.is_symlink() and path.stat().st_size <= 16384 and path.read_bytes() == content:
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".environment-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result
