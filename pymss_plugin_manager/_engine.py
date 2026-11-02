# Migrated from pymss-studio with MIT permission. See NOTICE and LICENSE.
"""Shared installation transactions migrated from the Studio installer."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
import time
import tomllib
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stdout
from pathlib import Path
from typing import Any, Iterator

from .context import current as management_context, check_cancelled, emit
from .locks import RuntimeLockBusy, runtime_lock
from . import inventory as application_inventory
from . import profiles as python_profiles

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)*\Z")
_DEPENDENCY_HANDOFF = ".studio-deps-pending"
_SOURCE_IGNORED_DIRS = {".studio-deps", ".pymss-python.json", "__pycache__", ".git", "node_modules", ".venv", "venv", ".tmp", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


class _ContextValue:
    def __init__(self, name):
        self.name = name

    def get(self):
        return getattr(management_context()[0], self.name)


_LOCAL_SOURCE = _ContextValue("local_source")
_REGISTRY_URL = _ContextValue("registry_url")


def _root():
    return management_context()[0].root


@contextmanager
def _plugin_registry():
    yield


@contextmanager
def isolate_protocol_stdout():
    with redirect_stdout(sys.stderr):
        yield


def _deferred_dependency_operations(plugin):
    return management_context()[1].deferred_operations(plugin)


def _primary_operation(plugin):
    return management_context()[1].primary_operation(plugin)


def _validate_requirements(plugin, *, operation=None):
    policy = management_context()[1]
    if policy.validate_requirements is not None:
        return policy.validate_requirements(plugin, operation=operation)
    from packaging.requirements import Requirement
    if operation in policy.deferred_operations(plugin):
        return
    paths = list(dict.fromkeys([*(str(p) for p in policy.component_paths(plugin)), *sys.path, str(plugin / ".studio-deps")]))
    for text in _plugin_requirements(plugin):
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        distribution = next(importlib.metadata.distributions(name=requirement.name, path=paths), None)
        if distribution is None:
            raise ValueError(f"Missing plugin dependency: {requirement.name}")
        if requirement.specifier and distribution.version not in requirement.specifier:
            raise ValueError(f"Plugin requires {requirement}; current runtime provides {distribution.version}")


def _validate(plugin):
    _plugin_requirements(plugin)
    from pymss.plugins.manifest import read_manifest
    from pymss.plugins.lifecycle import pymss_package_version, satisfies
    manifest = read_manifest(plugin)
    version = pymss_package_version()
    if manifest.requires_pymss and version and not satisfies(version, manifest.requires_pymss):
        raise ValueError(f"Plugin requires pymss {manifest.requires_pymss}; current: {version}")
    management_context()[1].validate(plugin)


def _local_catalog(root):
    registry = root / "registry.json"
    if registry.is_symlink() or not registry.is_file():
        raise ValueError("Local plugin registry is unavailable or is a symbolic link")
    data = json.loads(registry.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Local plugin registry must be an object")
    rows = []
    for name, entry in data.items():
        if name.startswith("_"):
            continue
        _name(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid local plugin registry entry: {name}")
        _local_package(root, name, entry.get("subpath"))
        rows.append({**entry, "name": name, "url": str(root)})
    return rows


def _recover_when_idle(root):
    # Callers already own the v1 writer lock. Execution readers may keep using
    # their committed snapshot while an update check or source acquisition runs.
    try:
        from .leases import binding_lease
        with binding_lease(root, exclusive=True):
            _recover(root)
    except RuntimeLockBusy:
        pass


def _probe(root, dependency_paths=None):
    policy = management_context()[1]
    if policy.probe is not None:
        return policy.probe(root, dependency_paths)
    from .service import probe
    return probe(root, dependency_paths, component_paths=policy.component_paths,
                 deferred_operations=policy.deferred_operations, cancelled=policy.cancelled)


def _name(value: Any) -> str:
    """Validate a plugin id; it is also the installation directory name."""
    name = str(value or "").strip()
    if len(name) > 100 or not _NAME.fullmatch(name):
        raise ValueError("Invalid plugin id")
    return name


def _read_manifest(plugin: Path) -> dict[str, Any]:
    """Read v2 identity and opaque host metadata without importing plugin code.

    pymss validates the complete manifest when the plugin is verified and loaded.
    """
    path = plugin / "pyproject.toml"
    if not path.is_file():
        raise ValueError("Plugin has no pyproject.toml with [tool.pymss.plugin]")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    tool = data.get("tool", {})
    if not isinstance(tool, dict) or not isinstance(tool.get("pymss", {}), dict):
        raise ValueError("Plugin tool metadata must be a table")
    pymss = tool.get("pymss", {})
    plugin_table = pymss.get("plugin")
    if not isinstance(plugin_table, dict):
        raise ValueError("Plugin has no [tool.pymss.plugin] manifest")
    if plugin_table.get("api_version") != 2:
        raise ValueError(f"Plugin uses api_version {plugin_table.get('api_version')!r}; this Studio requires 2")
    plugin_id = _name(plugin_table.get("id"))
    operations = plugin_table.get("operations", [])
    if not isinstance(operations, list) or not all(isinstance(item, dict) and isinstance(item.get("id"), str) for item in operations):
        raise ValueError("Plugin operations must be an array of tables with ids")
    studio = pymss.get("studio", {})
    project = data.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("Plugin project metadata must be a table")
    return {
        "id": plugin_id,
        "version": project.get("version") if isinstance(project.get("version"), str) else None,
        "description": project.get("description", "") if isinstance(project.get("description"), str) else "",
        "operations": [{"id": item["id"], "title": str(item.get("title") or item["id"])} for item in operations],
        "studio": studio if isinstance(studio, dict) else {},
    }


def _local_package(root: Path, name: str | None, subpath: object) -> Path:
    root = root.resolve()
    if not isinstance(subpath, str) or not subpath or "\\" in subpath or ":" in subpath:
        raise ValueError(f"Invalid local plugin registry subpath: {name}")
    relative = Path(subpath)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root):
        raise ValueError(f"Local plugin registry entry escapes its source root: {name}")
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Local plugin registry cannot contain symbolic links: {name}")
        current = current.parent
    if not path.is_dir():
        raise ValueError(f"Local registry entry does not select a plugin package: {name}")
    if name is not None and _read_manifest(path)["id"] != name:
        raise ValueError(f"Local registry entry does not select its matching plugin package: {name}")
    return path


def _uses_default_source(meta: dict) -> bool:
    return bool(meta.get("preinstalled") or meta.get("defaultRepository") or meta.get("source") == "registry")


def _migrate_v1(root: Path) -> None:
    """Move installations that are not v2 plugins named by their id out of the load path."""
    legacy = root / "legacy-v1"
    for store in ("installed", "disabled"):
        directory = root / store
        if not directory.is_dir():
            continue
        for plugin in directory.iterdir():
            if not plugin.is_dir() or plugin.name.startswith("."):
                continue
            try:
                current = _read_manifest(plugin)["id"] == plugin.name
            except Exception:
                current = False
            if current:
                continue
            target = legacy / plugin.name
            suffix = 2
            while target.exists() or target.is_symlink():
                target = legacy / f"{plugin.name}-{suffix}"
                suffix += 1
            legacy.mkdir(parents=True, exist_ok=True)
            plugin.rename(target)


def _legacy_plugins(root: Path) -> list[str]:
    legacy = root / "legacy-v1"
    return sorted(path.name for path in legacy.iterdir() if path.is_dir()) if legacy.is_dir() else []


def _recover(root: Path) -> None:
    for store, backup_store in (("installed", "backups"), ("disabled", "disabled-backups")):
        backups = root / backup_store
        backups.mkdir(parents=True, exist_ok=True)
        for backup in backups.iterdir():
            if not backup.is_dir() or not _NAME.fullmatch(backup.name):
                continue
            target = root / store / backup.name
            target.parent.mkdir(parents=True, exist_ok=True)
            opposite = root / ("disabled" if store == "installed" else "installed") / backup.name
            if not target.exists() and opposite.exists():
                previous = _manifest(backup).get("inventory", {}).get("revision", 0)
                replacement = _manifest(opposite).get("inventory", {}).get("revision", 0)
                if type(previous) is int and type(replacement) is int and replacement > previous:
                    _finish_dependency_handoff(opposite, backup)
                    shutil.rmtree(backup)
                    continue
            if target.exists():
                _finish_dependency_handoff(target, backup)
                shutil.rmtree(backup)
            else:
                backup.rename(target)
    transactions = root / "transactions"
    if transactions.exists():
        for transaction in transactions.iterdir():
            if transaction.is_dir() and transaction.name.startswith(("install-", "removed-")):
                try:
                    with runtime_lock(transaction / ".lease", timeout=0):
                        pass
                except RuntimeLockBusy:
                    continue
                shutil.rmtree(transaction, ignore_errors=True)
    _complete_recovered_uninstalls(root)
    _migrate_v1(root)


def _complete_recovered_uninstalls(root: Path) -> None:
    """Finish the matching intent only after this runtime's files are gone."""
    try:
        entries = application_inventory.merge(root, [], set())["plugins"]
    except (OSError, ValueError, TypeError, KeyError):
        # Invalid inventory must retain its existing diagnostics and user choices.
        return
    for name, entry in entries.items():
        if entry["state"] != "removed" or entry.get("pending") != {"action": "uninstall", "binding": root.name}:
            continue
        copies = [root / store / name for store in ("installed", "disabled", "backups", "disabled-backups")]
        copies.append(root / "transactions" / f"removed-{name}")
        if any(path.exists() or path.is_symlink() for path in copies):
            continue
        # complete() checks the revision under the inventory lock, so cleanup
        # cannot clear a newer action committed in another runtime meanwhile.
        application_inventory.complete(root, name, entry["revision"])


def _recover_for_read(root: Path) -> None:
    if not root.exists():
        return
    try:
        from .leases import binding_lease
        with binding_lease(root, exclusive=True), runtime_lock(root / ".lock", timeout=0, allow_read_only=True) as writable:
            if writable:
                _recover(root)
    except (RuntimeLockBusy, OSError):
        # A live writer owns its transaction; readers must never clean it up.
        # Read-only installations can still be inspected and executed.
        pass


def _installed(root: Path, *, disabled: bool = False) -> list[Path]:
    directory = root / ("disabled" if disabled else "installed")
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_dir() and _NAME.fullmatch(p.name))


def _location(root: Path, name: str) -> Path:
    active, inactive = root / "installed" / name, root / "disabled" / name
    if active.exists() and inactive.exists():
        raise ValueError(f"Plugin has conflicting active and disabled installations: {name}")
    return inactive if inactive.exists() else active


def _bundled() -> dict[str, dict]:
    if management_context()[0].bundled_root is None:
        return {}
    manifest = management_context()[0].bundled_root / "manifest.json"
    if not manifest.is_file():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != 1 or not isinstance(data.get("plugins"), list):
        raise ValueError("Invalid preinstalled plugin manifest")
    rows = {}
    for row in data["plugins"]:
        name = _name(row["name"])
        if name in rows or not isinstance(row.get("files"), dict):
            raise ValueError("Invalid preinstalled plugin entry")
        for relative in row["files"]:
            path = Path(relative)
            if path.is_absolute() or path.drive or "\\" in relative or ".." in path.parts:
                raise ValueError("Invalid preinstalled plugin file path")
        rows[name] = row
    return rows


def _removed_preinstalled(root: Path) -> set[str]:
    path = root / "preinstalled-removed.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(name, str) for name in data):
        raise ValueError("Invalid preinstalled plugin preferences")
    return {_name(name) for name in data}


def _mark_preinstalled_removed(root: Path, name: str, removed: bool) -> None:
    names = _removed_preinstalled(root)
    if removed:
        names.add(name)
    else:
        names.discard(name)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix=".plugin-state-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(sorted(names), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "preinstalled-removed.json")
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def _intent_definition(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key in {"id", "version", "description", "operations", "studio"}}


def _application_bindings(root: Path) -> tuple[list[dict], set[str], list[str]]:
    roots = [root]
    warnings = []
    if root.parent.is_dir():
        roots.extend(path for path in sorted(root.parent.iterdir())
                     if path != root and path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
                     and any((path / name).exists() for name in ("installed", "disabled", "preinstalled-removed.json")))
    observations, removed = [], set()
    for binding in roots:
        if not application_inventory.valid_binding(binding.name):
            warnings.append("A plugin runtime directory has an unsupported name and was skipped")
            continue
        try:
            removed.update(_removed_preinstalled(binding))
        except (OSError, ValueError, TypeError):
            warnings.append(f"Plugin removal preferences are unavailable for runtime {binding.name}")
        for enabled, directory in ((True, binding / "installed"), (False, binding / "disabled")):
            try:
                children = sorted(directory.iterdir()) if directory.is_dir() else []
            except OSError:
                warnings.append(f"Plugin records are unavailable for runtime {binding.name}")
                continue
            for package in children:
                if not package.is_dir() or package.is_symlink() or package.name.startswith((".", "_")):
                    continue
                try:
                    manifest, meta = _read_manifest(package), _manifest(package)
                    if manifest["id"] != package.name:
                        continue
                    content = meta.get("inventoryContent")
                    observations.append({
                        "name": manifest["id"], "binding": binding.name, "path": str(package),
                        "enabled": enabled, "version": manifest["version"], "definition": _intent_definition(manifest),
                        "marker": meta.get("inventory", {}), "meta": meta,
                        "content": content if isinstance(content, str) and re.fullmatch(r"[0-9a-f]{64}", content) else None,
                    })
                except (OSError, ValueError, TypeError, KeyError):
                    continue
    return observations, removed, warnings


def _application_state(root: Path) -> tuple[dict, list[dict]]:
    observations, removed, warnings = _application_bindings(root)
    state = application_inventory.merge(root, observations, removed)
    state["migrationWarnings"] = warnings
    return state, observations


def _execution_intents(root: Path) -> dict | None:
    if root != _root():
        return {}
    try:
        return _application_state(root)[0]["plugins"]
    except (OSError, ValueError, TypeError, KeyError):
        # Inventory corruption disables managed code; built-in operations remain usable.
        return None


def _binding_sources(observations: list[dict], name: str) -> list[dict]:
    sources = [item for item in observations if item["name"] == name]
    if len(sources) > 1:
        sources = [{**item, "content": item.get("content") or _plugin_content_digest(Path(item["path"]))}
                   for item in sources]
    return sources


def _binding_matches(package: Path, intent: dict) -> bool:
    try:
        manifest, meta = _read_manifest(package), _manifest(package)
        if manifest["id"] != package.name:
            return False
        if intent.get("revision", 0) == 0:
            return True
        return ((intent.get("version") is None or manifest["version"] == intent["version"])
                and (intent.get("content") is None or meta.get("inventoryContent") == intent["content"]))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _has_current_installation(root: Path, name: str) -> bool:
    intent = _application_state(root)[0]["plugins"].get(name, {})
    package = _location(root, name)
    return package.exists() and intent.get("state") != "removed" and _binding_matches(package, intent)


def _prepare_binding(root: Path, payload: dict, task_id: str | None, revision: int) -> None:
    name = _name(payload.get("name"))
    state, observations = _application_state(root)
    entry = state["plugins"].get(name)
    if entry is None:
        raise ValueError("Plugin is not installed")
    if entry["state"] == "removed":
        raise ValueError("Plugin is removed; reinstall it to restore its availability")
    sources = _binding_sources(observations, name)
    requested = payload.get("sourceRuntime")
    if requested is not None and not isinstance(requested, str):
        raise ValueError("Invalid plugin source runtime")
    if requested:
        source = next((item for item in sources if item["binding"] == requested), None)
        if source and "sourceVersion" in payload and payload["sourceVersion"] != source.get("version"):
            raise ValueError("Plugin source changed; refresh its available versions")
    elif entry.get("sourceSelectionRequired"):
        raise ValueError("Select the plugin source version before preparing this runtime")
    elif entry.get("revision", 0) > 0:
        source = next((item for item in sources if item["binding"] == entry.get("binding")
                       and (entry.get("version") is None or item.get("version") == entry["version"])
                       and (entry.get("content") is None or item.get("content") == entry["content"])), None)
        if source is None:
            source = next((item for item in sources if item.get("version") == entry.get("version") and item.get("content") == entry.get("content")), None)
    else:
        if len({(item.get("version"), item.get("content")) for item in sources}) > 1:
            raise ValueError("Select the plugin source version before preparing this runtime")
        source = sources[0] if sources else None
    if source is None:
        raise ValueError("The selected plugin files are unavailable; reinstall the plugin from its original source")
    path = Path(source["path"])
    expected = payload.get("sourceContent") or source.get("content")
    actual = _plugin_content_digest(path)
    if expected is not None and (not isinstance(expected, str) or expected != actual):
        raise ValueError("Plugin source files changed; repair the source installation before preparing another runtime")
    _install(root, {**payload, "action": "install", "name": name}, task_id, intent_revision=revision,
             prepared_binding={**source, "content": actual, "enabled": entry["state"] == "enabled" or payload.get("action") == "enable"})


def _record_intent(root: Path, name: str, state: str, *, pending: str | None = None, revision: int | None = None) -> int:
    observations, removed, _warnings = _application_bindings(root)
    current = application_inventory.merge(root, observations, removed)["plugins"].get(name)
    if current is None:
        raise ValueError("Plugin is not installed")
    if current.get("revision", 0) == 0:
        active = next((item for item in observations if item["name"] == name and item["binding"] == root.name), None)
        if active:
            current = {**current, **active}
        elif len({(item.get("version"), item.get("content")) for item in _binding_sources(observations, name)}) > 1:
            current = {**current, "binding": None, "version": None, "content": None, "sourceSelectionRequired": True}
    if revision is None:
        revision = application_inventory.reserve(root, observations, removed)
    application_inventory.commit(
        root, name, state, revision, version=current.get("version"), content=current.get("content"),
        definition=current.get("definition", {}), binding=current.get("binding"), pending=pending,
        selection_required=current.get("sourceSelectionRequired", False),
    )
    return revision


def _prepare_preinstalled(root: Path, task_id: str | None, requested: str | None = None, *, intent_revision: int | None = None) -> None:
    bundles = _bundled()
    state, _observations = _application_state(root)
    removed = _removed_preinstalled(root) | {name for name, entry in state["plugins"].items() if entry["state"] != "enabled"}
    if requested is not None and requested not in bundles:
        raise ValueError("Unknown preinstalled plugin")
    selected = [requested] if requested is not None else list(bundles)
    for name in selected:
        current = _location(root, name)
        if current.exists():
            if requested:
                _enable_binding(root, name, intent_revision)
            continue
        if requested is None and (name in removed or name in state["plugins"]):
            continue
        row = bundles[name]
        source = management_context()[0].bundled_root / name
        for relative, expected in row["files"].items():
            path = source / relative
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
                raise ValueError(f"Invalid preinstalled plugin file: {name}/{relative}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Preinstalled plugin checksum mismatch: {name}/{relative}")
        transactions = root / "transactions"
        transactions.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=transactions, prefix="install-preinstalled-") as temporary:
            staged = Path(temporary) / name
            staged.mkdir()
            for relative in row["files"]:
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
            if _read_manifest(staged)["id"] != name:
                raise ValueError(f"Preinstalled plugin id does not match its bundle: {name}")
            primary = _primary_operation(staged)
            _validate(staged)
            _validate_requirements(staged, operation=primary)
            (staged / ".pymss-install.json").write_text(json.dumps({
                "name": name, "version": row["version"], "source": "git", "url": row.get("url"),
                "subpath": row.get("subpath"), "ref": None, "preinstalled": True,
            }), encoding="utf-8")
            _stage(task_id, f"Preparing preinstalled plugin: {name}", "preparing", plugin_name=name)
            python_profiles.prepare_python(staged, root, lambda message: _stage(task_id, message, "preparing"))
            _verify_candidate(root, staged, name)
            if requested:
                _mark_preinstalled_removed(root, name, False)
            _publish(staged, root, name, intent_revision=intent_revision)


def _set_enabled(root: Path, name: str, enabled: bool, *, verify: bool = True) -> None:
    source = _location(root, name)
    if not source.is_dir():
        raise ValueError("Plugin is not installed in this runtime")
    target = root / ("installed" if enabled else "disabled") / name
    if enabled and verify:
        _validate(source)
        _verify_candidate(root, source, name)
    if source == target:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)


def _enable_binding(root: Path, name: str, revision: int | None) -> None:
    source = _location(root, name)
    _validate(source)
    _validate_dependency_publication(source, root, name)
    _verify_candidate(root, source, name)
    if revision is None:
        observations, removed, _warnings = _application_bindings(root)
        revision = application_inventory.reserve(root, observations, removed)
    with application_inventory.mutation(root, name, revision):
        _set_enabled(root, name, True, verify=False)
        _commit_binding_intent(root, _location(root, name), name, "enabled", revision)


def _manifest(plugin: Path) -> dict[str, Any]:
    path = plugin / ".pymss-install.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(task_id: str | None, message: str, phase: str = "working", *, plugin_name: str | None = None) -> None:
    emit("plugin_stage", {"message": message, "phase": phase, **({"pluginName": plugin_name} if plugin_name else {})}, task_id=task_id)


def _run_process(command: list[str], task_id: str | None, *, env: dict[str, str] | None = None) -> None:
    from pymss.operations import OperationContext, OperationError
    from pymss.operations._command_process import run_command_process
    check_cancelled()
    environment = dict(os.environ if env is None else env)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = run_command_process(command, cwd=Path.cwd(),
            context=OperationContext(cancelled=management_context()[1].cancelled), timeout=600, environment=environment)
    except OperationError as error:
        check_cancelled()
        if getattr(error, "detail", None):
            raise RuntimeError(f"Plugin dependency preparation failed: {error.detail[-3000:]}") from error
        raise
    for line in (result.stdout.splitlines() + result.stderr.splitlines())[-8:]:
        if line.strip():
            emit("plugin_log", {"message": line.strip()[-2000:]}, task_id=task_id)


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _plugin_requirements(plugin: Path) -> list[str]:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    project_file = plugin / "pyproject.toml"
    project = tomllib.loads(project_file.read_text(encoding="utf-8")).get("project", {}) if project_file.exists() else {}
    python_constraint = project.get("requires-python")
    if python_constraint and Version(".".join(map(str, sys.version_info[:3]))) not in SpecifierSet(python_constraint):
        raise ValueError(f"Plugin requires Python {python_constraint}")
    requirements = project.get("dependencies", [])
    if not isinstance(requirements, list) or not all(isinstance(item, str) for item in requirements):
        raise ValueError("Plugin dependencies must be a list of Python requirements")
    return requirements


def _dependency_versions(root: Path, exclude: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[_canonical(name)] = distribution.version
    intents = _application_state(root)[0]["plugins"]
    for other in _installed(root):
        if other.name == exclude:
            continue
        intent = intents.get(other.name, {})
        if intent.get("state", "enabled") != "enabled" or not _binding_matches(other, intent):
            continue
        for distribution in importlib.metadata.distributions(path=[str(other / ".studio-deps")]):
            name = _canonical(distribution.metadata["Name"])
            if name in versions and versions[name] != distribution.version:
                raise ValueError(f"Installed plugin dependency conflict: {name}")
            versions[name] = distribution.version
    return versions


def _validate_dependency_publication(plugin: Path, root: Path, name: str) -> None:
    dependencies = plugin / ".studio-deps"
    if not dependencies.is_dir():
        return
    versions = _dependency_versions(root, name)
    for distribution in importlib.metadata.distributions(path=[str(dependencies)]):
        dependency = _canonical(distribution.metadata["Name"])
        if dependency in versions and versions[dependency] != distribution.version:
            raise ValueError(f"Installed plugin dependency conflict: {dependency}")


def _install_dependencies(plugin: Path, root: Path, task_id: str | None, payload: dict) -> None:

    python_profiles.prepare_python(plugin, root, lambda message: _stage(task_id, message, "preparing"))
    requirements = _plugin_requirements(plugin)
    project_file = plugin / "pyproject.toml"
    if not requirements:
        return
    # Freeze host and peer sidecars, without borrowing another plugin's files.
    versions = _dependency_versions(root, plugin.name)
    constraints = plugin.parent / "constraints.txt"
    constraint_lines = [f"{name}=={version}\n" for name, version in sorted(versions.items())]
    constraint_lines.extend(management_context()[1].constraints(plugin))
    constraints.write_text("".join(constraint_lines), encoding="utf-8")
    report = plugin.parent / "pip-report.json"
    index_args = management_context()[1].index_args(payload)
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--prefer-binary"]
    # dependency_mode = "declared-only" (in [tool.pymss.plugin]): transitive
    # metadata is ignored — the plugin's dependency list IS the full closure.
    # Used when an upstream wheel pins a host-incompatible version of a package
    # that is actually compatible (e.g. gamemlx's conservative numpy<2 pin).
    plugin_meta = tomllib.loads(project_file.read_text(encoding="utf-8")).get("tool", {}).get("pymss", {}).get("plugin", {}) if project_file.exists() else {}
    declared_only = str(plugin_meta.get("dependency_mode") or "").strip().lower() == "declared-only"
    _stage(task_id, "Resolving plugin dependencies against the active runtime", "resolving")
    resolve_cmd = base + ["--dry-run", "--report", str(report), "--constraint", str(constraints)]
    if declared_only:
        resolve_cmd.append("--no-deps")
    _run_process(resolve_cmd + [*index_args, *requirements], task_id)
    resolved = json.loads(report.read_text(encoding="utf-8")).get("install", [])
    selected = [
        item["download_info"]["url"] if item.get("is_direct") else
        f"{item['metadata']['name']}=={item['metadata']['version']}"
        for item in resolved
        if not declared_only or _canonical(str(item["metadata"]["name"])) not in versions
    ]
    if selected:
        _stage(task_id, "Installing additional dependencies", "installing")
        _run_process(base + ["--no-deps", "--target", str(plugin / ".studio-deps"), *index_args, *selected], task_id)


def _finish_dependency_handoff(target: Path, backup: Path) -> None:
    marker = target / _DEPENDENCY_HANDOFF
    if marker.exists():
        dependencies = target / ".studio-deps"
        if not dependencies.exists():
            (backup / ".studio-deps").rename(dependencies)
        marker.unlink()


def _commit_binding_intent(root: Path, package: Path, name: str, state: str, revision: int | None) -> None:
    manifest, meta = _read_manifest(package), _manifest(package)
    if revision is None:
        observations, removed, _warnings = _application_bindings(root)
        revision = application_inventory.reserve(root, observations, removed)
    content = _plugin_content_digest(package)
    meta.update({"inventory": {"revision": revision, "state": state}, "inventoryContent": content})
    path = package / ".pymss-install.json"
    temporary = package / ".pymss-install-intent.tmp"
    try:
        temporary.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    application_inventory.commit(
        root, name, state, revision, version=manifest["version"], definition=_intent_definition(manifest), content=content,
    )


def _publish(plugin: Path, root: Path, name: str, *, disabled: bool = False, preserve_dependencies: bool = False,
             intent_revision: int | None = None) -> None:
    if intent_revision is None:
        observations, removed, _warnings = _application_bindings(root)
        intent_revision = application_inventory.reserve(root, observations, removed)
    with application_inventory.mutation(root, name, intent_revision):
        backup = _publish_locked(plugin, root, name, disabled=disabled, preserve_dependencies=preserve_dependencies, intent_revision=intent_revision)
    if backup.exists():
        # Deleting large sidecars must not block unrelated runtime publications.
        shutil.rmtree(backup, ignore_errors=True)


def _rename_plugin(source: Path, target: Path) -> None:
    deadline = time.monotonic() + 2
    while True:
        try:
            source.rename(target)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in {5, 32} or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _publish_locked(plugin: Path, root: Path, name: str, *, disabled: bool, preserve_dependencies: bool,
                    intent_revision: int) -> Path:
    target = root / ("disabled" if disabled else "installed") / name
    previous = _location(root, name)
    backup = root / ("disabled-backups" if previous.parent.name == "disabled" else "backups") / name
    target.parent.mkdir(parents=True, exist_ok=True)
    backup.parent.mkdir(parents=True, exist_ok=True)
    marker = plugin / _DEPENDENCY_HANDOFF
    marker.unlink(missing_ok=True)
    (plugin / ".pymss-install-intent.tmp").unlink(missing_ok=True)
    if preserve_dependencies:
        if not (previous / ".studio-deps").is_dir() or (plugin / ".studio-deps").exists():
            raise ValueError("Cannot reuse plugin dependencies during publication")
        # Recovery completes this handoff if the worker stops between the two renames.
        marker.write_text("1", encoding="utf-8")
    # The ordering marker travels with the atomically published package. A lost
    # application-record write can be reconstructed from this committed binding.
    meta = _manifest(plugin)
    content = _plugin_content_digest(plugin)
    meta.update({"inventory": {"revision": intent_revision, "state": "disabled" if disabled else "enabled"}, "inventoryContent": content})
    (plugin / ".pymss-install.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    if previous.exists():
        _rename_plugin(previous, backup)
    published = False
    try:
        _rename_plugin(plugin, target)
        published = True
        _finish_dependency_handoff(target, backup)
    except BaseException:
        if published:
            if preserve_dependencies and (target / ".studio-deps").exists() and not (backup / ".studio-deps").exists():
                (target / ".studio-deps").rename(backup / ".studio-deps")
            _rename_plugin(target, plugin)
        if backup.exists():
            _rename_plugin(backup, previous)
        raise
    manifest = _read_manifest(target)
    application_inventory.commit(
        root, name, "disabled" if disabled else "enabled", intent_revision,
        version=manifest["version"], definition=_intent_definition(manifest), content=content,
    )
    return backup


def _verify_candidate(root: Path, candidate: Path, name: str, *, candidate_dependencies: Path | None = None) -> None:
    """Validate activation without replacing a current plugin or breaking healthy peers."""
    transactions = root / "transactions"
    transactions.mkdir(parents=True, exist_ok=True)
    state = _application_state(root)[0]
    peers = [peer for peer in _installed(root)
             if state["plugins"].get(peer.name, {}).get("state", "enabled") == "enabled"
             and _binding_matches(peer, state["plugins"].get(peer.name, {}))]
    previous_failures = {row["name"] for row in _probe(root) if row.get("managed") and not row.get("loaded")} if peers else set()
    with tempfile.TemporaryDirectory(dir=transactions, prefix="install-verify-") as temporary:
        probe = Path(temporary)
        directory = probe / "installed"
        directory.mkdir()
        dependencies = [candidate_dependencies if candidate_dependencies is not None else candidate / ".studio-deps"]
        for peer in peers:
            if peer.name != name:
                shutil.copytree(peer, directory / peer.name, ignore=shutil.ignore_patterns(".studio-deps"))
                dependencies.append(peer / ".studio-deps")
        shutil.copytree(candidate, directory / name, ignore=shutil.ignore_patterns(".studio-deps"))
        from .leases import candidate_snapshot
        with candidate_snapshot(probe):
            reports = _probe(probe, dependencies)
        failed = [row for row in reports if row.get("managed") and not row.get("loaded") and (row["name"] == name or row["name"] not in previous_failures)]
        if not any(row.get("managed") and row["name"] == name for row in reports):
            raise ValueError(f"Plugin produced no registration report: {name}")
        if failed:
            raise ValueError("; ".join(f"{row['name']}: {row.get('error')}" for row in failed))


def _acquire_source(target: str, directory: Path, subpath: str | None = None):
    from datetime import datetime, timezone
    from pymss.plugins.install import InstallResult, install, list_available

    local_root = _LOCAL_SOURCE.get()
    from_default = bool(local_root and _NAME.fullmatch(target))
    if from_default:
        entry = next((row for row in _local_catalog(local_root) if row["name"] == target), None)
        if entry is None:
            raise ValueError(f"Local plugin registry has no entry for: {target}")
        target, subpath = str(local_root), entry["subpath"]

    local = Path(target).expanduser()
    if not local.is_dir():
        registry_name = target if _NAME.fullmatch(target) else None
        remote = target.startswith(("http://", "https://", "git@", "ssh://"))
        if not remote and (local.is_absolute() or target.startswith((".", "~")) or "/" in target or "\\" in target):
            raise ValueError(f"Local plugin source folder is unavailable: {local}")
        kwargs = {}
        expected_version = None
        if registry_name:
            entry = next((row for row in list_available(_REGISTRY_URL.get()) if row["name"] == registry_name), None)
            if entry is None or not entry.get("url"):
                raise ValueError(f"Plugin source catalog has no entry for: {registry_name}")
            target = entry["url"]
            subpath = entry.get("subpath") or subpath
            expected_version = entry.get("version")
            if entry.get("ref") is not None:
                if not isinstance(entry["ref"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", entry["ref"]) or ".." in entry["ref"]:
                    raise ValueError("Invalid plugin repository ref")
                kwargs["ref"] = entry["ref"]
        result = install(target, plugins_dir=directory, subpath=subpath, no_deps=True,
                         registry_url=_REGISTRY_URL.get(), temporary_root=directory.parent, **kwargs)
        expected_commit = kwargs.get("ref", "")
        if re.fullmatch(r"[0-9a-f]{40,64}", expected_commit) and _manifest(result.path).get("resolved_commit") != expected_commit:
            raise ValueError("Plugin source commit does not match the catalog")
        if registry_name:
            if expected_version is not None and _project_version(result.path) != expected_version:
                raise ValueError(f"Plugin source version does not match the catalog: {registry_name}")
            if result.name != registry_name:
                raise ValueError(f"Catalog entry {registry_name} installed plugin {result.name}")
            metadata = _manifest(result.path)
            metadata["name"] = registry_name
            metadata["defaultRepository"] = True
            metadata["sourceContext"] = {"registryUrl": _REGISTRY_URL.get(),
                                         "localSource": str(_LOCAL_SOURCE.get()) if _LOCAL_SOURCE.get() else None}
            (result.path / ".pymss-install.json").write_text(json.dumps(metadata), encoding="utf-8")
        return result
    source = local / subpath if subpath else local
    if not source.is_dir():
        raise ValueError(f"Local plugin source folder is unavailable: {source}")
    name = _read_manifest(source)["id"]
    destination = directory / name
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*_SOURCE_IGNORED_DIRS, ".env", ".env.*", ".DS_Store", "*.pyc", "*.pyo"))
    version = _project_version(destination)
    # Preserve the pymss manifest format without its CLI name@ref parsing of local paths.
    (destination / ".pymss-install.json").write_text(json.dumps({
        "name": name, "version": version, "source": "path", "url": str(Path(target).expanduser().resolve()),
        "subpath": subpath, "ref": None, "installed_at": datetime.now(timezone.utc).isoformat(),
        "defaultRepository": from_default,
        **({"sourceContext": {"registryUrl": _REGISTRY_URL.get(), "localSource": str(local_root)}} if from_default else {}),
    }), encoding="utf-8")
    return InstallResult(name=name, path=destination, source="path", version=version)


@contextmanager
def _installation_transaction(root: Path, *, concurrent: bool) -> Iterator[Path]:
    # Keep a lease journal until source cleanup; profiles stay at stable paths.
    try:
        with ExitStack() as resources:
            with runtime_lock(root / ".lock", timeout=180, cancelled=management_context()[1].cancelled) if concurrent else nullcontext():
                if concurrent:
                    _recover_when_idle(root)
                transactions = root / "transactions"
                transactions.mkdir(parents=True, exist_ok=True)
                directory = Path(resources.enter_context(tempfile.TemporaryDirectory(dir=transactions, prefix="install-")))
                resources.enter_context(runtime_lock(directory / ".lease", timeout=0))
            yield directory
    finally:
        # A cancelled preparation defers profile GC to the next management action.
        if not management_context()[1].cancelled():
            with runtime_lock(root / ".lock", timeout=180, cancelled=management_context()[1].cancelled) if concurrent else nullcontext():
                python_profiles.collect_python_profiles(root)


def _install(root: Path, payload: dict, task_id: str | None, *, intent_revision: int | None=None, prepared_binding: dict | None=None) -> str:
    from .leases import binding_lease

    action = payload['action']
    requested = _name(payload['name']) if payload.get('name') else None
    current = _location(root, requested) if requested else None
    prepared_source = Path(prepared_binding['path']) if prepared_binding is not None else None
    state = _application_state(root)[0]
    desired = state['plugins'].get(requested, {})
    disabled = not prepared_binding['enabled'] if prepared_binding is not None else action in {'update', 'repair'} and (desired.get('state') == 'disabled' if desired.get('revision', 0) > 0 else current is not None and current.parent.name == 'disabled')
    if prepared_source is not None:
        target = str(prepared_source)
    elif action in {'update', 'repair'}:
        if desired.get('state') == 'removed':
            raise ValueError('Plugin is not installed')
        if current is None or not current.is_dir():
            raise ValueError('Plugin is not installed in this runtime')
        meta = _manifest(current)
        if action == 'repair':
            target = str(current)
        else:
            target = requested if _uses_default_source(meta) else str(meta.get('url') or '')
            if not target:
                raise ValueError('Plugin has no update source; install it again from its source')
            if meta.get('source') == 'path' and target != requested and (not Path(target).is_dir()):
                raise ValueError(f'Local plugin source folder is unavailable: {target}')
    else:
        target = str(payload.get('target') or requested or '').strip()
        if not target:
            raise ValueError('Missing plugin source')
    if action == 'install' and prepared_source is None:
        local_candidate = Path(target).expanduser()
        if local_candidate.is_dir() and payload.get('subpath') is not None:
            local_candidate = _local_package(local_candidate, None, payload['subpath'])
        candidate = requested or (_read_manifest(local_candidate)['id'] if local_candidate.is_dir() else None)
        if candidate and _has_current_installation(root, _name(candidate)):
            raise ValueError('Plugin is already installed; use update or repair')
    with _installation_transaction(root, concurrent=action == 'install' and prepared_source is None) as work:
        _stage(task_id, 'Acquiring plugin source', 'acquiring')
        subpath = payload.get('subpath')
        if subpath is not None and (not isinstance(subpath, str)):
            raise ValueError('Plugin subpath must be a string')
        if action == 'update' and current and (not _uses_default_source(_manifest(current))):
            subpath = _manifest(current).get('subpath')
        with isolate_protocol_stdout():
            with _plugin_registry():
                result = _acquire_source(target, work / 'sources', subpath)
        name = _name(result.name)
        if prepared_source is not None:
            if _read_manifest(result.path)['version'] != prepared_binding.get('version') or _plugin_content_digest(result.path) != prepared_binding['content']:
                raise ValueError('Plugin source changed while preparing the runtime; refresh and try again')
            original = prepared_binding['meta']
            acquired = _manifest(result.path)
            acquired.update({key: value for key, value in original.items() if key not in {'inventory', 'inventoryContent', 'installed_at'}})
            if not original.get('source'):
                acquired.update({'source': 'recorded', 'url': None})
            (result.path / '.pymss-install.json').write_text(json.dumps(acquired), encoding='utf-8')
        _validate(result.path)
        if requested and (action in {'update', 'repair'} or prepared_source is not None) and (name != requested):
            raise ValueError(f'The update source provides plugin {name}, not {requested}')
        if action == 'install' and prepared_source is None and _has_current_installation(root, name):
            raise ValueError('Plugin is already installed; use update or repair')
        if action == 'repair' and current:
            shutil.copy2(current / '.pymss-install.json', result.path / '.pymss-install.json')
        elif current and _manifest(current).get('preinstalled'):
            meta = _manifest(result.path)
            meta['preinstalled'] = True
            (result.path / '.pymss-install.json').write_text(json.dumps(meta), encoding='utf-8')
        shutil.rmtree(result.path / '.studio-deps', ignore_errors=True)
        (result.path / python_profiles.BINDING).unlink(missing_ok=True)
        primary = _primary_operation(result.path)
        deferred = primary is not None and primary in _deferred_dependency_operations(result.path)
        isolated_python = python_profiles.declares_python(result.path)
        code_update = action == 'update' and (not isolated_python) and (not payload.get('installDependencies', False))
        deferred_install = action == 'install' and (not isolated_python) and deferred and (not payload.get('installDependencies', False))
        preserved_dependencies = None
        if code_update:
            _plugin_requirements(result.path)
            _stage(task_id, 'Preserving installed plugin dependencies', 'preserving')
            if current is not None and (current / '.studio-deps').is_dir():
                preserved_dependencies = current / '.studio-deps'
        elif deferred_install:
            _plugin_requirements(result.path)
            _stage(task_id, 'Optional backend dependencies are managed separately from the plugin files', 'preserving')
        else:
            _install_dependencies(result.path, root, task_id, payload)
        with ExitStack() as publication:
            if action == 'install' and prepared_source is None:
                # Queue installers before the short execution-occupancy check.
                publication.enter_context(runtime_lock(root / '.publication.lock', timeout=180,
                                                       cancelled=management_context()[1].cancelled))
                # Match the binding-to-writer lock order of other management actions.
                publication.enter_context(binding_lease(root, exclusive=True, timeout=5,
                                                        cancelled=management_context()[1].cancelled))
                publication.enter_context(runtime_lock(root / '.lock', timeout=180,
                                                       cancelled=management_context()[1].cancelled))
            if action == 'install' and prepared_source is None and _has_current_installation(root, name):
                raise ValueError('Plugin is already installed; use update or repair')
            if action == 'install':
                _validate_dependency_publication(result.path, root, name)
            _stage(task_id, 'Verifying plugin registration', 'verifying')
            if preserved_dependencies is not None:
                _verify_candidate(root, result.path, name, candidate_dependencies=preserved_dependencies)
            else:
                _verify_candidate(root, result.path, name)
            _stage(task_id, 'Activating plugin', 'activating')
            _publish(result.path, root, name, disabled=disabled, preserve_dependencies=preserved_dependencies is not None, intent_revision=intent_revision)
        return name


def _plugin_content_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for parent, subdirs, files in os.walk(directory):
        subdirs[:] = sorted(name for name in subdirs if name not in _SOURCE_IGNORED_DIRS)
        if any((Path(parent) / name).is_symlink() for name in subdirs):
            raise ValueError("Plugin sources with symbolic links cannot be compared")
        for name in sorted(files):
            if name in {".pymss-install.json", ".pymss-install-intent.tmp", ".pymss-python.json", _DEPENDENCY_HANDOFF, ".DS_Store", ".env"} or name.startswith(".env.") or name.endswith((".pyc", ".pyo")):
                continue
            path = Path(parent) / name
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
    return digest.hexdigest()


def _check_updates(root: Path, task_id: str | None, *, names: set[str] | None=None) -> dict:
    with runtime_lock(root / '.lock'):
        manifests = [(plugin.name, _manifest(plugin)) for plugin in [*_installed(root), *_installed(root, disabled=True)] if names is None or plugin.name in names]
    updates = {}
    checks = root / 'checks'
    checks.mkdir(parents=True, exist_ok=True)
    catalog_cache: dict | Exception | None = None
    commit_cache: dict[tuple[str, str | None], str] = {}

    def default_entry(name: str) -> dict:
        nonlocal catalog_cache
        if catalog_cache is None:
            try:
                if _LOCAL_SOURCE.get():
                    items = _local_catalog(_LOCAL_SOURCE.get())
                else:
                    from pymss.plugins.install import list_available
                    items = list_available(_REGISTRY_URL.get())
                catalog_cache = {item['name']: item for item in items}
            except Exception as error:
                catalog_cache = error
        if isinstance(catalog_cache, Exception):
            raise ValueError(f'Plugin source catalog is unavailable: {catalog_cache}')
        if name not in catalog_cache:
            raise ValueError(f'Plugin source catalog has no entry for: {name}')
        return catalog_cache[name]
    with tempfile.TemporaryDirectory(dir=checks) as temporary:
        view = Path(temporary) / 'installed'
        for name, meta in manifests:
            _stage(task_id, f'Checking {name}', 'checking', plugin_name=name)
            try:
                with _plugin_registry():
                    source = None
                    catalog_version = None
                    catalog_ref = None
                    if _uses_default_source(meta):
                        entry = default_entry(name)
                        if _LOCAL_SOURCE.get():
                            source = _local_package(_LOCAL_SOURCE.get(), name, entry.get('subpath'))
                        else:
                            meta = {**meta, 'url': entry['url'], 'subpath': entry.get('subpath')}
                            catalog_ref = entry.get('ref')
                            if isinstance(entry.get('version'), str) and entry['version'].strip():
                                catalog_version = entry['version']
                    elif meta.get('source') == 'path':
                        source = Path(meta['url'])
                        if meta.get('subpath'):
                            source = source / meta['subpath']
                    if source is not None:
                        if not source.is_dir():
                            raise ValueError(f'Local plugin source folder is unavailable: {source}')
                        remote = _project_version(source)
                        if remote is None:
                            raise ValueError('Local plugin source does not declare a version')
                        changed = _plugin_content_digest(source) != _plugin_content_digest(_location(root, name))
                        update = {'local_version': meta.get('version'), 'remote_version': remote, 'content_changed': changed, 'up_to_date': remote == meta.get('version') and (not changed)}
                    elif catalog_version is not None:
                        from pymss.plugins.sources import github_repository, resolve_github_commit
                        remote_commit = None
                        if github_repository(meta['url']):
                            key = (meta['url'], catalog_ref)
                            if key not in commit_cache:
                                commit_cache[key] = resolve_github_commit(*key)
                            remote_commit = commit_cache[key]
                        changed = remote_commit is not None and remote_commit != meta.get('resolved_commit')
                        update = {'local_version': meta.get('version'), 'remote_version': catalog_version, 'local_commit': meta.get('resolved_commit'), 'remote_commit': remote_commit, 'content_changed': changed, 'up_to_date': catalog_version == meta.get('version') and (not changed)}
                    else:
                        from pymss.plugins.install import check_update
                        directory = view / name
                        directory.mkdir(parents=True)
                        (directory / '.pymss-install.json').write_text(json.dumps(meta), encoding='utf-8')
                        update = check_update(name, plugins_dir=view)
                if update.get('remote_version') is None:
                    raise ValueError('The update source does not declare a version')
                updates[name] = update
            except Exception as error:
                updates[name] = {'error': str(error)}
    return updates


def _project_version(directory: Path) -> str | None:
    project = directory / "pyproject.toml"
    if not project.exists():
        return None
    return tomllib.loads(project.read_text(encoding="utf-8")).get("project", {}).get("version")



def validate_request(payload):
    action = str(payload.get("action") or "").strip()
    if action not in {"install", "update", "repair", "uninstall", "check_updates", "enable", "disable", "prepare", "prepare_preinstalled", "install_preinstalled"}:
        raise ValueError("Unsupported plugin action")
    if action in {"update", "repair", "uninstall", "enable", "disable", "prepare", "install_preinstalled"}:
        _name(payload.get("name"))
    if payload.get("subpath") is not None and not isinstance(payload["subpath"], str):
        raise ValueError("Plugin subpath must be a string")
    if "installDependencies" in payload and type(payload["installDependencies"]) is not bool:
        raise ValueError("installDependencies must be a boolean")
    if payload.get("defaultPolicy") not in {None, "auto", "review"}:
        raise ValueError("Invalid default plugin policy")


def manage(payload):
    task_id = str(payload.get("taskId") or "").strip() or None
    action = str(payload.get("action") or "").strip()
    validate_request(payload)
    root = _root()
    if payload.get("expectedRuntime") is not None and payload["expectedRuntime"] != root.name:
        raise ValueError("Active runtime changed; read the catalog again before acquiring plugins")
    if action in {"install", "update", "repair", "enable", "prepare", "prepare_preinstalled", "install_preinstalled"}:
        pass
    root.mkdir(parents=True, exist_ok=True)
    revision = None
    if action != "check_updates":
        observations, removed, _warnings = _application_bindings(root)
        revision = application_inventory.reserve(root, observations, removed)
    updates = {}
    if action == "check_updates":
        updates = _check_updates(root, task_id)
    elif action == "install":
        _install(root, {**payload, "action": action}, task_id, intent_revision=revision)
    else:
        with runtime_lock(root / ".lock"):
            _recover(root)
            if action == "uninstall":
                name = _name(payload.get("name"))
                target = _location(root, name)
                state = _application_state(root)[0]
                entry = state["plugins"].get(name)
                if entry is None:
                    raise ValueError("Plugin is not installed")
                if entry.get("pending", {}).get("binding", root.name) != root.name:
                    raise ValueError("Switch to the source runtime to finish removing its plugin files")
                with application_inventory.mutation(root, name, revision):
                    trash = root / "transactions" / f"removed-{name}"
                    _record_intent(root, name, "removed", pending="uninstall" if target.is_dir() or trash.exists() else None, revision=revision)
                    _stage(task_id, "Removing plugin and its additional dependencies", "removing")
                    if name in _bundled():
                        _mark_preinstalled_removed(root, name, True)
                    # Rename under the intent lock; large sidecars are deleted after it.
                    trash.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and not trash.exists():
                        target.rename(trash)
                if trash.exists():
                    shutil.rmtree(trash)
                if target.exists():
                    # A previous failed removal occupied the trash directory.
                    with application_inventory.mutation(root, name, revision):
                        target.rename(trash)
                    shutil.rmtree(trash)
                application_inventory.complete(root, name, revision)
            elif action in {"enable", "disable"}:
                _stage(task_id, "Changing plugin availability", "activating")
                name = _name(payload.get("name"))
                target = _location(root, name)
                entry = _application_state(root)[0]["plugins"].get(name)
                if entry is None:
                    raise ValueError("Plugin is not installed")
                if entry["state"] == "removed":
                    raise ValueError("Plugin is removed; reinstall it to restore its availability")
                if action == "enable":
                    if not target.is_dir() or not _binding_matches(target, entry):
                        _prepare_binding(root, {**payload, "name": name}, task_id, revision)
                    else:
                        _enable_binding(root, name, revision)
                else:
                    with application_inventory.mutation(root, name, revision):
                        _record_intent(root, name, "disabled", pending="disable" if target.is_dir() else None, revision=revision)
                        if target.is_dir():
                            _set_enabled(root, name, False)
                        application_inventory.complete(root, name, revision)
            elif action == "prepare":
                _stage(task_id, "Preparing plugin for the selected runtime", "preparing")
                _prepare_binding(root, payload, task_id, revision)
            elif action in {"prepare_preinstalled", "install_preinstalled"}:
                name = _name(payload.get("name")) if action == "install_preinstalled" else None
                if name is None:
                    state, _observations = _application_state(root)
                    if state["migrationWarnings"]:
                        raise ValueError("Plugin removal preferences must be readable before preparing default tools")
                    observations, removed, _warnings = _application_bindings(root)
                    application_inventory.set_default_policy(root, "auto", observations, removed)
                _prepare_preinstalled(root, task_id, name, intent_revision=revision)
            else:
                _install(root, {**payload, "action": action}, task_id, intent_revision=revision)
    if action != "check_updates":
        with runtime_lock(root / ".lock"):
            python_profiles.collect_python_profiles(root)
    return updates


# All durable mutations, including recovery, participate in the same execution
# gate. Acquisition can proceed independently; publication never replaces files
# held by another process. These guards retain the existing v1 writer locks.
def _guard(function, root_index, *, exclusive):
    from functools import wraps
    from .leases import binding_lease
    @wraps(function)
    def guarded(*args, **kwargs):
        root = args[root_index] if len(args) > root_index else kwargs["root"]
        with binding_lease(root, exclusive=exclusive, timeout=5, cancelled=management_context()[1].cancelled):
            if exclusive:
                check_cancelled()
            return function(*args, **kwargs)
    return guarded

for _function, _index in [
    ("_recover", 0), ("_publish", 1), ("_publish_locked", 1),
    ("_enable_binding", 0), ("_set_enabled", 0), ("_prepare_preinstalled", 0),
    ("_commit_binding_intent", 0), ("_record_intent", 0),
]:
    globals()[_function] = _guard(globals()[_function], _index, exclusive=True)
_verify_candidate = _guard(_verify_candidate, 0, exclusive=False)
