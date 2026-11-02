"""Persistent plugin management shared by command-line and graphical hosts."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sys

from .context import PluginContext, PluginPolicy, use, check_cancelled
from .leases import binding_lease, hold_runtime, release_binding, is_candidate
from .locks import runtime_lock


class PluginService:
    def __init__(self, context: PluginContext, policy: PluginPolicy | None = None):
        self.context = context
        self.policy = policy or PluginPolicy()

    @contextmanager
    def bound(self):
        with use(self.context, self.policy):
            yield self

    def _identity(self, *, write=False):
        root = self.context.root
        path = root / ".pymss-binding.json"
        actual = {"python": os.path.normcase(os.path.abspath(sys.executable)),
                  "prefix": os.path.normcase(os.path.abspath(sys.prefix)),
                  "pythonVersion": list(sys.version_info[:2]),
                  "platform": sys.platform, "architecture": platform.machine().lower()}
        if path.exists():
            if path.is_symlink() or path.stat().st_size > 16384:
                raise ValueError("PLUGIN_ENV_MISMATCH: Invalid plugin binding identity")
            recorded = json.loads(path.read_text(encoding="utf-8"))
            if recorded.get("schemaVersion") != 1 or recorded.get("environment") != actual:
                raise ValueError("PLUGIN_ENV_MISMATCH: Use this plugin binding's Python interpreter")
        elif write:
            # The writer lock excludes competing initializers. Existing stores
            # retain their paths and v1 inventory, and are adopted in place.
            root.mkdir(parents=True, exist_ok=True)
            temporary = root / ".pymss-binding.tmp"
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump({"schemaVersion": 1, "environment": actual}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)

    def manage(self, action: str, **payload):
        from . import _engine
        _engine.validate_request({**payload, "action": action})
        if payload.get("expectedRuntime") not in {None, self.context.root.name}:
            raise ValueError("Active runtime changed; refresh the plugin context")
        with self.bound():
            check_cancelled()
            self._identity()
            if action != "check_updates" and not (self.context.root / ".pymss-binding.json").exists():
                with binding_lease(self.context.root, exclusive=True, timeout=5, cancelled=self.policy.cancelled), runtime_lock(self.context.root / ".lock"):
                    _engine.application_inventory.merge(self.context.root, [], set())
                    self._identity(write=True)
            if action == "check_updates":
                updates = self._check_sources(payload.get("taskId"))
            elif action == "install":
                updates = _engine.manage({**payload, "action": action})
            else:
                source_context = self._origin_context(payload.get("name")) if action == "update" else self.context
                with binding_lease(self.context.root, exclusive=True), use(source_context, self.policy):
                    updates = _engine.manage({**payload, "action": action})
            return {"action": action, "applied": action != "check_updates", "updates": updates,
                    "binding": self.context.root.name}

    def _origin_context(self, name):
        """CLI defaults retain the source selected when the plugin was installed."""
        from . import _engine
        if self.context.registry_url is not None or self.context.local_source is not None:
            return self.context
        if not name:
            return self.context
        meta = _engine._manifest(_engine._location(self.context.root, _engine._name(name)))
        if not _engine._uses_default_source(meta):
            return self.context
        origin = meta.get("sourceContext")
        if origin is None:
            registry = meta.get("registry_url")
            if not registry and str(meta.get("url", "")).removesuffix(".git").rstrip("/") == "https://github.com/pymss-project/pymss-plugins":
                registry = "https://raw.githubusercontent.com/pymss-project/pymss-plugins/main/registry.json"
            if registry:
                origin = {"registryUrl": registry, "localSource": None}
        if not isinstance(origin, dict):
            raise ValueError("Select the original plugin catalog before updating this installation")
        registry, local = origin.get("registryUrl"), origin.get("localSource")
        if (registry is not None and not isinstance(registry, str)) or (local is not None and (not isinstance(local, str) or not Path(local).is_absolute())):
            raise ValueError("Invalid recorded plugin source context")
        return replace(self.context, registry_url=registry, local_source=Path(local) if local else None)

    def _check_sources(self, task_id):
        from . import _engine
        groups, updates = {}, {}
        intents = _engine._application_state(self.context.root)[0]["plugins"]
        for plugin in [*_engine._installed(self.context.root), *_engine._installed(self.context.root, disabled=True)]:
            if intents.get(plugin.name, {}).get("state") == "removed":
                continue
            try:
                context = self._origin_context(plugin.name)
                groups.setdefault(context, set()).add(plugin.name)
            except ValueError as error:
                updates[plugin.name] = {"error": str(error)}
        for context, names in groups.items():
            with use(context, self.policy):
                updates.update(_engine._check_updates(self.context.root, task_id, names=names))
        return updates

    def list(self):
        """Read committed metadata without importing or executing plugin code."""
        from . import _engine
        with self.bound():
            self._identity()
            state, observations = _engine._application_state(self.context.root)
            rows = []
            for name, entry in state["plugins"].items():
                if entry["state"] == "removed" and not entry.get("pending"):
                    continue
                path = _engine._location(self.context.root, name)
                prepared = path.is_dir() and _engine._binding_matches(path, entry)
                if entry["state"] == "enabled" and entry.get("revision", 0) > 0 and path.parent.name == "disabled":
                    prepared = False
                rows.append({"name": name, "version": entry.get("version"),
                             "enabled": entry["state"] == "enabled", "prepared": prepared,
                             "state": entry["state"], "pending": entry.get("pending"),
                             "path": str(path) if prepared else "", "definition": entry.get("definition", {}),
                             "availableBindings": [item["binding"] for item in observations if item["name"] == name]})
            return {"installed": rows, "revision": state["revision"],
                    "runtime": {"identity": self.context.root.name, "pythonPath": sys.executable,
                                "bindingDir": str(self.context.root), "managementSupported": True},
                    "pluginsDir": str(self.context.root / "installed")}

    def install(self, source: str, **options):
        return self.manage("install", target=source, **options)

    def update(self, name: str, **options):
        return self.manage("update", name=name, **options)

    def repair(self, name: str, **options):
        return self.manage("repair", name=name, **options)

    def enable(self, name: str, **options):
        return self.manage("enable", name=name, **options)

    def disable(self, name: str, **options):
        return self.manage("disable", name=name, **options)

    def uninstall(self, name: str, **options):
        return self.manage("uninstall", name=name, **options)

    def prepare(self, name: str, **options):
        return self.manage("prepare", name=name, **options)

    def check_updates(self):
        return self.manage("check_updates")["updates"]

    def recover(self):
        from . import _engine
        if not self.context.root.exists():
            return
        with self.bound(), binding_lease(self.context.root, exclusive=True), runtime_lock(self.context.root / ".lock"):
            self._identity()
            _engine._recover(self.context.root)
            _engine.python_profiles.collect_python_profiles(self.context.root)

    def hold_runtime(self):
        self._identity()
        acquired = hold_runtime(self.context.root)
        try:
            self._identity()
        except BaseException:
            if acquired:
                release_binding(self.context.root)
            raise

    def activation_validator(self):
        """Freeze enabled bindings while the caller holds its execution lease."""
        from . import _engine
        self.hold_runtime()
        with self.bound():
            try:
                intents = _engine._application_state(self.context.root)[0]["plugins"]
            except (OSError, ValueError, TypeError, KeyError):
                intents = None

        def validate(path):
            with self.bound():
                if intents is None:
                    raise ValueError("Plugin inventory is unavailable; managed code cannot run")
                intent = intents.get(path.name, {})
                if intent.get("state", "enabled") != "enabled":
                    raise ValueError("Plugin is disabled by the plugin configuration")
                if not _engine._binding_matches(path, intent):
                    raise ValueError("Plugin version is not prepared for this runtime")
                sidecar = path / ".studio-deps"
                if sidecar.is_dir() and str(sidecar) not in sys.path:
                    sys.path.append(str(sidecar))
                _engine._validate(path)
                try:
                    _engine._validate_requirements(path)
                except ValueError:
                    declared = {item["id"] for item in _engine._read_manifest(path)["operations"]}
                    if not declared.intersection(self.policy.deferred_operations(path)):
                        raise
        return validate

    def load_into(self, runtime, *, activation_context=None):
        """Load only this binding, applying intent before executing any setup."""
        validate_binding = self.activation_validator()
        with self.bound():
            def validate(path):
                validate_binding(path)

            return runtime.load_plugins([self.context.root / "installed"], validate_plugin=validate,
                                        activation_context=activation_context)


def probe(root: Path, dependency_paths=None, *, component_paths=lambda plugin: [],
          deferred_operations=lambda plugin: set(), cancelled=lambda: False):
    from contextlib import nullcontext
    temporary = is_candidate(root)
    if not temporary:
        PluginService(PluginContext(root))._identity()
    with nullcontext() if temporary else binding_lease(root):
        if not temporary:
            PluginService(PluginContext(root))._identity()
        return _probe_snapshot(root, dependency_paths, component_paths=component_paths,
                               deferred_operations=deferred_operations, cancelled=cancelled)


def _probe_snapshot(root: Path, dependency_paths=None, *, component_paths, deferred_operations, cancelled):
    """A candidate probe loads explicit paths, never reacquiring a parent writer's lease."""
    from . import _engine
    temporary = is_candidate(root)
    allowed = None
    if not temporary:
        with use(PluginContext(root), PluginPolicy()):
            try:
                intents = _engine._application_state(root)[0]["plugins"]
                allowed = [path.name for path in _engine._installed(root)
                           if intents.get(path.name, {}).get("state", "enabled") == "enabled"
                           and _engine._binding_matches(path, intents.get(path.name, {}))]
            except (OSError, ValueError, TypeError, KeyError):
                allowed = []
    packages = [p for p in (root / "installed").glob("*") if allowed is None or p.name in allowed]
    deferred, components, errors = [], [], {}
    for package in packages:
        try:
            declared = {item["id"] for item in _engine._read_manifest(package)["operations"]}
            if declared.intersection(deferred_operations(package)):
                deferred.append(package.name)
            components.extend(str(path) for path in component_paths(package))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors[package.name] = str(error)[-3000:]
    paths = list(dict.fromkeys([*(str(p) for p in dependency_paths or []),
                              *(str(p / ".studio-deps") for p in packages),
                              *components]))
    code = """
import json, sys
from pathlib import Path
from contextlib import redirect_stdout
from pymss_plugin_manager.leases import candidate_snapshot
sys.path.extend(json.loads(sys.argv[2]))
from pymss.operations import default_runtime
from pymss_plugin_manager import PluginContext, PluginPolicy, _engine
from pymss_plugin_manager.context import use
allowed = json.loads(sys.argv[3])
deferred = json.loads(sys.argv[4])
errors = json.loads(sys.argv[5])
def validate(path):
    if allowed is not None and path.name not in allowed:
        raise ValueError('Plugin intent does not permit activation')
    if path.name in errors:
        raise ValueError(errors[path.name])
    if path.name not in deferred:
        _engine._validate_requirements(path)
with candidate_snapshot(Path(sys.argv[1])), use(PluginContext(Path(sys.argv[1])), PluginPolicy()), redirect_stdout(sys.stderr):
    reports = default_runtime().load_plugins([Path(sys.argv[1]) / 'installed'], validate_plugin=validate)
print('__PYMSS_PLUGIN_PROBE__' + json.dumps([{**row, 'managed': True} for row in reports]))
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # A probe must not recursively open the published binding or use the
    # caller's default plugin roots; it receives only the candidate snapshot.
    env.pop("PYMSS_PLUGIN_BINDING", None)
    from pymss.operations import OperationContext, OperationError
    from pymss.operations._command_process import run_command_process
    try:
        result = run_command_process([sys.executable, "-B", "-c", code, str(root), json.dumps(paths), json.dumps(allowed), json.dumps(deferred), json.dumps(errors)],
                                     cwd=Path.cwd(), context=OperationContext(cancelled=cancelled),
                                     timeout=45, environment=env)
    except OperationError as error:
        if cancelled():
            raise InterruptedError("Plugin management was cancelled") from None
        if getattr(error, "detail", None):
            raise RuntimeError(f"Plugin verification failed: {error.detail[-3000:]}") from error
        raise
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("__PYMSS_PLUGIN_PROBE__"):
            return json.loads(line[len("__PYMSS_PLUGIN_PROBE__"):])
    raise RuntimeError((result.stderr or "Plugin verification produced no report")[-3000:])
