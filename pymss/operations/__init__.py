"""Operation model: serializable specs, execution contract, and runtime.

Design (round 2 of the plugin redesign, docs/plugin-redesign-plan.md):

- ``OperationSpec`` describes one executable processing step: named inputs,
  JSON-Schema params, named outputs. It is plain data — serializable without
  the handler — so hosts (Studio worker, Comfy, CLI) can render UI from it.
- The handler is attached at ``Runtime.register`` time and never travels with
  the spec.
- ``create_runtime()`` builds an *isolated* registry instance. The legacy
  global ``_REGISTRY`` keeps serving old-style plugins; runtimes own their
  operations outright.
- Validation is deliberately small: object-typed params with type/default/
  minimum/maximum/enum support and additionalProperties checking. It is not a
  full JSON-Schema implementation; specs that use unsupported keywords are
  rejected at register time rather than silently mis-validated.
"""
from __future__ import annotations

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

__all__ = [
    "OperationError",
    "OperationSpec",
    "OperationContext",
    "Runtime",
    "create_runtime",
    "AUDIO_TYPE",
]

AUDIO_TYPE = "audio"

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){1,}$")

_SUPPORTED_PARAM_TYPES = {"object", "number", "integer", "string", "boolean"}


class OperationError(Exception):
    """Raised for spec construction, registration, and execution failures."""


@dataclass(frozen=True)
class OperationSpec:
    """Serializable description of one operation.

    ``inputs``/``outputs`` map a port name to ``{"type": "audio"}``-style
    descriptors. ``params`` is a JSON-Schema object schema for the operation's
    parameter set. ``run`` is intentionally not part of the serialized form.
    """

    id: str
    schema_version: int
    title: str
    inputs: Mapping[str, Mapping[str, Any]]
    params: Mapping[str, Any]
    outputs: Mapping[str, Mapping[str, Any]]
    run: Callable[..., Any] | None = None  # attached at Runtime.register
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_PATTERN.match(self.id):
            raise OperationError(
                f"operation id must be a dot-namespaced lowercase identifier, got {self.id!r}"
            )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise OperationError("schema_version must be a positive integer")
        _validate_ports("inputs", self.inputs)
        _validate_ports("outputs", self.outputs)
        _validate_params_schema(self.params)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form safe for JSON serialization (no handler)."""
        return {
            "id": self.id,
            "schemaVersion": self.schema_version,
            "title": self.title,
            "description": self.description,
            "inputs": {k: dict(v) for k, v in self.inputs.items()},
            "params": dict(self.params),
            "outputs": {k: dict(v) for k, v in self.outputs.items()},
        }


def _validate_ports(kind: str, ports: Mapping[str, Mapping[str, Any]]) -> None:
    if not isinstance(ports, Mapping):
        raise OperationError(f"{kind} must be a mapping of port name -> descriptor")
    for name, descriptor in ports.items():
        if not isinstance(name, str) or not name:
            raise OperationError(f"{kind} port names must be non-empty strings")
        if not isinstance(descriptor, Mapping) or "type" not in descriptor:
            raise OperationError(
                f"{kind}[{name!r}] descriptor must be a mapping with a 'type' key"
            )


def _validate_params_schema(schema: Mapping[str, Any]) -> None:
    if not isinstance(schema, Mapping):
        raise OperationError("params must be a mapping (JSON-Schema object)")
    t = schema.get("type")
    if t != "object":
        raise OperationError("params schema must have type 'object' at the top level")
    props = schema.get("properties", {})
    if not isinstance(props, Mapping):
        raise OperationError("params.properties must be a mapping")
    for pname, pschema in props.items():
        if not isinstance(pschema, Mapping):
            raise OperationError(f"params.properties[{pname!r}] must be a mapping")
        ptype = pschema.get("type")
        if ptype not in _SUPPORTED_PARAM_TYPES - {"object"}:
            raise OperationError(
                f"param {pname!r} has unsupported type {ptype!r} "
                f"(supported: number, integer, string, boolean)"
            )
        for keyword in ("minimum", "maximum", "default", "enum"):
            if keyword in pschema and not _check_keyword_value(ptype, keyword, pschema[keyword]):
                raise OperationError(
                    f"param {pname!r} has a {keyword!r} value inconsistent with type {ptype!r}"
                )


def _check_keyword_value(ptype: str, keyword: str, value: Any) -> bool:
    if keyword == "enum":
        return isinstance(value, list) and value
    if keyword == "default":
        return _matches_type(ptype, value)
    if ptype == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if ptype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if ptype == "string":
        return isinstance(value, str)
    if ptype == "boolean":
        return isinstance(value, bool)
    return False


def _matches_type(ptype: str, value: Any) -> bool:
    if ptype == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if ptype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if ptype == "string":
        return isinstance(value, str)
    if ptype == "boolean":
        return isinstance(value, bool)
    return False


@dataclass
class OperationContext:
    """Execution context handed to every operation run.

    Round 2 keeps this minimal (progress, cancel, logger); later rounds add
    audio-service access. ``cancelled`` is checked by long-running operations
    cooperatively via ``ctx.check_cancelled()``.
    """

    workdir: str | None = None
    progress: Callable[[float, str], None] | None = None
    cancelled: Callable[[], bool] | None = None
    logger: Any = None

    def report_progress(self, fraction: float, message: str = "") -> None:
        if self.progress is not None:
            self.progress(fraction, message)

    def check_cancelled(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise OperationError("operation cancelled")


@dataclass
class _OperationEntry:
    spec: OperationSpec
    run: Callable[..., Any]
    source: str


@dataclass
class Runtime:
    """Isolated operation registry + executor.

    Not shared across processes or hosts; each host (CLI, Studio worker,
    Comfy) creates its own instance. The legacy global plugin registry is not
    consulted unless ``load_plugins`` discovers old-style plugins, in which
    case their capabilities/nodes stay in the global registry as before.
    """

    _operations: dict[str, _OperationEntry] = field(default_factory=dict)
    _load_report: list[dict[str, Any]] = field(default_factory=list)
    _manifest_plugin_dirs: dict[str, str] = field(default_factory=dict)

    # ---- registration ----

    def register(
        self,
        *,
        spec: OperationSpec,
        run: Callable[..., Any],
        source: str = "builtin",
    ) -> None:
        if spec.id in self._operations:
            existing = self._operations[spec.id]
            raise OperationError(
                f"operation {spec.id!r} already registered by {existing.source!r}"
            )
        if not callable(run):
            raise OperationError(f"run for {spec.id!r} must be callable")
        self._operations[spec.id] = _OperationEntry(spec=spec, run=run, source=source)

    # ---- discovery ----

    def list_operations(self) -> list[dict[str, Any]]:
        return [entry.spec.to_dict() for entry in self._operations.values()]

    def describe_operation(self, operation_id: str) -> dict[str, Any]:
        entry = self._require_entry(operation_id)
        return entry.spec.to_dict()

    def _require_entry(self, operation_id: str) -> _OperationEntry:
        entry = self._operations.get(operation_id)
        if entry is None:
            raise OperationError(f"unknown operation {operation_id!r}")
        return entry

    # ---- execution ----

    def run(
        self,
        operation_id: str,
        *,
        inputs: Mapping[str, Any],
        params: Mapping[str, Any],
        context: OperationContext | None = None,
    ) -> dict[str, Any]:
        entry = self._require_entry(operation_id)
        _check_inputs(entry.spec, inputs)
        resolved = _resolve_input_values(entry.spec, inputs)
        merged = _merge_and_validate_params(operation_id, entry.spec.params, params)
        ctx = context or OperationContext()
        result = entry.run(ctx, resolved, merged)
        if not isinstance(result, Mapping):
            raise OperationError(
                f"operation {operation_id!r} must return a mapping of named outputs"
            )
        return dict(result)

    # ---- plugin loading (setup protocol) ----

    def load_plugins(self) -> list[dict[str, Any]]:
        """Scan the plugins dir; each plugin's setup() registers into *this* runtime.

        Registrations made by a plugin whose setup() raises are rolled back,
        and the failure recorded in the report.

        Protocol v1 (docs/plugin-protocol.md): plugins with a
        ``[tool.pymss.plugin]`` manifest go through the PluginManager
        (dependency resolution, ordered activation, blocked diagnostics).
        Manifest-less plugins keep the legacy scan below.
        """
        import os
        from pathlib import Path

        self._load_report = []
        plugins_dir = os.environ.get("PYMSS_PLUGINS_DIR")
        if not plugins_dir:
            plugins_dir = str(Path.home() / ".pymss" / "plugins")
        root = Path(plugins_dir)
        if not root.exists():
            return self._load_report
        import importlib.util
        import sys

        # ---- protocol v1: manifest plugins (resolve → ordered activate) ----
        manifest_rows: dict[str, dict[str, Any]] = {}
        try:
            from ..plugins.lifecycle import PluginManager, pymss_package_version

            manager = PluginManager(root)
            states = manager.discover()
            for state in states:
                if state.manifest is None:
                    continue
                plugin_id = state.manifest.plugin_id
                previous = self._manifest_plugin_dirs.get(plugin_id)
                if previous is not None and previous != str(state.directory.resolve()):
                    state.status = "blocked"
                    state.error = f"duplicate plugin id {plugin_id!r}: already activated from {previous}"
            manager.resolve(pymss_version=pymss_package_version())
            rows = manager.activate(lambda source: _RuntimePluginHandle(self, source=source))
            for row in rows:
                name = row.get("name")
                if isinstance(name, str):
                    manifest_rows[name] = row
                    if row.get("loaded"):
                        self._manifest_plugin_dirs[row["plugin_id"]] = str((root / name).resolve())
        except Exception as exc:  # noqa: BLE001 — lifecycle must not break legacy loading
            self._load_report.append(
                {"name": "<plugin-lifecycle>", "loaded": False, "error": f"{type(exc).__name__}: {exc}"}
            )

        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            entry_file = next(
                (child / name for name in ("__init__.py", "plugin.py") if (child / name).exists()),
                None,
            )
            pyproject = child / "pyproject.toml"
            if entry_file is None:
                if pyproject.exists() and child.name in manifest_rows:
                    self._load_report.append(manifest_rows[child.name])
                elif pyproject.exists():
                    # Broken manifest (blocked at discovery) or manifest without entry file.
                    from ..plugins.manifest import ManifestError, read_manifest

                    try:
                        read_manifest(child)
                        self._load_report.append({"name": child.name, "loaded": False, "error": "no entry file"})
                    except ManifestError as exc:
                        self._load_report.append({"name": child.name, "loaded": False, "error": str(exc)})
                else:
                    self._load_report.append(
                        {"name": child.name, "loaded": False, "error": "no entry file"}
                    )
                continue
            if child.name in manifest_rows:
                # Manifest plugin already handled by the lifecycle manager.
                row = manifest_rows[child.name]
                if row.get("loaded"):
                    registered = sorted(
                        op_id for op_id, entry in self._operations.items()
                        if entry.source == f"plugin:{row.get('plugin_id')}"
                    )
                    row["operations"] = registered
                self._load_report.append(row)
                continue
            modname = f"_pymss_runtime_plugin_{child.name}"
            before = set(self._operations)
            handle = _RuntimePluginHandle(self, source=modname)
            try:
                spec = importlib.util.spec_from_file_location(modname, entry_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[modname] = module
                try:
                    spec.loader.exec_module(module)
                    setup = getattr(module, "setup", None)
                    if setup is not None:
                        setup(handle)
                except Exception:
                    sys.modules.pop(modname, None)
                    raise
            except Exception as exc:  # noqa: BLE001
                handle.rollback()
                self._load_report.append(
                    {
                        "name": child.name,
                        "loaded": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                self._load_report.append(
                    {
                        "name": child.name,
                        "loaded": True,
                        "legacy": True,
                        "api_version": 0,
                        "operations": sorted(set(self._operations) - before),
                    }
                )
        return self._load_report

    def plugin_report(self) -> list[dict[str, Any]]:
        return list(self._load_report)


@dataclass
class _RuntimePluginHandle:
    """setup(plugin) handle whose registrations target one runtime."""

    runtime: Runtime
    source: str

    def __init__(self, runtime: "Runtime", source: str):
        from ..plugins.registry import get_registry

        self.runtime = runtime
        self.source = source
        registry = get_registry()
        self._operations_before = dict(runtime._operations)
        self._capabilities_before = dict(registry.capabilities)
        self._nodes_before = dict(registry.nodes)
        self._cli_before = list(registry.cli_commands)

    def register_operation(self, *, id, params, inputs=None, outputs=None, run=None, **kwargs):
        spec = OperationSpec(
            id=id,
            schema_version=int(kwargs.pop("schema_version", 1)),
            title=kwargs.pop("title", id),
            description=kwargs.pop("description", ""),
            inputs=inputs or {},
            params=params,
            outputs=outputs or {},
        )
        if run is None:
            raise OperationError(f"operation {id!r} requires a run handler")
        self.runtime.register(spec=spec, run=run, source=self.source)

    def rollback(self) -> None:
        """Restore registrations made or replaced during this plugin's activation."""
        from ..plugins.registry import get_registry

        self.runtime._operations.clear()
        self.runtime._operations.update(self._operations_before)
        registry = get_registry()
        for current, before in (
            (registry.capabilities, self._capabilities_before),
            (registry.nodes, self._nodes_before),
        ):
            # Importing graph modules may initialize builtins. Preserve those
            # registrations because Python retains the imported modules too.
            builtins = {
                name: entry for name, entry in current.items()
                if name not in before and entry.source in {"builtin", "pymss.builtin"}
            }
            current.clear()
            current.update(builtins)
            current.update(before)
        builtin_commands = [
            entry for entry in registry.cli_commands
            if entry.source == "builtin" and not any(entry is previous for previous in self._cli_before)
        ]
        registry.cli_commands[:] = [*self._cli_before, *builtin_commands]

    # Legacy compatibility surface: a runtime plugin may still contribute
    # capabilities/nodes/cli via the old global registry if it wants to.

    def register_capability(self, name, func, *, description=""):
        from pymss.plugins import register_capability

        register_capability(name, func, source=self.source, description=description)

    def register_node(self, node_type, func, *, description="", signature=None):
        from pymss.plugins import register_node

        register_node(node_type, func, source=self.source, description=description, signature=signature)

    def register_cli(self, path, func, *, help="", add_arguments=None):
        from pymss.plugins import register_cli

        register_cli(path, func, help=help, source=self.source, add_arguments=add_arguments)


def _resolve_input_values(spec: OperationSpec, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce file descriptors into typed artifacts per the declared port type.

    The CLI passes ``{"kind": "file", "path": ...}`` descriptors; operations
    declared with an ``audio`` port expect an ``AudioArtifact``. Load at the
    boundary so every caller (CLI, bridge, tests) shares one conversion path.
    """
    resolved: dict[str, Any] = dict(inputs)
    for port_name, port_spec in spec.inputs.items():
        value = resolved.get(port_name)
        if not isinstance(value, Mapping) or value.get("kind") != "file":
            continue
        if not str(port_spec.get("type", "")).lower() == "audio":
            continue
        from ..graph.core import AudioArtifact
        from ..audio_io import load_audio

        path = str(value.get("path", ""))
        mix, sr, layout = load_audio(path, sr=None, mono=False, return_layout=True)
        resolved[port_name] = AudioArtifact(mix, int(sr), source_path=path, channel_layout=layout)
    return resolved


def _check_inputs(spec: OperationSpec, inputs: Mapping[str, Any]) -> None:
    for port_name in spec.inputs:
        if port_name not in inputs:
            raise OperationError(f"operation {spec.id!r} missing required input {port_name!r}")


def _merge_and_validate_params(
    operation_id: str,
    schema: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    props = schema.get("properties", {})
    allow_additional = schema.get("additionalProperties", True)
    merged: dict[str, Any] = {}
    for pname, pschema in props.items():
        if pname in params:
            merged[pname] = params[pname]
        elif "default" in pschema:
            merged[pname] = pschema["default"]
        elif pname in schema.get("required", []):
            raise OperationError(f"operation {operation_id!r} missing required param {pname!r}")
    for pname in params:
        if pname not in props and not allow_additional:
            raise OperationError(
                f"operation {operation_id!r} received unknown param {pname!r}"
            )
    for pname, value in merged.items():
        pschema = props[pname]
        ptype = pschema["type"]
        if not _matches_type(ptype, value):
            raise OperationError(
                f"operation {operation_id!r} param {pname!r} must be {ptype}, got {type(value).__name__}"
            )
        if "minimum" in pschema and value < pschema["minimum"]:
            raise OperationError(
                f"operation {operation_id!r} param {pname!r} below minimum {pschema['minimum']}: {value}"
            )
        if "maximum" in pschema and value > pschema["maximum"]:
            raise OperationError(
                f"operation {operation_id!r} param {pname!r} above maximum {pschema['maximum']}: {value}"
            )
        if "enum" in pschema and value not in pschema["enum"]:
            raise OperationError(
                f"operation {operation_id!r} param {pname!r} must be one of {pschema['enum']}"
            )
    return merged


def create_runtime(*, load_plugins: bool = False) -> Runtime:
    """Create an isolated runtime; optionally load plugins immediately."""
    runtime = Runtime()
    if load_plugins:
        runtime.load_plugins()
    return runtime


_DEFAULT_RUNTIME: Runtime | None = None


def default_runtime() -> Runtime:
    """Process-lazy default runtime used by the legacy setup() bridge.

    Distinct from ``create_runtime()`` instances: hosts that want isolation
    create their own; ``PluginRegistration.register_operation`` (the global
    loader's setup handle) lands here. The first call also registers the
    built-in operations so every host gets them without extra wiring.
    """
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = Runtime()
        from .builtins import register_builtin_operations

        register_builtin_operations()
    return _DEFAULT_RUNTIME
