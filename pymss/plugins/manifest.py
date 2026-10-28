"""Plugin manifest parsing for the pymss plugin protocol (docs/plugin-protocol.md).

Reads ``[tool.pymss.plugin]`` from a plugin's pyproject.toml. Discovery only —
no plugin code is executed here. Manifest-less directories remain legacy
plugins with unchanged behaviour.
"""

from __future__ import annotations

import re

import re
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Chinese vendor images ship 3.10
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLUGIN_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SINGLE_CONSTRAINT = r"(==|>=|<=|>|<|~)?\s*[0-9][0-9A-Za-z.\-+]*\s*"
VERSION_CONSTRAINT_RE = re.compile(rf"^\s*{_SINGLE_CONSTRAINT}(,\s*{_SINGLE_CONSTRAINT})*$")
API_VERSION = 1
SUPPORTED_PLACEMENTS = {"tool", "workspace"}
SUPPORTED_PLAYER_SCOPES = {"clip", "track"}
SUPPORTED_PLAYER_MODES = {"rendered_preview", "offline_export"}


class ManifestError(ValueError):
    """Raised when a plugin manifest is invalid."""


def _check_constraint(value: Any, field_name: str) -> str:
    text = ",".join(part.strip() for part in str(value or "").split(","))
    if not VERSION_CONSTRAINT_RE.match(text):
        raise ManifestError(f"{field_name} must be PEP 440-style version constraints, got {value!r}")
    return text


@dataclass
class GraphContribution:
    id: str
    file: Path
    operation: str


@dataclass
class NodeContribution:
    """A workflow-graph node type contributed by the plugin.

    Pure data: hosts render the node (palette, ports, widgets) from this
    declaration without any plugin-side frontend code.
    """

    type: str
    title: str
    category: str
    inputs: tuple[dict, ...]
    outputs: tuple[dict, ...]
    widgets: tuple[dict, ...]
    description: str = ""


@dataclass
class ViewContribution:
    id: str
    title: str
    entry: Path
    placements: tuple[str, ...]
    operation: str | None
    permissions: tuple[str, ...]


@dataclass
class PlayerContribution:
    operation: str
    scopes: tuple[str, ...]
    modes: tuple[str, ...]


@dataclass
class PluginManifest:
    """Parsed [tool.pymss.plugin] section; validated, paths resolved."""

    plugin_id: str
    api_version: int
    entry: str | None
    requires_pymss: str | None
    version: str | None
    dependencies: dict[str, str] = field(default_factory=dict)
    graphs: list[GraphContribution] = field(default_factory=list)
    nodes: list[NodeContribution] = field(default_factory=list)
    views: list[ViewContribution] = field(default_factory=list)
    player: list[PlayerContribution] = field(default_factory=list)
    legacy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "version": self.version,
            "api_version": self.api_version,
            "requires_pymss": self.requires_pymss,
            "dependencies": dict(self.dependencies),
            "contributions": {
                "graphs": [{"id": g.id, "file": g.file.name, "operation": g.operation} for g in self.graphs],
                "nodes": [
                    {
                        "type": n.type, "title": n.title, "category": n.category,
                        "inputs": [dict(i) for i in n.inputs], "outputs": [dict(o) for o in n.outputs],
                        "widgets": [dict(w) for w in n.widgets], "description": n.description,
                    }
                    for n in self.nodes
                ],
                "views": [
                    {
                        "id": v.id, "title": v.title, "entry": str(v.entry),
                        "placements": list(v.placements), "operation": v.operation,
                        "permissions": list(v.permissions),
                    }
                    for v in self.views
                ],
                "player": [
                    {"operation": p.operation, "scopes": list(p.scopes), "modes": list(p.modes)}
                    for p in self.player
                ],
            },
        }


def _resolve_inside(plugin_dir: Path, raw: Any, field_name: str, *, must_exist: bool = False) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ManifestError(f"{field_name} must be a non-empty relative path")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ManifestError(f"{field_name} must be relative to the plugin directory")
    resolved = (plugin_dir / candidate).resolve()
    root = plugin_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise ManifestError(f"{field_name} escapes the plugin directory: {raw!r}")
    if must_exist and not resolved.is_file():
        raise ManifestError(f"{field_name} does not exist inside the plugin: {raw!r}")
    return resolved


def _parse_graphs(plugin_dir: Path, contributions: Any) -> list[GraphContribution]:
    items = contributions.get("graphs", [])
    if not isinstance(items, list):
        raise ManifestError("contributions.graphs must be an array of tables")
    graphs: list[GraphContribution] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ManifestError("each graphs entry must be a table")
        graph_id = str(item.get("id") or "").strip()
        if not PLUGIN_ID_RE.fullmatch(graph_id):
            raise ManifestError(f"graphs entry id {item.get('id')!r} is invalid")
        if graph_id in seen:
            raise ManifestError(f"duplicate graphs id {graph_id!r}")
        seen.add(graph_id)
        operation = str(item.get("operation") or "").strip()
        if not operation:
            raise ManifestError(f"graphs entry {graph_id!r} must reference an operation")
        graphs.append(GraphContribution(
            id=graph_id,
            file=_resolve_inside(plugin_dir, item.get("file"), f"graphs[{graph_id}].file", must_exist=True),
            operation=operation,
        ))
    return graphs


NODE_PORT_TYPES = {"AUDIO", "STRING", "COMBO", "BOOLEAN", "PYMSS_MSS_PARAMS", "PYMSS_VR_PARAMS"}
NODE_WIDGET_TYPES = {"text", "number", "toggle", "combo", "slider"}
NODE_TYPE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _parse_node_ports(value: Any, field_name: str) -> tuple[dict, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{field_name} must be an array of tables")
    ports = []
    for item in value:
        if not isinstance(item, dict):
            raise ManifestError(f"{field_name} entries must be tables")
        name = str(item.get("name") or "").strip()
        ptype = str(item.get("type") or "").strip().upper()
        if not name:
            raise ManifestError(f"{field_name} entry is missing 'name'")
        if ptype not in NODE_PORT_TYPES:
            raise ManifestError(f"{field_name} entry {name!r} has unsupported port type {item.get('type')!r} (supported: {sorted(NODE_PORT_TYPES)})")
        entry = {"name": name, "type": ptype}
        if item.get("widget") is not None:
            entry["widget"] = str(item["widget"])
        ports.append(entry)
    return tuple(ports)


def _parse_node_widgets(value: Any, field_name: str) -> tuple[dict, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{field_name} must be an array of tables")
    widgets = []
    for item in value:
        if not isinstance(item, dict):
            raise ManifestError(f"{field_name} entries must be tables")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ManifestError(f"{field_name} entry is missing 'name'")
        wtype = str(item.get("type") or "text").strip().lower()
        if wtype not in NODE_WIDGET_TYPES:
            raise ManifestError(f"{field_name} entry {name!r} has unsupported widget type {wtype!r} (supported: {sorted(NODE_WIDGET_TYPES)})")
        widget: dict = {"name": name, "type": wtype}
        if item.get("default") is not None:
            widget["default"] = item["default"]
        if item.get("min") is not None:
            widget["min"] = item["min"]
        if item.get("max") is not None:
            widget["max"] = item["max"]
        if item.get("step") is not None:
            widget["step"] = item["step"]
        if item.get("options") is not None:
            options = item["options"]
            if not isinstance(options, list):
                raise ManifestError(f"{field_name} entry {name!r} options must be an array")
            widget["options"] = [str(option) for option in options]
        widgets.append(widget)
    return tuple(widgets)


def _parse_nodes(contributions: Any) -> list[NodeContribution]:
    items = contributions.get("nodes", [])
    if not isinstance(items, list):
        raise ManifestError("contributions.nodes must be an array of tables")
    nodes: list[NodeContribution] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ManifestError("each nodes entry must be a table")
        node_type = str(item.get("type") or "").strip()
        if not NODE_TYPE_RE.fullmatch(node_type):
            raise ManifestError(f"nodes entry type {item.get('type')!r} is invalid")
        if node_type in seen:
            raise ManifestError(f"duplicate nodes type {node_type!r}")
        seen.add(node_type)
        nodes.append(NodeContribution(
            type=node_type,
            title=str(item.get("title") or node_type),
            category=str(item.get("category") or "pymss/plugins"),
            inputs=_parse_node_ports(item.get("inputs", []), f"nodes[{node_type}].inputs"),
            outputs=_parse_node_ports(item.get("outputs", []), f"nodes[{node_type}].outputs"),
            widgets=_parse_node_widgets(item.get("widgets", []), f"nodes[{node_type}].widgets"),
            description=str(item.get("description") or ""),
        ))
    return nodes


def _parse_views(plugin_dir: Path, contributions: Any) -> list[ViewContribution]:
    items = contributions.get("views", [])
    if not isinstance(items, list):
        raise ManifestError("contributions.views must be an array of tables")
    views: list[ViewContribution] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ManifestError("each views entry must be a table")
        view_id = str(item.get("id") or "").strip()
        if not PLUGIN_ID_RE.fullmatch(view_id):
            raise ManifestError(f"views entry id {item.get('id')!r} is invalid")
        if view_id in seen:
            raise ManifestError(f"duplicate views id {view_id!r}")
        seen.add(view_id)
        placements_raw = item.get("placements", ["tool"])
        if not isinstance(placements_raw, list) or not placements_raw:
            raise ManifestError(f"views[{view_id}].placements must be a non-empty array")
        placements = tuple(str(p) for p in placements_raw)
        unknown = [p for p in placements if p not in SUPPORTED_PLACEMENTS]
        if unknown:
            raise ManifestError(f"views[{view_id}].placements has unsupported values {unknown}")
        permissions_raw = item.get("permissions", ["operations"])
        if not isinstance(permissions_raw, list):
            raise ManifestError(f"views[{view_id}].permissions must be an array")
        views.append(ViewContribution(
            id=view_id,
            title=str(item.get("title") or view_id),
            entry=_resolve_inside(plugin_dir, item.get("entry"), f"views[{view_id}].entry", must_exist=True),
            placements=placements,
            operation=(str(item["operation"]).strip() or None) if item.get("operation") else None,
            permissions=tuple(str(p) for p in permissions_raw),
        ))
    return views


def _parse_player(contributions: Any) -> list[PlayerContribution]:
    items = contributions.get("player", [])
    if not isinstance(items, list):
        raise ManifestError("contributions.player must be an array of tables")
    entries: list[PlayerContribution] = []
    for item in items:
        if not isinstance(item, dict):
            raise ManifestError("each player entry must be a table")
        operation = str(item.get("operation") or "").strip()
        if not operation:
            raise ManifestError("player entry must reference an operation")
        scopes_raw = item.get("scopes", ["clip", "track"])
        modes_raw = item.get("modes", ["rendered_preview", "offline_export"])
        scopes = tuple(str(s) for s in scopes_raw) if isinstance(scopes_raw, list) else ()
        modes = tuple(str(m) for m in modes_raw) if isinstance(modes_raw, list) else ()
        if not scopes or any(s not in SUPPORTED_PLAYER_SCOPES for s in scopes):
            raise ManifestError(f"player entry for {operation!r} has invalid scopes")
        if not modes or any(m not in SUPPORTED_PLAYER_MODES for m in modes):
            raise ManifestError(f"player entry for {operation!r} has invalid modes")
        entries.append(PlayerContribution(operation=operation, scopes=scopes, modes=modes))
    return entries


def read_manifest(plugin_dir: Path) -> PluginManifest | None:
    """Parse the plugin manifest. Returns None for legacy (manifest-less) plugins.

    Raises ManifestError for malformed manifests — a broken manifest blocks the
    plugin at load time instead of failing later during import.
    """
    pyproject = plugin_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ManifestError(f"cannot read pyproject.toml: {exc}") from exc
    section = data.get("tool", {}).get("pymss", {}).get("plugin")
    if not isinstance(section, dict) or not section:
        return None

    plugin_id = str(section.get("id") or "").strip()
    if not PLUGIN_ID_RE.fullmatch(plugin_id):
        raise ManifestError(f"plugin id {section.get('id')!r} must be lowercase dot/dash/underscore identifiers")
    api_version = section.get("api_version", API_VERSION)
    if not isinstance(api_version, int) or api_version < 1:
        raise ManifestError(f"api_version must be a positive integer, got {api_version!r}")

    dependencies: dict[str, str] = {}
    raw_deps = section.get("dependencies", {})
    if not isinstance(raw_deps, dict):
        raise ManifestError("plugin dependencies must be a table of plugin-id -> constraint")
    for dep_id, constraint in raw_deps.items():
        normalized = str(dep_id).strip()
        if not PLUGIN_ID_RE.fullmatch(normalized):
            raise ManifestError(f"dependency id {dep_id!r} is invalid")
        dependencies[normalized] = _check_constraint(constraint, f"dependencies[{dep_id}]")

    entry = section.get("entry")
    if entry is not None and (not isinstance(entry, str) or ":" not in entry):
        raise ManifestError(f"entry must be 'module:attr', got {entry!r}")

    contributions = section.get("contributions", {})
    if not isinstance(contributions, dict):
        raise ManifestError("contributions must be a table")

    project = data.get("project", {})
    return PluginManifest(
        plugin_id=plugin_id,
        api_version=api_version,
        entry=str(entry) if entry else None,
        requires_pymss=_check_constraint(section["requires_pymss"], "requires_pymss")
        if section.get("requires_pymss") else None,
        version=str(project.get("version")) if isinstance(project, dict) and project.get("version") else None,
        dependencies=dependencies,
        graphs=_parse_graphs(plugin_dir, contributions),
        nodes=_parse_nodes(contributions),
        views=_parse_views(plugin_dir, contributions),
        player=_parse_player(contributions),
    )


def parse_entry_point(entry: str | None, plugin_dir: Path) -> tuple[str, str]:
    """Resolve the manifest entry to (module, attr). Defaults to the directory name."""
    if entry:
        module, _, attr = entry.partition(":")
        return module.strip(), (attr.strip() or "setup")
    return plugin_dir.name, "setup"
