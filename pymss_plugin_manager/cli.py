"""CLI adapter for an explicitly selected managed binding."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import PluginContext, PluginPolicy, PluginService


def binding_from_args(args):
    value = getattr(args, "plugin_store", None) or os.environ.get("PYMSS_PLUGIN_BINDING")
    return Path(value).expanduser().absolute() if value else None


def run(args, *, action=None):
    root = binding_from_args(args)
    if root is None:
        raise ValueError("Select a managed binding with --plugin-store or PYMSS_PLUGIN_BINDING")
    service = PluginService(PluginContext(root, registry_url=os.environ.get("PYMSS_PLUGINS_REGISTRY")),
                            PluginPolicy(event=lambda event, payload, task_id: None))
    action = action or getattr(args, "plugins_command", "list")
    exit_code = 0
    enabled = set()
    if action in {"list", "dir"}:
        result = service.list()
        if action == "dir":
            print(result["pluginsDir"])
            return 0
    elif action == "probe":
        from .service import probe
        from .leases import binding_lease
        with binding_lease(root):
            enabled = {row["name"] for row in service.list()["installed"] if row["enabled"]}
            result = {"reports": probe(root)}
            exit_code = int(any(not row.get("loaded") and row.get("name") in enabled for row in result["reports"]))
    elif action == "recover":
        service.recover()
        result = service.list()
    else:
        payload = {"taskId": "cli-plugin-management"}
        if action == "install":
            if getattr(args, "no_deps", False):
                raise ValueError("Managed installations verify dependencies; --no-deps applies only to legacy directories")
            payload["target"] = args.target
            if getattr(args, "subpath", None):
                payload["subpath"] = args.subpath
        elif action != "check_updates":
            payload["name"] = args.name
        if getattr(args, "sync_dependencies", False):
            payload["installDependencies"] = True
        outcome = service.manage(action, **payload)
        result = service.list()
        if action == "check_updates":
            result["updates"] = outcome["updates"]
    if getattr(args, "json", False):
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Plugin binding: {root}")
        for row in result.get("reports", []):
            status = "OK" if row.get("loaded") else "FAILED" if row.get("name") in enabled else "SKIPPED"
            detail = f": {row['error']}" if row.get("error") else ""
            print(f"{row['name']} {status}{detail}")
        for row in result.get("installed", []):
            print(f"{row['name']} {row.get('version') or '?'} {row['state']} "
                  f"{'prepared' if row['prepared'] else 'not prepared'}")
    return exit_code
