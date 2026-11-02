"""CLI entry point and adapter for the selected environment's plugin binding."""
from __future__ import annotations

import json
import argparse
import subprocess
import os
from pathlib import Path
import sys

from . import PluginContext, PluginPolicy, PluginService


def binding_from_args(args):
    value = getattr(args, "plugin_store", None) or os.environ.get("PYMSS_PLUGIN_BINDING")
    if value:
        return Path(value).expanduser().absolute()
    if os.environ.get("PYMSS_PLUGINS_DIR"):
        return None
    from .environment import environment_binding

    registered = environment_binding()
    return registered.binding_dir if registered else None


def main(argv=None):
    """Enter the core selected for this interpreter before importing pymss."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--plugin-store", "--plugin-binding")
    selected, _ = selector.parse_known_args(arguments)
    explicit = selected.plugin_store is not None or os.environ.get("PYMSS_PLUGIN_BINDING") or os.environ.get("PYMSS_PLUGINS_DIR")
    registered = None
    if not explicit:
        from .environment import environment_binding

        try:
            registered = environment_binding()
            current_core = Path(__file__).resolve().parent.parent
            if registered:
                prefixes = [str(registered.core_path or current_core), *(str(path) for path in registered.import_paths)]
                environment = os.environ.copy()
                environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(filter(None, [
                    *prefixes, *environment.get("PYTHONPATH", "").split(os.pathsep),
                ])))
            if registered and registered.core_path is not None and registered.core_path != current_core:
                if os.environ.get("_PYMSS_CLI_CORE"):
                    raise ValueError("The interpreter did not load its registered pymss core")
                environment["_PYMSS_CLI_CORE"] = str(registered.core_path)
                options = subprocess._args_from_interpreter_flags()
                # CPython's helper only includes a subset of -X settings.
                options.extend(f"-X{name}" if value is True else f"-X{name}={value}"
                               for name, value in sys._xoptions.items())
                if getattr(sys.stdout, "write_through", False):
                    options.append("-u")
                # -I and -E ignore PYTHONPATH. Inject the registered paths in
                # the child without enabling ambient environment/user paths.
                bootstrap = (f"import runpy, sys; sys.path[:0] = {prefixes!r}; "
                             "runpy.run_module('pymss', run_name='__main__', alter_sys=True)")
                # The Windows venv launcher cannot reliably exec itself.
                # Keep stdio and the console process group attached to the CLI.
                try:
                    with subprocess.Popen([sys.executable, *options, "-B", "-P", "-c", bootstrap, *arguments],
                                          env=environment) as child:
                        return child.wait()
                except KeyboardInterrupt:
                    return 130
        except (OSError, ValueError) as error:
            print(f"pymss: {error}", file=sys.stderr)
            return 1
    os.environ.pop("_PYMSS_CLI_CORE", None)
    if registered:
        # Imported plugin code and its dependency paths belong to this process.
        sys.path[:] = [*prefixes, *(path for path in sys.path if path not in prefixes)]
        os.environ["PYTHONPATH"] = environment["PYTHONPATH"]
    from pymss.cli import main as core_main

    return core_main(arguments)


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
        for name, update in sorted(result.get("updates", {}).items()):
            if update.get("error"):
                print(f"{name}: check failed: {update['error']}")
            elif update.get("up_to_date"):
                print(f"{name}: up to date ({update.get('remote_version') or '?'})")
            else:
                print(f"{name}: update available ({update.get('local_version') or '?'} -> {update.get('remote_version') or '?'})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
