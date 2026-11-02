from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from pymss_plugin_manager import environment
from pymss_plugin_manager.cli import binding_from_args


@pytest.fixture(autouse=True)
def isolated_environments(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMSS_PLUGIN_ENVIRONMENTS_DIR", str(tmp_path / "environments"))
    monkeypatch.delenv("PYMSS_PLUGIN_BINDING", raising=False)
    monkeypatch.delenv("PYMSS_PLUGINS_DIR", raising=False)


def test_registration_is_per_interpreter_and_idempotent(tmp_path):
    binding = tmp_path / "plugins/cpu"
    registered = environment.register_environment_binding(binding)
    assert registered == environment.environment_binding()
    record = environment._record_path(sys.executable)
    os.utime(record, ns=(1_000_000_000, 1_000_000_000))
    environment.register_environment_binding(binding)
    assert record.stat().st_mtime_ns == 1_000_000_000
    other_python = tmp_path / "cuda/Scripts/python.exe"
    assert environment.environment_binding(other_python) is None
    with mock.patch.object(sys, "executable", str(other_python)), mock.patch.object(sys, "prefix", str(other_python.parent.parent)):
        environment.register_environment_binding(tmp_path / "plugins/cuda")
        assert environment.environment_binding().binding_dir == (tmp_path / "plugins/cuda").resolve()
    assert environment.environment_binding().binding_dir == binding.resolve()
    assert environment.environment_binding(other_python).binding_dir == (tmp_path / "plugins/cuda").resolve()


def test_python_identity_preserves_virtual_environment_paths(tmp_path, monkeypatch):
    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("Interpreter symlinks must retain their environment path")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)
    first = environment.python_identity(tmp_path / "cpu/bin/python")
    second = environment.python_identity(tmp_path / "cuda/bin/python")
    assert first != second


def test_default_binding_and_explicit_overrides(tmp_path, monkeypatch):
    args = SimpleNamespace(plugin_store=None)
    assert binding_from_args(args) is None
    default = tmp_path / "plugins/cpu"
    environment.register_environment_binding(default)
    assert binding_from_args(args) == default.resolve()
    monkeypatch.setenv("PYMSS_PLUGINS_DIR", str(tmp_path / "legacy"))
    assert binding_from_args(args) is None
    explicit = tmp_path / "explicit/cpu"
    monkeypatch.setenv("PYMSS_PLUGIN_BINDING", str(explicit))
    assert binding_from_args(args) == explicit.absolute()
    selected = tmp_path / "selected/cpu"
    assert binding_from_args(SimpleNamespace(plugin_store=str(selected))) == selected.absolute()


def test_corrupt_record_cannot_select_a_legacy_directory(tmp_path):
    environment.register_environment_binding(tmp_path / "plugins/cpu")
    environment._record_path(sys.executable).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="record"):
        binding_from_args(SimpleNamespace(plugin_store=None))
    explicit = tmp_path / "selected/cpu"
    assert binding_from_args(SimpleNamespace(plugin_store=str(explicit))) == explicit.absolute()


@pytest.mark.parametrize("change", [
    {"pythonPath": "/another/python"}, {"bindingDir": "relative/cpu"},
    {"schemaVersion": True}, {"corePath": "/missing/core"},
])
def test_invalid_records_are_rejected(tmp_path, change):
    environment.register_environment_binding(tmp_path / "plugins/cpu")
    path = environment._record_path(sys.executable)
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(change)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        environment.environment_binding()


def test_failed_publication_keeps_the_previous_binding(tmp_path, monkeypatch):
    first = tmp_path / "plugins/cpu"
    environment.register_environment_binding(first)
    path = environment._record_path(sys.executable)
    original = path.read_bytes()

    def fail(*args):
        raise PermissionError("Record is locked")

    monkeypatch.setattr(environment.os, "replace", fail)
    with pytest.raises(PermissionError):
        environment.register_environment_binding(tmp_path / "plugins/other")
    assert path.read_bytes() == original
    assert environment.environment_binding().binding_dir == first.resolve()
    assert not list(path.parent.glob(".environment-*"))


@pytest.mark.parametrize("shadow", [False, True])
def test_registered_core_is_selected_before_cli_import(tmp_path, shadow):
    repository = Path(__file__).resolve().parents[1]
    overlay = tmp_path / "overlay"
    shutil.copytree(repository / "pymss_plugin_manager", overlay / "pymss_plugin_manager",
                    ignore=shutil.ignore_patterns("__pycache__"))
    package = overlay / "pymss"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "from pymss_plugin_manager.cli import main\nraise SystemExit(main())\n", encoding="utf-8")
    (package / "cli.py").write_text(
        "import json, os\nfrom types import SimpleNamespace\n"
        "from pymss_plugin_manager.cli import binding_from_args\n"
        "def main(argv):\n"
        " print(json.dumps({'core':'overlay','args':argv,'binding':str(binding_from_args(SimpleNamespace(plugin_store=None))), 'cwd':os.getcwd(), 'marker':os.environ.get('_PYMSS_CLI_CORE')}))\n"
        " return 0\n", encoding="utf-8")
    working = tmp_path / "working"
    working.mkdir()
    if shadow:
        foreign = working / "pymss"
        foreign.mkdir()
        (foreign / "__init__.py").write_text("", encoding="utf-8")
        (foreign / "__main__.py").write_text("raise SystemExit('Wrong core from working directory')", encoding="utf-8")
    binding = tmp_path / "plugins/cpu"
    environment.register_environment_binding(binding, core_path=overlay)
    result = subprocess.run([sys.executable, "-B", "-m", "pymss_plugin_manager.cli", "ops", "list"],
                            cwd=working, env={**os.environ, "PYTHONPATH": str(repository)},
                            capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "core": "overlay", "args": ["ops", "list"], "binding": str(binding.resolve()),
        "cwd": str(working), "marker": None,
    }


def _entry_point_cores(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    base, overlay = tmp_path / "base", tmp_path / "overlay"
    for root in (base, overlay):
        shutil.copytree(repository / "pymss_plugin_manager", root / "pymss_plugin_manager",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (root / "pymss").mkdir()
        shutil.copyfile(repository / "pymss/__main__.py", root / "pymss/__main__.py")
    shutil.copyfile(repository / "pymss/__init__.py", base / "pymss/__init__.py")
    (base / "pymss/separator.py").write_text("raise ImportError('Base inference is unavailable')\n", encoding="utf-8")
    (overlay / "pymss/__init__.py").write_text("", encoding="utf-8")
    (overlay / "pymss/cli.py").write_text(
        "import json, sys\nimport registered_dependency\n"
        "def main(argv):\n"
        " print(json.dumps({'args':argv, 'utf8':sys.flags.utf8_mode, 'isolated':sys.flags.isolated, "
        "'ignore_environment':sys.flags.ignore_environment, 'no_user_site':sys.flags.no_user_site, "
        "'optimize':sys.flags.optimize, 'warnings':sys.warnoptions, 'xoptions':sys._xoptions, "
        "'unbuffered':sys.stdout.write_through}))\n"
        " return 0\n", encoding="utf-8")
    components = tmp_path / "components"
    components.mkdir()
    (components / "registered_dependency.py").write_text("", encoding="utf-8")
    environment.register_environment_binding(tmp_path / "plugins/cpu", core_path=overlay, import_paths=(components,))
    return base


def test_module_entry_delegates_before_loading_base_inference(tmp_path):
    base = _entry_point_cores(tmp_path)
    result = subprocess.run([sys.executable, "-B", "-m", "pymss", "plugins", "list"],
                            cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(base)},
                            capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["args"] == ["plugins", "list"]


@pytest.mark.parametrize("options, expected", [
    (["-X", "utf8"], {"utf8": 1}),
    (["-I", "-X", "utf8"], {"utf8": 1, "isolated": 1, "ignore_environment": 1, "no_user_site": 1}),
    (["-I", "-u", "-X", "utf8=0"], {"utf8": 0, "isolated": 1, "unbuffered": True}),
    (["-E", "-s", "-O", "-W", "error::UserWarning", "-X", "utf8", "-X", "int_max_str_digits=1000"],
     {"utf8": 1, "ignore_environment": 1, "no_user_site": 1, "optimize": 1,
      "warnings": ["error::UserWarning"], "xoptions": {"utf8": True, "int_max_str_digits": "1000"}}),
])
def test_core_delegation_preserves_interpreter_options(tmp_path, options, expected):
    base = _entry_point_cores(tmp_path)
    # Isolated interpreters ignore PYTHONPATH; simulate the installed CLI entry.
    code = (f"import sys; sys.path.insert(0, {str(base)!r}); "
            "from pymss_plugin_manager.cli import main; raise SystemExit(main(['plugins', 'list']))")
    result = subprocess.run([sys.executable, "-B", *options, "-c", code],
                            cwd=tmp_path, env={**os.environ, "PYTHONUTF8": "0"},
                            capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["args"] == ["plugins", "list"]
    assert {key: actual[key] for key in expected} == expected


@pytest.mark.parametrize("first_import", ["pymss.plugins", "pymss.ensemble"])
def test_lazy_public_api_preserves_exports_and_builtin_capabilities(first_import):
    repository = Path(__file__).resolve().parents[1]
    code = (
        f"import sys, importlib; sys.path.insert(0, {str(repository)!r}); import pymss; "
        "assert 'torch' not in sys.modules; assert set(pymss.__all__) <= set(dir(pymss)); "
        f"importlib.import_module({first_import!r}); "
        "from pymss.plugins import require_capability; "
        "assert all(callable(require_capability(name)) for name in ('ensemble', 'to_mono', 'wav_encode')); "
        "namespace = {}; exec('from pymss import *', namespace); "
        "assert set(pymss.__all__) <= namespace.keys(); "
        "from pymss.separator import MSSeparator; from pymss.model_download import ProxyError; "
        "assert pymss.MSSeparator is MSSeparator and pymss.ProxyError is ProxyError; "
        "assert pymss.MSSeparator is namespace['MSSeparator']"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_context_records_are_bounded(tmp_path):
    environment.register_environment_binding(tmp_path / "plugins/cpu")
    environment._record_path(sys.executable).write_bytes(b" " * 16385)
    with pytest.raises(ValueError, match="size limit"):
        environment.environment_binding()


def test_core_delegation_cannot_recurse(tmp_path, monkeypatch, capsys):
    from pymss_plugin_manager import cli

    registered = environment.EnvironmentBinding(Path(sys.executable), tmp_path / "plugins/cpu", tmp_path / "overlay")
    monkeypatch.setenv("_PYMSS_CLI_CORE", str(registered.core_path))
    with mock.patch.object(environment, "environment_binding", return_value=registered), mock.patch.object(cli.subprocess, "Popen") as process:
        assert cli.main(["ops", "list"]) == 1
    process.assert_not_called()
    assert "did not load" in capsys.readouterr().err


def test_interrupted_core_delegation_unwinds_its_child(tmp_path):
    from pymss_plugin_manager import cli

    registered = environment.EnvironmentBinding(Path(sys.executable), tmp_path / "plugins/cpu", tmp_path / "overlay")
    with mock.patch.object(environment, "environment_binding", return_value=registered), mock.patch.object(cli.subprocess, "Popen") as process:
        process.return_value.__enter__.return_value.wait.side_effect = KeyboardInterrupt
        assert cli.main(["ops", "list"]) == 130
    assert process.return_value.__exit__.call_args.args[0] is KeyboardInterrupt


@pytest.mark.parametrize("options", [
    {"binding_dir": ""}, {"binding_dir": " "}, {"core_path": ""},
])
def test_empty_registration_paths_are_rejected(tmp_path, options):
    arguments = {"binding_dir": tmp_path / "plugins/cpu", **options}
    with pytest.raises(ValueError, match="empty"):
        environment.register_environment_binding(**arguments)
    assert environment.environment_binding() is None


def test_empty_interpreter_identity_cannot_be_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", "")
    with pytest.raises(ValueError, match="empty"):
        environment.register_environment_binding(tmp_path / "plugins/cpu")


@pytest.mark.parametrize("configuration", ["argument", "binding-variable", "legacy-directory"])
def test_explicit_context_skips_automatic_core_selection(tmp_path, monkeypatch, configuration):
    from pymss_plugin_manager import cli

    arguments = ["plugins", "list"]
    if configuration == "argument":
        arguments = ["--plugin-binding", str(tmp_path / "explicit/cpu"), *arguments]
    else:
        variable = "PYMSS_PLUGIN_BINDING" if configuration == "binding-variable" else "PYMSS_PLUGINS_DIR"
        monkeypatch.setenv(variable, str(tmp_path / "explicit/cpu"))
    with (
        mock.patch.object(environment, "environment_binding", side_effect=AssertionError("Automatic context was used")),
        mock.patch.dict(sys.modules, {
            "pymss": SimpleNamespace(), "pymss.cli": SimpleNamespace(main=lambda argv: 42 if argv == arguments else 1),
        }),
    ):
        assert cli.main(arguments) == 42


def test_recreated_environment_cannot_use_a_stale_core_record(tmp_path, monkeypatch):
    environment.register_environment_binding(tmp_path / "plugins/cpu")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "other-environment"))
    with pytest.raises(ValueError, match="environment changed"):
        environment.environment_binding()


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-path spelling")
def test_managed_identity_accepts_windows_extended_path_aliases(tmp_path, monkeypatch):
    from pymss_plugin_manager import PluginContext, PluginService

    manager = PluginService(PluginContext(tmp_path / "cpu"))
    manager._identity(write=True)
    path = manager.context.root / ".pymss-binding.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    prefix = chr(92) * 2 + "?" + chr(92)
    for name in ("python", "prefix"):
        value["environment"][name] = prefix + value["environment"][name]
    path.write_text(json.dumps(value), encoding="utf-8")
    original = path.read_bytes()
    manager._identity()
    monkeypatch.setattr(sys, "executable", prefix + sys.executable)
    monkeypatch.setattr(sys, "prefix", prefix + sys.prefix)
    manager._identity()
    assert path.read_bytes() == original


def test_explicit_legacy_directory_still_uses_metadata_only_listing(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    legacy = tmp_path / "legacy"
    package = legacy / "test.legacy"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname="legacy"\nversion="1.0"\n'
        '[tool.pymss.plugin]\nid="test.legacy"\napi_version=2\n', encoding="utf-8")
    (package / "__init__.py").write_text("raise AssertionError('Listing executed setup')", encoding="utf-8")
    environment.register_environment_binding(tmp_path / "managed/cpu")
    result = subprocess.run([sys.executable, "-B", "-m", "pymss", "plugins", "list", "--json"],
                            cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(repository), "PYMSS_PLUGINS_DIR": str(legacy)},
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    entries = json.loads(result.stdout)
    assert len(entries) == 1
    assert entries[0]["name"] == "test.legacy"


def test_optional_component_paths_follow_install_and_removal(tmp_path):
    component = tmp_path / "components/funasr"
    registered = environment.register_environment_binding(tmp_path / "plugins/cpu", import_paths=(component,))
    assert registered.import_paths == ()
    record = environment._record_path(sys.executable)
    original = record.read_bytes()
    component.mkdir(parents=True)
    assert environment.environment_binding().import_paths == (component.resolve(),)
    component.rmdir()
    assert environment.environment_binding().import_paths == ()
    assert record.read_bytes() == original


@pytest.mark.parametrize("paths", ["not-a-list", ("",), (None,), ["/unused"] * 33])
def test_optional_component_paths_are_bounded_and_validated(tmp_path, paths):
    with pytest.raises(ValueError):
        environment.register_environment_binding(tmp_path / "plugins/cpu", import_paths=paths)
