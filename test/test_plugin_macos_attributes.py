from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

install_mod = importlib.import_module("pymss.plugins.install")


def package(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="provider"\nversion="1.0"\n'
        '[tool.pymss.plugin]\nid="author.provider"\napi_version=2\n', encoding="utf-8")
    (source / "__init__.py").write_text("def setup(plugin): pass\n", encoding="utf-8")
    return source


@pytest.mark.parametrize("remote", [False, True])
def test_macos_install_clears_only_the_staged_plugin_before_publication(tmp_path, monkeypatch, remote):
    source = package(tmp_path)
    store = tmp_path / "installed"
    calls = []

    def run(command, **options):
        staged = Path(command[-1])
        assert staged.is_relative_to(store)
        assert staged != source
        assert (staged / "pyproject.toml").is_file()
        assert not (store / "author.provider").exists()
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, "", "")

    def clone(url, destination, **options):
        shutil.copytree(source, destination)
        return {"resolved_commit": "a" * 40}

    monkeypatch.setattr(install_mod, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(install_mod.subprocess, "run", run)
    monkeypatch.setattr(install_mod, "_clone", clone)
    result = install_mod.install("https://github.com/author/plugin" if remote else str(source),
                                 plugins_dir=store, no_deps=True)
    assert result.path == store / "author.provider"
    assert len(calls) == 1
    command, options = calls[0]
    assert command[:3] == ["/usr/bin/xattr", "-rc", "-s"]
    assert options["check"] is True
    assert options["timeout"] == 30
    assert (source / "__init__.py").read_text() == "def setup(plugin): pass\n"


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_other_platforms_do_not_run_xattr(tmp_path, monkeypatch, platform):
    source = package(tmp_path)
    run = Mock(side_effect=AssertionError("Unexpected native command"))
    monkeypatch.setattr(install_mod, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(install_mod.subprocess, "run", run)
    result = install_mod.install(str(source), plugins_dir=tmp_path / "installed", no_deps=True)
    assert result.path.is_dir()
    run.assert_not_called()


@pytest.mark.parametrize("error", [
    subprocess.CalledProcessError(1, "xattr", stderr="Operation not permitted"),
    subprocess.TimeoutExpired("xattr", 30),
    FileNotFoundError("xattr unavailable"),
])
def test_cleanup_failure_aborts_before_publishing_the_plugin(tmp_path, monkeypatch, error):
    source = package(tmp_path)
    monkeypatch.setattr(install_mod, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(install_mod.subprocess, "run", Mock(side_effect=error))
    store = tmp_path / "installed"
    with pytest.raises(install_mod.InstallError, match="macOS plugin attribute cleanup failed"):
        install_mod.install(str(source), plugins_dir=store, no_deps=True)
    assert not (store / "author.provider").exists()
    assert (source / "pyproject.toml").is_file()
