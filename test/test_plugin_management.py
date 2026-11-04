from __future__ import annotations

import importlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from pymss_plugin_manager import PluginContext, PluginPolicy, PluginService
from pymss_plugin_manager import _engine, inventory
from pymss_plugin_manager.leases import binding_lease, release_runtime
from pymss_plugin_manager.locks import RuntimeLockBusy, runtime_lock


def source(tmp_path, plugin_id="author.provider"):
    package = tmp_path / "source"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname="provider"\nversion="1.0"\ndependencies=[]\n'
        f'[tool.pymss.plugin]\nid="{plugin_id}"\napi_version=2\n', encoding="utf-8")
    (package / "__init__.py").write_text(
        'def setup(plugin):\n    plugin.register_capability("management_probe", lambda: 42)\n', encoding="utf-8")
    return package


def service(root):
    # These tests exercise persistent transactions; real process loading is
    # tested separately and by the cross-host integration suite.
    return PluginService(PluginContext(root), PluginPolicy(probe=lambda root, paths: [
        {"name": path.name, "managed": True, "loaded": True} for path in (root / "installed").iterdir()
    ]))


@pytest.fixture(autouse=True)
def release_leases():
    yield
    release_runtime()


def test_macos_management_cleans_each_acquired_copy_before_verification(tmp_path, monkeypatch):
    install_mod = importlib.import_module("pymss.plugins.install")
    events = []
    package = source(tmp_path)

    def cleanup(command, **options):
        assert command[:3] == ["/usr/bin/xattr", "-rc", "-s"]
        candidate = Path(command[-1])
        assert candidate.is_relative_to(tmp_path / "plugins")
        assert candidate != package
        events.append("cleanup")
        return subprocess.CompletedProcess(command, 0, "", "")

    def probe(root, paths):
        events.append("probe")
        return [{"name": path.name, "managed": True, "loaded": True} for path in (root / "installed").iterdir()]

    monkeypatch.setattr(install_mod, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(install_mod.subprocess, "run", cleanup)
    cpu = PluginService(PluginContext(tmp_path / "plugins/cpu"), PluginPolicy(probe=probe))
    cuda = PluginService(PluginContext(tmp_path / "plugins/cuda"), PluginPolicy(probe=probe))
    actions = [lambda: cpu.install(str(package)), lambda: cpu.update("author.provider"),
               lambda: cpu.manage("repair", name="author.provider"), lambda: cuda.prepare("author.provider")]
    for action in actions:
        events.clear()
        action()
        assert events[0] == "cleanup"
        assert events.count("cleanup") == 1
        assert "probe" in events


def test_failed_macos_cleanup_keeps_the_installed_binding_and_skips_verification(tmp_path, monkeypatch):
    install_mod = importlib.import_module("pymss.plugins.install")
    manager = service(tmp_path / "plugins/cpu")
    package = source(tmp_path)
    manager.install(str(package))
    installed = manager.context.root / "installed/author.provider"
    before = {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*") if p.is_file()}
    probe = mock.Mock(side_effect=AssertionError("The candidate must not be loaded"))
    manager.policy.probe = probe
    monkeypatch.setattr(install_mod, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(install_mod.subprocess, "run", mock.Mock(side_effect=
        subprocess.CalledProcessError(1, "xattr", stderr="Operation not permitted")))
    with pytest.raises(install_mod.InstallError, match="Operation not permitted"):
        manager.update("author.provider")
    assert {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*") if p.is_file()} == before
    probe.assert_not_called()


def test_inventory_api_import_and_query_do_not_initialize_the_core(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    code = (
        f"import sys;sys.path.insert(0,{str(repository)!r});"
        "from pathlib import Path;from pymss_plugin_manager import PluginContext,PluginService;"
        f"assert not PluginService(PluginContext(Path({str(tmp_path / 'cpu')!r}))).list()['installed'];"
        "assert 'pymss' not in sys.modules and 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_sibling_bindings_share_intent_without_sharing_mutable_dependencies(tmp_path):
    package = source(tmp_path)
    cpu, cuda = service(tmp_path / "plugins/cpu"), service(tmp_path / "plugins/cuda")
    cpu.install(str(package))
    row = cuda.list()["installed"][0]
    assert row["enabled"] and not row["prepared"]
    cuda.prepare("author.provider")
    assert cuda.list()["installed"][0]["prepared"]
    cuda.uninstall("author.provider")
    assert (cpu.context.root / "installed/author.provider").is_dir()
    assert cpu.list()["installed"] == []
    with pytest.raises(ValueError, match="removed"):
        cpu.enable("author.provider")
    with pytest.raises(ValueError, match="removed"):
        cpu.prepare("author.provider")


def test_read_lease_cannot_be_upgraded_even_in_the_same_execution(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    with binding_lease(manager.context.root):
        with pytest.raises(RuntimeLockBusy):
            manager.disable("author.provider")
    assert manager.list()["installed"][0]["enabled"]


def test_install_publication_excludes_a_competing_management_writer(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    package = source(tmp_path / "pending", "author.pending")
    ready, published = threading.Event(), threading.Event()
    outcomes = []
    original_lock, original_publish = _engine.runtime_lock, _engine._publish

    def disable():
        try:
            manager.disable("author.provider")
            outcomes.append("disabled")
        except RuntimeLockBusy:
            outcomes.append("busy")
        except Exception as error:
            outcomes.append(error)
        finally:
            ready.set()

    contender = threading.Thread(target=disable)

    @contextmanager
    def runtime_lock(path, **options):
        if threading.current_thread() is contender and path == manager.context.root / ".lock":
            # Pause the contender after admission, before it waits for the
            # publisher's writer lock. Publication must already exclude it.
            ready.set()
            assert published.wait(10), "Publication did not release the competing writer"
        with original_lock(path, **options) as writable:
            yield writable

    def publish(*args, **options):
        contender.start()
        try:
            assert ready.wait(10), "Competing writer did not reach admission"
            return original_publish(*args, **options)
        finally:
            published.set()

    with mock.patch.object(_engine, "runtime_lock", runtime_lock), mock.patch.object(_engine, "_publish", publish):
        try:
            manager.install(str(package))
        finally:
            published.set()
            if contender.ident is not None:
                contender.join(15)
    assert not contender.is_alive()
    assert outcomes == ["busy"]
    assert {row["name"] for row in manager.list()["installed"] if row["enabled"]} == {"author.provider", "author.pending"}


def test_parallel_installs_wait_for_slow_candidate_verification(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    packages = [source(tmp_path / name, "author." + name) for name in ("slow", "fast")]
    acquiring, verifying = threading.Event(), threading.Event()
    original_acquire, original_probe = _engine._acquire_source, manager.policy.probe

    def acquire(target, *args, **options):
        if target == str(packages[1]):
            acquiring.set()
            assert verifying.wait(10), "First candidate did not reach verification"
        return original_acquire(target, *args, **options)

    def probe(root, paths):
        if root != manager.context.root and (root / "installed/author.slow").exists() and not verifying.is_set():
            verifying.set()
            # Longer than the execution lease's five-second busy timeout.
            time.sleep(7)
        return original_probe(root, paths)

    manager.policy.probe = probe
    with mock.patch.object(_engine, "_acquire_source", acquire), ThreadPoolExecutor(max_workers=2) as pool:
        fast = pool.submit(manager.install, str(packages[1]))
        assert acquiring.wait(10), "Second source acquisition did not start"
        slow = pool.submit(manager.install, str(packages[0]))
        slow.result(timeout=30)
        fast.result(timeout=30)
    assert {row["name"] for row in manager.list()["installed"]} == {"author.provider", "author.slow", "author.fast"}


def test_parallel_source_acquisition_preserves_protocol_stdout(tmp_path, capsys):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    packages = [source(tmp_path / name, "author." + name) for name in ("first", "second")]
    entered = [threading.Event(), threading.Event()]
    proceed = [threading.Event(), threading.Event()]
    original_acquire, original_stdout = _engine._acquire_source, sys.stdout

    def acquire(target, *args, **options):
        index = [str(package) for package in packages].index(target)
        entered[index].set()
        assert proceed[index].wait(10), "Source acquisition was not released"
        return original_acquire(target, *args, **options)

    try:
        with mock.patch.object(_engine, "_acquire_source", acquire), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(manager.install, str(packages[0]))
            assert entered[0].wait(10)
            second = pool.submit(manager.install, str(packages[1]))
            assert entered[1].wait(10)
            print('{"event":"host-protocol"}')
            proceed[0].set()
            first.result(timeout=15)
            proceed[1].set()
            second.result(timeout=15)
        assert sys.stdout is original_stdout
        assert '{"event":"host-protocol"}' in capsys.readouterr().out
    finally:
        for event in proceed:
            event.set()
        sys.stdout = original_stdout


def test_profile_collection_waits_for_installation_transactions(tmp_path):
    from pymss_plugin_manager.profiles import collect_python_profiles

    root = tmp_path / "plugins/cpu"
    profile = root / "python-profiles" / ("python-" + "a" * 32)
    profile.mkdir(parents=True)
    transaction = root / "transactions/install-active"
    transaction.mkdir(parents=True)
    with mock.patch("pymss.operations.python_profiles.remove_python_environment") as remove:
        with runtime_lock(root / ".lock"):
            collect_python_profiles(root)
        remove.assert_not_called()
        assert profile.is_dir()
        transaction.rmdir()
        with runtime_lock(root / ".lock"):
            collect_python_profiles(root)
        remove.assert_called_once_with(profile)


def test_cancelled_install_leaves_the_publication_queue_without_waiting_for_gc(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    package = source(tmp_path / "pending", "author.pending")
    queued, cancelled = threading.Event(), threading.Event()
    manager.policy.cancelled = cancelled.is_set
    original_lock = _engine.runtime_lock
    queue = manager.context.root / ".publication.lock"

    @contextmanager
    def observe_queue(path, **options):
        if path == queue:
            queued.set()
        with original_lock(path, **options) as writable:
            yield writable

    with ThreadPoolExecutor(max_workers=1) as pool, mock.patch.object(_engine, "runtime_lock", observe_queue):
        with runtime_lock(queue):
            pending = pool.submit(manager.install, str(package))
            assert queued.wait(10), "Installer did not reach the publication queue"
            # An unrelated writer may still be verifying while cancellation
            # cleans the source. GC must not wait for its runtime lock.
            with runtime_lock(manager.context.root / ".lock"):
                cancelled.set()
                with pytest.raises(InterruptedError, match="cancelled"):
                    pending.result(timeout=3)
    assert [row["name"] for row in manager.list()["installed"]] == ["author.provider"]
    assert not list((manager.context.root / "transactions").glob("install-*"))


@pytest.mark.parametrize("phase", ["preparation", "publication"])
def test_install_writer_wait_is_cancellable(tmp_path, phase):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    package = source(tmp_path / "pending", "author.pending")
    waiting, resume, cancelled = threading.Event(), threading.Event(), threading.Event()
    manager.policy.cancelled = cancelled.is_set
    original_lock = _engine.runtime_lock
    root_lock = manager.context.root / ".lock"
    publishing = False

    @contextmanager
    def observe_writer(path, **options):
        nonlocal publishing
        if path == manager.context.root / ".publication.lock":
            publishing = True
        if path == root_lock and publishing == (phase == "publication"):
            waiting.set()
            assert resume.wait(10), "Writer wait did not resume"
        with original_lock(path, **options) as writable:
            yield writable

    with ThreadPoolExecutor(max_workers=1) as pool, mock.patch.object(_engine, "runtime_lock", observe_writer):
        pending = pool.submit(manager.install, str(package))
        try:
            assert waiting.wait(10), "Installer did not reach the writer lock"
            with runtime_lock(root_lock):
                cancelled.set()
                resume.set()
                with pytest.raises(InterruptedError, match="cancelled"):
                    pending.result(timeout=3)
        finally:
            resume.set()
    assert [row["name"] for row in manager.list()["installed"]] == ["author.provider"]
    assert not list((manager.context.root / "transactions").glob("install-*"))


def test_process_death_releases_lease_without_deleting_its_lock_file(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    code = (
        "from pathlib import Path;from pymss_plugin_manager.leases import hold_runtime;"
        f"hold_runtime(Path({str(manager.context.root)!r}));print('ready',flush=True);input()"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    child = subprocess.Popen([sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeLockBusy):
            manager.disable("author.provider")
    finally:
        child.terminate()
        child.communicate(timeout=10)
    assert (manager.context.root.parent / ".cpu.execution.lock").exists()
    manager.disable("author.provider")
    assert not manager.list()["installed"][0]["enabled"]


def test_binding_identity_mismatch_preserves_existing_files(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    marker = manager.context.root / ".pymss-binding.json"
    previous = marker.read_bytes()
    with mock.patch.object(sys, "prefix", str(tmp_path / "other")):
        with pytest.raises(ValueError, match="PLUGIN_ENV_MISMATCH"):
            manager.uninstall("author.provider")
    assert marker.read_bytes() == previous
    assert manager.list()["installed"][0]["enabled"]


def test_published_marker_recovers_a_lost_inventory_commit(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    with mock.patch.object(inventory, "commit", side_effect=RuntimeError("interrupted inventory write")):
        with pytest.raises(RuntimeError, match="interrupted"):
            manager.install(str(source(tmp_path)))
    assert manager.list()["installed"][0]["prepared"]
    with manager.bound():
        _engine._recover(manager.context.root)
    assert manager.list()["installed"][0]["enabled"]


def test_invalid_action_does_not_create_a_store(tmp_path):
    root = tmp_path / "plugins/cpu"
    with pytest.raises(ValueError):
        PluginService(PluginContext(root)).manage("unknown")
    assert not root.exists()


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_probe_reports_failure_and_returns_nonzero_for_an_enabled_plugin(tmp_path, capsys, as_json):
    from pymss_plugin_manager.cli import run

    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    args = SimpleNamespace(plugin_store=str(manager.context.root), plugins_command="probe", json=as_json)
    assert run(args) == 0
    healthy = capsys.readouterr().out
    if as_json:
        assert json.loads(healthy)["reports"][0]["loaded"]
    else:
        assert "author.provider" in healthy and "OK" in healthy
    (manager.context.root / "installed/author.provider/__init__.py").write_text(
        'raise RuntimeError("damaged plugin")\n', encoding="utf-8")
    assert run(args) == 1
    failed = capsys.readouterr().out
    if as_json:
        assert not json.loads(failed)["reports"][0]["loaded"]
    else:
        assert "FAILED" in failed and "damaged plugin" in failed


@pytest.mark.parametrize("action", ["disable", "uninstall"])
def test_cli_probe_does_not_fail_for_a_retained_inactive_binding(tmp_path, capsys, action):
    from pymss_plugin_manager.cli import run

    cpu, cuda = service(tmp_path / "plugins/cpu"), service(tmp_path / "plugins/cuda")
    cpu.install(str(source(tmp_path)))
    cuda.prepare("author.provider")
    cpu.manage(action, name="author.provider")
    assert (cuda.context.root / "installed/author.provider").exists()
    args = SimpleNamespace(plugin_store=str(cuda.context.root), plugins_command="probe", json=True)
    assert run(args) == 0
    reports = json.loads(capsys.readouterr().out)["reports"]
    assert reports and not reports[0]["loaded"]


def test_check_updates_is_read_only_while_an_execution_uses_the_binding(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    previous = receipt.read_bytes()
    with binding_lease(manager.context.root):
        result = manager.manage("check_updates")
    assert result["updates"]["author.provider"]["up_to_date"]
    assert receipt.read_bytes() == previous


@pytest.mark.parametrize("ref", ["stable", "--upload-pack=untrusted", "../main", 123])
def test_catalog_ref_without_a_version_is_used_for_update_checks(tmp_path, ref):
    manager = PluginService(PluginContext(tmp_path / "plugins/cpu", registry_url="https://example.invalid/registry.json"),
                            service(tmp_path / "unused").policy)
    manager.install(str(source(tmp_path)))
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    url, commit = "https://github.com/example/plugins", "a" * 40
    metadata.update(defaultRepository=True, url=url, resolved_commit=commit)
    receipt.write_text(json.dumps(metadata), encoding="utf-8")
    entry = {"name": "author.provider", "url": url, "ref": ref, "subpath": "plugins/provider"}

    def remote(_url, _subpath, ref):
        return {"version": "1.0" if ref == "stable" else "2.0", "resolved_commit": commit if ref == "stable" else "b" * 40}

    with mock.patch("pymss.plugins.install.list_available", return_value=[entry]), \
         mock.patch("pymss.plugins.install._fetch_remote_meta", side_effect=remote) as fetch:
        result = manager.check_updates()
    if ref == "stable":
        assert result["author.provider"]["up_to_date"]
        fetch.assert_called_once_with(url, "plugins/provider", ref="stable")
    else:
        assert "Invalid plugin repository ref" in result["author.provider"]["error"]
        fetch.assert_not_called()


@pytest.mark.parametrize("host", ["github.com", "gitcode.com"])
@pytest.mark.parametrize("remote_commit", ["a" * 40, "b" * 40])
def test_immutable_catalog_commits_detect_same_version_updates_on_both_hosts(tmp_path, host, remote_commit):
    manager = PluginService(PluginContext(tmp_path / "plugins/cpu", registry_url="https://example.invalid/registry.json"),
                            service(tmp_path / "unused").policy)
    manager.install(str(source(tmp_path)))
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    url = f"https://{host}/pymss-project/pymss-plugins"
    metadata.update(defaultRepository=True, url=url, resolved_commit="a" * 40)
    receipt.write_text(json.dumps(metadata), encoding="utf-8")
    entry = {"name": "author.provider", "url": url, "version": "1.0", "ref": remote_commit}
    with mock.patch("pymss.plugins.install.list_available", return_value=[entry]), \
         mock.patch("pymss.plugins.install._fetch_remote_meta", side_effect=AssertionError("Already pinned")), \
         mock.patch("pymss.plugins.sources.resolve_github_commit", side_effect=AssertionError("Already pinned")):
        result = manager.check_updates()["author.provider"]
    assert result["remote_commit"] == remote_commit
    assert result["up_to_date"] == (remote_commit == "a" * 40)
    assert not result["content_changed"]
    assert result["status"] == ("latest" if remote_commit == "a" * 40 else "source_changed")


@pytest.mark.parametrize("ref", [None, "stable"])
def test_unpinned_non_github_catalog_resolves_a_reviewed_update_target(tmp_path, ref):
    manager = PluginService(PluginContext(tmp_path / "plugins/cpu", registry_url="https://example.invalid/registry.json"),
                            service(tmp_path / "unused").policy)
    manager.install(str(source(tmp_path)))
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    metadata.update(defaultRepository=True)
    receipt.write_text(json.dumps(metadata), encoding="utf-8")
    url, commit = "https://gitcode.com/example/plugins", "b" * 40
    entry = {"name": "author.provider", "url": url, "subpath": "plugins/provider", "version": "2.0", "ref": ref}
    with mock.patch("pymss.plugins.install.list_available", return_value=[entry]), \
         mock.patch("pymss.plugins.install._fetch_remote_meta", return_value={
             "version": "2.0", "resolved_commit": commit,
         }) as fetch:
        result = manager.check_updates()["author.provider"]
    fetch.assert_called_once_with(url, "plugins/provider", ref=ref)
    assert result["status"] == "update"
    assert result["target"]["candidate"] == {"version": "2.0", "commit": commit}


@pytest.mark.parametrize("remote_version, status", [
    ("2.0", "update"), ("0.9", "older"), ("1.0rc1", "older"), ("nightly", "incomparable"),
])
def test_update_checks_compare_versions_without_offering_downgrades(tmp_path, remote_version, status):
    from datetime import datetime

    package = source(tmp_path)
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(package))
    project = package / "pyproject.toml"
    project.write_text(project.read_text(encoding="utf-8").replace('version="1.0"', f'version="{remote_version}"'), encoding="utf-8")
    result = manager.check_updates()["author.provider"]
    assert result["status"] == status
    assert not result["up_to_date"]
    assert ("target" in result) == (status == "update")
    assert datetime.fromisoformat(result["checked_at"]).tzinfo is not None


def test_same_version_local_content_is_reviewed_and_published_with_dependencies_preserved(tmp_path):
    package = source(tmp_path)
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(package))
    installed = manager.context.root / "installed/author.provider"
    sidecar = installed / ".studio-deps"
    sidecar.mkdir()
    (sidecar / "dependency.txt").write_text("retained", encoding="utf-8")
    assert manager.check_updates()["author.provider"]["status"] == "latest"
    code = package / "__init__.py"
    code.write_text(code.read_text(encoding="utf-8") + "# updated content\n", encoding="utf-8")
    result = manager.check_updates()["author.provider"]
    assert result["status"] == "content"
    assert result["content_changed"]
    manager.update("author.provider", expectedUpdate=result)
    assert (installed / "__init__.py").read_bytes() == code.read_bytes()
    assert (sidecar / "dependency.txt").read_text(encoding="utf-8") == "retained"
    assert manager.check_updates()["author.provider"]["status"] == "latest"


@pytest.mark.parametrize("changed", ["installed", "source", "receipt"])
def test_reviewed_update_rejects_modified_installation_or_local_source(tmp_path, changed):
    package = source(tmp_path)
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(package))
    (package / "change.txt").write_text("checked", encoding="utf-8")
    result = manager.check_updates()["author.provider"]
    installed = manager.context.root / "installed/author.provider"
    receipt = installed / ".pymss-install.json"
    if changed == "receipt":
        meta = json.loads(receipt.read_text(encoding="utf-8"))
        meta["subpath"] = "changed"
        receipt.write_text(json.dumps(meta), encoding="utf-8")
    else:
        (installed if changed == "installed" else package).joinpath("change.txt").write_text("later edit", encoding="utf-8")
    previous = receipt.read_bytes()
    with pytest.raises(ValueError, match="changed since the update check"):
        manager.update("author.provider", expectedUpdate=result)
    assert receipt.read_bytes() == previous
    assert (installed / "change.txt").exists() == (changed == "installed")


def test_reviewed_local_source_is_revalidated_after_verification(tmp_path):
    package = source(tmp_path)
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(package))
    (package / "change.txt").write_text("checked", encoding="utf-8")
    result = manager.check_updates()["author.provider"]
    original_probe = manager.policy.probe

    def change_source(root, paths):
        (package / "change.txt").write_text("changed during verification", encoding="utf-8")
        return original_probe(root, paths)

    manager.policy.probe = change_source
    with pytest.raises(ValueError, match="Local plugin source changed"):
        manager.update("author.provider", expectedUpdate=result)
    assert not (manager.context.root / "installed/author.provider/change.txt").exists()


def test_reviewed_update_cannot_be_applied_to_a_different_runtime_binding(tmp_path):
    package = source(tmp_path)
    cpu, cuda = service(tmp_path / "plugins/cpu"), service(tmp_path / "plugins/cuda")
    cpu.install(str(package))
    cuda.prepare("author.provider")
    (package / "change.txt").write_text("checked", encoding="utf-8")
    result = cpu.check_updates()["author.provider"]
    with pytest.raises(ValueError, match="reviewed update is unavailable"):
        cuda.update("author.provider", expectedUpdate=result)
    assert not (cuda.context.root / "installed/author.provider/change.txt").exists()


def test_selective_update_check_retries_only_the_requested_plugin(tmp_path):
    package = source(tmp_path / "one")
    other = source(tmp_path / "two", plugin_id="author.other")
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(package))
    manager.install(str(other))
    (package / "pyproject.toml").unlink()
    results = manager.check_updates()
    assert results["author.provider"]["status"] == "error"
    assert results["author.provider"]["checked_at"]
    assert results["author.other"]["status"] == "latest"
    assert set(manager.manage("check_updates", name="author.other")["updates"]) == {"author.other"}
    with pytest.raises(ValueError, match="Invalid plugin id"):
        manager.manage("check_updates", name="../other")


@pytest.fixture
def reviewed_remote_update(tmp_path):
    manager = PluginService(PluginContext(tmp_path / "plugins/cpu", registry_url="https://example.invalid/registry.json"),
                            service(tmp_path / "unused").policy)
    package = source(tmp_path)
    manager.install(str(package))
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    meta = json.loads(receipt.read_text(encoding="utf-8"))
    url = "https://github.com/example/plugins"
    meta.update(source="url", defaultRepository=True, url=url, subpath="plugins/provider", resolved_commit="a" * 40)
    receipt.write_text(json.dumps(meta), encoding="utf-8")
    entry = {"name": "author.provider", "url": url, "version": "2.0", "ref": "main", "subpath": "plugins/provider"}
    with mock.patch("pymss.plugins.install.list_available", return_value=[entry]), \
         mock.patch("pymss.plugins.sources.resolve_github_commit", return_value="b" * 40):
        checked = manager.check_updates()["author.provider"]
    assert checked["status"] == "update"
    return manager, package, entry, checked


@pytest.fixture
def reviewed_official_update(tmp_path, monkeypatch):
    from pymss.plugins import install as installer, marketplace

    manager = PluginService(PluginContext(tmp_path / "plugins/cpu", registry_url=marketplace.GITHUB_REGISTRY),
                            service(tmp_path / "unused").policy)
    package = source(tmp_path)
    manager.install(str(package))
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    metadata.update(source="url", defaultRepository=True, url=marketplace.GITHUB_REPOSITORY,
                    subpath="plugins/provider", resolved_commit="a" * 40)
    receipt.write_text(json.dumps(metadata), encoding="utf-8")
    project = package / "pyproject.toml"
    project.write_text(project.read_text(encoding="utf-8").replace('version="1.0"', 'version="2.0"'), encoding="utf-8")
    state = {"provider": "gitcode", "commit": "b" * 40, "version": "2.0"}

    def probe(provider):
        repository = marketplace.GITHUB_REPOSITORY if provider == "github" else marketplace.GITCODE_REPOSITORY
        return {"author.provider": {"url": repository, "ref": state["commit"], "version": state["version"],
                                    "subpath": "plugins/provider"}}, {
            "provider": provider, "commit": state["commit"], "latencyMs": 1 if provider == state["provider"] else 10,
        }

    def acquire(url, destination, *, subpath, ref, temporary_root):
        assert url == marketplace.GITCODE_REPOSITORY
        assert subpath == "plugins/provider"
        assert ref == "b" * 40
        shutil.copytree(package, destination)
        return {"resolved_commit": ref}

    monkeypatch.setenv("PYMSS_PLUGINS_SOURCE", "auto")
    monkeypatch.setattr(installer, "_git_available", lambda: True)
    probe_mock, acquire_mock = mock.Mock(side_effect=probe), mock.Mock(side_effect=acquire)
    monkeypatch.setattr(marketplace, "_probe", probe_mock)
    monkeypatch.setattr(installer, "_clone", acquire_mock)
    checked = manager.check_updates()["author.provider"]
    assert checked["status"] == "update"
    assert checked["target"]["source"]["url"] == marketplace.GITCODE_REPOSITORY
    return manager, state, checked, probe_mock, acquire_mock


@pytest.mark.parametrize("change", ["mirror", "branch", "verification", "offline"])
def test_reviewed_official_update_keeps_its_snapshot_when_catalog_resolution_changes(reviewed_official_update, change):
    from pymss.plugins.sources import SourceError

    manager, state, checked, probe, acquire = reviewed_official_update
    if change == "mirror":
        state["provider"] = "github"
    elif change == "branch":
        state.update(commit="c" * 40, version="3.0")
    elif change == "offline":
        probe.side_effect = SourceError("Catalog is temporarily unavailable")
    else:
        original_probe = manager.policy.probe

        def advance_during_verification(root, paths):
            state.update(commit="c" * 40, version="3.0")
            return original_probe(root, paths)

        manager.policy.probe = advance_during_verification
    manager.update("author.provider", expectedUpdate=checked)
    acquire.assert_called_once()
    assert probe.call_count == 2
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["resolved_commit"] == "b" * 40
    assert manager.list()["installed"][0]["version"] == "2.0"
    if change in {"branch", "verification"}:
        next_update = manager.check_updates()["author.provider"]
        assert next_update["status"] == "update"
        assert next_update["target"]["candidate"] == {"version": "3.0", "commit": "c" * 40}


@pytest.mark.parametrize("change", ["registry", "local", "preference"])
def test_reviewed_official_update_rejects_changes_to_user_source_settings(reviewed_official_update, monkeypatch, change):
    from dataclasses import replace
    from pymss.plugins import marketplace

    manager, _state, checked, _probe, acquire = reviewed_official_update
    if change == "registry":
        manager = PluginService(replace(manager.context, registry_url=marketplace.GITCODE_REGISTRY), manager.policy)
    elif change == "local":
        manager = PluginService(replace(manager.context, local_source=manager.context.root / "other-source"), manager.policy)
    else:
        monkeypatch.setenv("PYMSS_PLUGINS_SOURCE", "github")
    with pytest.raises(ValueError, match="Plugin source changed"):
        manager.update("author.provider", expectedUpdate=checked)
    acquire.assert_not_called()


def test_reviewed_official_update_revalidates_source_settings_before_publication(reviewed_official_update, monkeypatch):
    manager, _state, checked, _probe, acquire = reviewed_official_update
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    previous = receipt.read_bytes()
    original_probe = manager.policy.probe

    def change_settings_during_verification(root, paths):
        monkeypatch.setenv("PYMSS_PLUGINS_SOURCE", "github")
        return original_probe(root, paths)

    manager.policy.probe = change_settings_during_verification
    with pytest.raises(ValueError, match="Plugin source changed"):
        manager.update("author.provider", expectedUpdate=checked)
    acquire.assert_called_once()
    assert receipt.read_bytes() == previous
    assert manager.list()["installed"][0]["version"] == "1.0"


@pytest.mark.parametrize("mismatch", [None, "commit", "version", "id"])
def test_reviewed_remote_update_pins_the_checked_commit_and_rejects_other_candidates(reviewed_remote_update, mismatch):
    from pymss.plugins.install import InstallResult

    manager, package, entry, checked = reviewed_remote_update
    receipt = manager.context.root / "installed/author.provider/.pymss-install.json"
    previous = receipt.read_bytes()
    project = package / "pyproject.toml"
    project.write_text(project.read_text(encoding="utf-8").replace('version="1.0"', 'version="2.0"'), encoding="utf-8")

    def acquire(target, directory, subpath, *, ref):
        assert target == entry["url"]
        assert subpath == entry["subpath"]
        assert ref == checked["target"]["candidate"]["commit"] == "b" * 40
        candidate = directory / "author.provider"
        shutil.copytree(package, candidate)
        candidate_project = candidate / "pyproject.toml"
        if mismatch == "version":
            candidate_project.write_text(candidate_project.read_text(encoding="utf-8").replace('version="2.0"', 'version="3.0"'), encoding="utf-8")
        elif mismatch == "id":
            candidate_project.write_text(candidate_project.read_text(encoding="utf-8").replace('id="author.provider"', 'id="author.other"'), encoding="utf-8")
        (candidate / ".pymss-install.json").write_text(json.dumps({
            "name": "author.provider", "version": "2.0", "source": "url", "url": entry["url"],
            "subpath": entry["subpath"], "resolved_commit": "c" * 40 if mismatch == "commit" else ref,
        }), encoding="utf-8")
        return InstallResult(name="author.provider", path=candidate, source="url", version="2.0")

    with mock.patch("pymss.plugins.install.list_available", return_value=[entry]), \
         mock.patch.object(_engine, "_acquire_source", side_effect=acquire):
        if mismatch:
            with pytest.raises(ValueError, match="does not match the reviewed update"):
                manager.update("author.provider", expectedUpdate=checked)
            assert receipt.read_bytes() == previous
        else:
            manager.update("author.provider", expectedUpdate=checked)
            installed = json.loads(receipt.read_text(encoding="utf-8"))
            assert installed["resolved_commit"] == "b" * 40
            assert installed["defaultRepository"]
            assert manager.list()["installed"][0]["version"] == "2.0"


@pytest.mark.parametrize("field, value", [
    ("url", "https://gitcode.com/example/plugins"), ("subpath", "plugins/other"),
    ("ref", "stable"), ("version", "3.0"),
])
def test_reviewed_remote_update_rejects_changed_catalog_configuration(reviewed_remote_update, field, value):
    manager, _package, entry, checked = reviewed_remote_update
    with mock.patch("pymss.plugins.install.list_available", return_value=[{**entry, field: value}]), \
         mock.patch.object(_engine, "_acquire_source") as acquire:
        with pytest.raises(ValueError, match="Plugin source changed"):
            manager.update("author.provider", expectedUpdate=checked)
    acquire.assert_not_called()


@pytest.mark.parametrize("ref", ["main", "b" * 40])
@pytest.mark.parametrize("field, value", [
    ("url", "https://gitcode.com/example/plugins"), ("subpath", "plugins/other"),
    ("ref", "stable"), ("version", "3.0"),
])
def test_official_catalog_third_party_entries_keep_strict_source_checks(reviewed_remote_update, ref, field, value):
    from dataclasses import replace
    from pymss.plugins import marketplace

    manager, _package, entry, _checked = reviewed_remote_update
    manager = PluginService(replace(manager.context, registry_url=marketplace.GITHUB_REGISTRY), manager.policy)
    entry = {**entry, "ref": ref}
    with mock.patch("pymss.plugins.install.list_available", return_value=[entry]), \
         mock.patch("pymss.plugins.sources.resolve_github_commit", return_value="b" * 40):
        checked = manager.check_updates()["author.provider"]
    assert checked["target"]["source"]["officialSelection"] is None
    with mock.patch("pymss.plugins.install.list_available", return_value=[{**entry, field: value}]), \
         mock.patch.object(_engine, "_acquire_source") as acquire:
        with pytest.raises(ValueError, match="Plugin source changed"):
            manager.update("author.provider", expectedUpdate=checked)
    acquire.assert_not_called()


def test_reviewed_remote_update_uses_the_pinned_commit_in_the_real_acquisition_path(reviewed_remote_update):
    manager, package, entry, checked = reviewed_remote_update
    project = package / "pyproject.toml"
    project.write_text(project.read_text(encoding="utf-8").replace('version="1.0"', 'version="2.0"'), encoding="utf-8")

    def clone(url, destination, *, subpath, ref, temporary_root):
        assert url == entry["url"]
        assert subpath == entry["subpath"]
        assert ref == "b" * 40
        shutil.copytree(package, destination)
        return {"resolved_commit": ref}

    with mock.patch("pymss.plugins.install.list_available", return_value=[entry]), \
         mock.patch("pymss.plugins.install._clone", side_effect=clone) as acquire:
        manager.update("author.provider", expectedUpdate=checked)
    acquire.assert_called_once()
    assert manager.list()["installed"][0]["version"] == "2.0"


def test_reviewed_update_rejects_a_changed_catalog_provider(reviewed_remote_update):
    from dataclasses import replace

    manager, _package, entry, checked = reviewed_remote_update
    selected = PluginService(replace(manager.context, registry_url="https://other.invalid/registry.json"), manager.policy)
    with mock.patch("pymss.plugins.install.list_available", return_value=[entry]), \
         mock.patch.object(_engine, "_acquire_source") as acquire:
        with pytest.raises(ValueError, match="Plugin source changed"):
            selected.update("author.provider", expectedUpdate=checked)
    acquire.assert_not_called()


@pytest.mark.parametrize("failure", ["catalog", "commit"])
def test_update_checks_reuse_source_failures_instead_of_repeating_network_requests(tmp_path, failure):
    manager = PluginService(PluginContext(tmp_path / "plugins/cpu", registry_url="https://example.invalid/registry.json"),
                            service(tmp_path / "unused").policy)
    packages = [source(tmp_path / name, plugin_id=f"author.{name}") for name in ["one", "two"]]
    entries = []
    for package in packages:
        manager.install(str(package))
        name = _engine._read_manifest(package)["id"]
        receipt = manager.context.root / "installed" / name / ".pymss-install.json"
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        metadata.update(defaultRepository=True)
        receipt.write_text(json.dumps(metadata), encoding="utf-8")
        entries.append({"name": name, "url": "https://github.com/example/plugins", "version": "1.0", "ref": "main"})
    with mock.patch("pymss.plugins.install.list_available", side_effect=ValueError("offline") if failure == "catalog" else None,
                    return_value=entries) as catalog, \
         mock.patch("pymss.plugins.sources.resolve_github_commit", side_effect=ValueError("offline")) as commit:
        results = manager.check_updates()
    assert {item["status"] for item in results.values()} == {"error"}
    catalog.assert_called_once()
    assert commit.call_count == (0 if failure == "catalog" else 1)


def test_cli_check_updates_renders_statuses_and_errors(tmp_path, capsys):
    from pymss_plugin_manager.cli import run

    updates = {
        "author.current": {"local_version": "1.0", "remote_version": "1.0", "up_to_date": True},
        "author.changed": {"local_version": "1.0", "remote_version": "2.0", "up_to_date": False},
        "author.failed": {"error": "Source is unavailable"},
        "author.older": {"local_version": "2.0", "remote_version": "1.0", "status": "older"},
        "author.unordered": {"local_version": "1.0", "remote_version": "nightly", "status": "incomparable"},
        "author.repository": {"status": "source_changed"},
        "author.content": {"status": "content", "remote_version": "1.0"},
    }
    with mock.patch("pymss_plugin_manager.cli.PluginService") as factory:
        factory.return_value.list.return_value = {"installed": []}
        factory.return_value.manage.return_value = {"updates": updates}
        assert run(SimpleNamespace(plugin_store=str(tmp_path / "cpu"), plugins_command="check_updates", json=False)) == 0
    output = capsys.readouterr().out
    assert "author.current: up to date (1.0)" in output
    assert "author.changed: update available (1.0 -> 2.0)" in output
    assert "author.failed: check failed: Source is unavailable" in output
    assert "author.older: source version is older (2.0 -> 1.0)" in output
    assert "author.unordered: versions cannot be compared (1.0 -> nightly)" in output
    assert "author.repository: source repository has new commits; plugin content is unconfirmed" in output
    assert "author.content: plugin content has changed (1.0)" in output


def test_bad_inventory_blocks_managed_code_without_losing_builtin_operations(tmp_path):
    from pymss.operations import create_runtime
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    (manager.context.root.parent / "inventory.json").write_text('{"schemaVersion":2}', encoding="utf-8")
    runtime = create_runtime()
    before = runtime.list_operations()
    reports = manager.load_into(runtime)
    assert not reports[0]["loaded"]
    assert "inventory" in reports[0]["error"].lower()
    assert runtime.list_operations() == before


def test_probe_cannot_execute_a_removed_plugin_retained_in_another_binding(tmp_path):
    from pymss_plugin_manager.service import probe
    package = source(tmp_path)
    sentinel = tmp_path / "setup-executed"
    (package / "__init__.py").write_text(
        f'from pathlib import Path\ndef setup(plugin):\n    Path({str(sentinel)!r}).write_text("executed")\n', encoding="utf-8")
    cpu, cuda = service(tmp_path / "plugins/cpu"), service(tmp_path / "plugins/cuda")
    cpu.install(str(package))
    cuda.prepare("author.provider")
    cpu.uninstall("author.provider")
    assert (cuda.context.root / "installed/author.provider").exists()
    reports = probe(cuda.context.root)
    assert not reports[0]["loaded"]
    assert not sentinel.exists()


def test_candidate_probes_do_not_leave_per_install_execution_lock_files(tmp_path):
    manager = PluginService(PluginContext(tmp_path / "plugins/cpu"))
    manager.install(str(source(tmp_path)))
    assert not list((manager.context.root / "transactions").glob("*.execution.lock"))


def test_cli_metadata_query_does_not_release_an_ambient_runtime_lease(tmp_path, capsys):
    from pymss.cli import main
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    manager.hold_runtime()
    assert main(["--plugin-store", str(manager.context.root), "plugins", "list", "--json"]) == 0
    with pytest.raises(RuntimeLockBusy):
        manager.disable("author.provider")


def test_another_host_retains_the_installations_local_catalog(tmp_path):
    package = source(tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"author.provider": {"url": "https://github.com/example/plugins", "subpath": "source"}}), encoding="utf-8")
    manager = service(tmp_path / "plugins/cpu")
    origin = PluginService(PluginContext(manager.context.root, registry_url=registry.as_uri(), local_source=tmp_path), manager.policy)
    origin.install("author.provider")
    project = package / "pyproject.toml"
    project.write_text(project.read_text(encoding="utf-8").replace('version="1.0"', 'version="2.0"'), encoding="utf-8")
    with mock.patch("pymss.plugins.install.list_available", side_effect=AssertionError("unexpected remote catalog")):
        manager.update("author.provider")
        assert manager.check_updates()["author.provider"]["up_to_date"]
    assert manager.list()["installed"][0]["version"] == "2.0"


def test_native_command_keeps_its_lease_until_descendant_cleanup(tmp_path):
    from pymss.operations import OperationContext
    from pymss.operations import _command_process
    from pymss_plugin_manager.leases import hold_runtime

    root = tmp_path / "plugins/cpu"
    hold_runtime(root)
    cleanup_entered, finish_cleanup = threading.Event(), threading.Event()
    target, method = (_command_process._WindowsJob, "close") if os.name == "nt" else (_command_process, "_terminate_posix_group")
    original_cleanup = getattr(target, method)

    def cleanup(*args, **kwargs):
        cleanup_entered.set()
        assert finish_cleanup.wait(10), "Descendant cleanup was not released"
        return original_cleanup(*args, **kwargs)

    native = "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])"
    with mock.patch.object(target, method, cleanup), ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(_command_process.run_command_process, [sys.executable, "-c", native],
                              cwd=tmp_path, context=OperationContext(), timeout=15)
        try:
            assert cleanup_entered.wait(10), "Native command did not reach cleanup"
            release_runtime()
            with pytest.raises(RuntimeLockBusy):
                with binding_lease(root, exclusive=True):
                    pass
        finally:
            finish_cleanup.set()
        running.result(timeout=15)
    with binding_lease(root, exclusive=True):
        pass


def test_native_gate_retains_lease_after_the_parent_releases_it(tmp_path):
    root = tmp_path / "plugins/cpu"
    started, released = tmp_path / "started", tmp_path / "released"
    native = f"from pathlib import Path;import time;Path({str(started)!r}).touch();time.sleep(30)"
    code = f'''
from pathlib import Path
import sys, time, threading
from pymss_plugin_manager.leases import hold_runtime, release_runtime
from pymss.operations import OperationContext
from pymss.operations._command_process import run_command_process
root, started, released = Path({str(root)!r}), Path({str(started)!r}), Path({str(released)!r})
hold_runtime(root)
def release_parent():
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    release_runtime()
    released.touch()
threading.Thread(target=release_parent, daemon=True).start()
run_command_process([sys.executable, '-c', {native!r}], cwd=Path.cwd(), context=OperationContext(), timeout=30)
'''
    child = subprocess.Popen([sys.executable, "-X", "utf8", "-c", code],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    try:
        deadline = time.monotonic() + 15
        while not released.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)

        assert started.exists() and released.exists()
        with pytest.raises(RuntimeLockBusy):
            with binding_lease(root, exclusive=True):
                pass
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)
    deadline = time.monotonic() + 10
    while True:
        try:
            with binding_lease(root, exclusive=True):
                break
        except RuntimeLockBusy:
            assert time.monotonic() < deadline, "Native gate retained a lease after its process tree exited"
            time.sleep(.02)


def test_a_broken_peer_does_not_poison_other_probe_results(tmp_path):
    from pymss_plugin_manager.service import probe
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    broken = manager.context.root / "installed/broken"
    broken.mkdir()
    (broken / "pyproject.toml").write_text('[tool.pymss.plugin]\nid="broken"\napi_version=1\n', encoding="utf-8")
    reports = {row["name"]: row for row in probe(manager.context.root)}
    assert reports["author.provider"]["loaded"]
    assert not reports["broken"]["loaded"]


def test_read_recovery_does_not_wait_for_its_own_live_runtime(tmp_path):
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(source(tmp_path)))
    manager.hold_runtime()
    with manager.bound(), mock.patch.object(_engine, "_recover", side_effect=AssertionError("read attempted recovery")):
        _engine._recover_for_read(manager.context.root)
    assert manager.list()["installed"][0]["enabled"]


def test_cached_cli_runtime_keeps_its_execution_lease(tmp_path, monkeypatch):
    from pymss.cli import main
    from pymss.plugins import loader, get_registry
    import pymss.operations as operations
    package = source(tmp_path)
    (package / "__init__.py").write_text(
        'def run(args):\n    return 0\ndef setup(plugin):\n'
        '    plugin.register_cli("managed-cache-lease", run)\n', encoding="utf-8")
    monkeypatch.setattr(loader, "_LAST_REPORT", None)
    monkeypatch.setattr(loader, "_LAST_CONTEXT", None)
    monkeypatch.setattr(operations, "_DEFAULT_RUNTIME", None)
    monkeypatch.setattr(get_registry(), "cli_commands", list(get_registry().cli_commands))
    manager = service(tmp_path / "plugins/cpu")
    manager.install(str(package))
    arguments = ["--plugin-store", str(manager.context.root), "managed-cache-lease"]
    assert main(arguments) == 0
    assert main(arguments) == 0
    with pytest.raises(RuntimeLockBusy):
        manager.disable("author.provider")


def test_enabling_another_binding_does_not_mark_a_disabled_copy_prepared(tmp_path):
    cpu, cuda = service(tmp_path / "plugins/cpu"), service(tmp_path / "plugins/cuda")
    cpu.install(str(source(tmp_path)))
    cpu.disable("author.provider")
    cuda.enable("author.provider")
    row = cpu.list()["installed"][0]
    assert row["enabled"] and not row["prepared"]
    cpu.prepare("author.provider")
    assert cpu.list()["installed"][0]["prepared"]
