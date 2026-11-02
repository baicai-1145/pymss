from __future__ import annotations

import json
import os
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
         mock.patch("pymss.plugins.sources.resolve_github_commit", side_effect=AssertionError("Already pinned")):
        result = manager.check_updates()["author.provider"]
    assert result["remote_commit"] == remote_commit
    assert result["up_to_date"] == (remote_commit == "a" * 40)
    assert result["content_changed"] == (remote_commit != "a" * 40)


def test_cli_check_updates_renders_statuses_and_errors(tmp_path, capsys):
    from pymss_plugin_manager.cli import run

    updates = {
        "author.current": {"local_version": "1.0", "remote_version": "1.0", "up_to_date": True},
        "author.changed": {"local_version": "1.0", "remote_version": "2.0", "up_to_date": False},
        "author.failed": {"error": "Source is unavailable"},
    }
    with mock.patch("pymss_plugin_manager.cli.PluginService") as factory:
        factory.return_value.list.return_value = {"installed": []}
        factory.return_value.manage.return_value = {"updates": updates}
        assert run(SimpleNamespace(plugin_store=str(tmp_path / "cpu"), plugins_command="check_updates", json=False)) == 0
    output = capsys.readouterr().out
    assert "author.current: up to date (1.0)" in output
    assert "author.changed: update available (1.0 -> 2.0)" in output
    assert "author.failed: check failed: Source is unavailable" in output


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
