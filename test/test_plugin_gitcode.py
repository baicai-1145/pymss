"""Offline coverage for public GitCode HTTP acquisition."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import threading
from unittest.mock import Mock
import urllib.error

import pytest

from pymss.plugins import gitcode, install as installer
from pymss.plugins.sources import SourceError
from pymss_plugin_manager.context import PluginContext, PluginPolicy, use

URL = "https://gitcode.com/owner/plugins"
COMMIT = "a" * 40
MANIFEST = b'[project]\nname="plugin-example"\nversion="1.0.0"\n[tool.pymss.plugin]\nid="test.example"\napi_version=2\n'


def blob(path, body=b"data", mode="100644"):
    return {"type": "blob", "mode": mode, "path": path,
            "sha": hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(gitcode, "_open", Mock(side_effect=AssertionError("Unexpected network request")))


@contextmanager
def fake_source(monkeypatch, entries=None, bodies=None):
    bodies = bodies or {"plugins/example/pyproject.toml": MANIFEST,
                        "plugins/example/__init__.py": b"def setup(plugin): pass\n",
                        "plugins/example/bin/native": b"\x00\xff\xfe\x00\r\n"}
    entries = entries or [blob(path, body, "100755" if path.endswith("native") else "100644") for path, body in bodies.items()]
    requests = []

    def download(url, output, *, limit, stopped=None, consume=None):
        requests.append(url)
        if "/git/trees/" in url:
            output.write(json.dumps({"tree": entries}).encode())
            return {"total_count": str(len(entries)), "total_page": "1"}
        path = url.split("/raw/", 1)[1].split("?", 1)[0]
        body = bodies[path]
        if consume:
            consume(len(body))
        assert len(body) <= limit
        output.write(body)
        return {}

    monkeypatch.setattr(gitcode, "_download", download)
    yield requests, bodies


@pytest.mark.parametrize("available", [False, True])
def test_public_install_never_uses_git_and_records_commit(tmp_path, monkeypatch, available):
    run = Mock(side_effect=AssertionError("Unexpected Git execution"))
    monkeypatch.setattr(installer.subprocess, "run", run)
    monkeypatch.setattr(installer, "_git_available", lambda: available)
    monkeypatch.setattr(installer, "_clear_macos_attributes", lambda path: None)
    with fake_source(monkeypatch) as (requests, bodies):
        installed = installer.install(URL, plugins_dir=tmp_path / "installed", subpath="plugins/example", ref=COMMIT, no_deps=True)
    run.assert_not_called()
    assert installed.name == "test.example"
    receipt = json.loads((installed.path / ".pymss-install.json").read_text())
    assert receipt["resolved_commit"] == COMMIT
    assert all(COMMIT in request for request in requests)
    for path, body in bodies.items():
        assert (installed.path / path.removeprefix("plugins/example/")).read_bytes() == body
    if installer.sys.platform != "win32":
        assert (installed.path / "bin/native").stat().st_mode & 0o111


def test_metadata_reads_only_manifest_without_git_or_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.subprocess, "run", Mock(side_effect=AssertionError("Unexpected Git")))
    with fake_source(monkeypatch) as (requests, _):
        metadata = installer._fetch_remote_meta(URL, "plugins/example", COMMIT)
    assert metadata["version"] == "1.0.0" and metadata["resolved_commit"] == COMMIT
    assert len(requests) == 1 and "/raw/plugins/example/pyproject.toml" in requests[0]


@pytest.mark.parametrize("ref", [None, "release/v1"])
def test_resolves_default_branch_and_named_refs(monkeypatch, ref):
    requests = []
    def read(url):
        requests.append(url)
        return ({"sha": COMMIT} if "/commits/" in url else {"default_branch": "trunk"}), {}
    monkeypatch.setattr(gitcode, "_json", read)
    assert gitcode.resolve_commit(URL, ref) == COMMIT
    assert requests[-1].endswith("/commits/" + ("trunk" if ref is None else "release%2Fv1"))


def test_pinned_commit_does_not_make_metadata_request(monkeypatch):
    monkeypatch.setattr(gitcode, "_json", Mock(side_effect=AssertionError("Already pinned")))
    assert gitcode.resolve_commit(URL, COMMIT.upper()) == COMMIT


@pytest.mark.parametrize("path", ["plugins/example/../escape", "plugins/example/a\\b", "plugins/example/CON", "other/file"])
def test_unsafe_or_outside_paths_are_rejected_before_writing(tmp_path, monkeypatch, path):
    with fake_source(monkeypatch, [blob(path)]), pytest.raises(SourceError):
        gitcode.download_source(URL, tmp_path / "plugin", subpath="plugins/example", ref=COMMIT)
    assert not (tmp_path / "plugin").exists()


@pytest.mark.parametrize("entries", [
    [blob("plugins/example/file"), blob("plugins/example/File")],
    [blob("plugins/example/file"), blob("plugins/example/file/child")],
    [blob("plugins/example/Upper/a"), blob("plugins/example/upper/b")],
    [blob("plugins/example/link", mode="120000")],
    [{**blob("plugins/example/submodule"), "mode": "160000", "type": "commit"}],
    [{**blob("plugins/example/file"), "sha": "invalid"}],
])
def test_conflicting_paths_special_files_and_invalid_hashes_are_rejected(tmp_path, monkeypatch, entries):
    with fake_source(monkeypatch, entries), pytest.raises(SourceError):
        gitcode.download_source(URL, tmp_path / "plugin", subpath="plugins/example", ref=COMMIT)
    assert not (tmp_path / "plugin").exists()


def test_explicit_directory_entries_and_children_are_accepted(tmp_path, monkeypatch):
    entries = [{"type": "tree", "mode": "040000", "path": "plugins/example/bin", "sha": "b" * 40},
               blob("plugins/example/bin/native")]
    with fake_source(monkeypatch, entries, {"plugins/example/bin/native": b"data"}):
        gitcode.download_source(URL, tmp_path / "plugin", subpath="plugins/example", ref=COMMIT)
    assert (tmp_path / "plugin/bin/native").read_bytes() == b"data"


@pytest.mark.parametrize("failure", ["hash", "size", "cancel"])
def test_failed_install_cleans_staging_and_does_not_use_git(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(installer, "_git_available", Mock(side_effect=AssertionError("No fallback")))
    entries = [blob("plugins/example/pyproject.toml", b"different" if failure == "hash" else MANIFEST)]
    if failure == "size":
        monkeypatch.setattr(gitcode, "_MAX_TOTAL_BYTES", 1)
    if failure == "cancel":
        monkeypatch.setattr(gitcode, "_check_cancelled", Mock(side_effect=InterruptedError("cancelled")))
    with fake_source(monkeypatch, entries, {"plugins/example/pyproject.toml": MANIFEST}), pytest.raises((installer.InstallError, InterruptedError)):
        installer.install(URL, plugins_dir=tmp_path / "installed", subpath="plugins/example", ref=COMMIT, no_deps=True)
    assert list((tmp_path / "installed").iterdir()) == []


def test_cancellation_context_reaches_download_threads(tmp_path, monkeypatch):
    main_thread = threading.get_ident()
    def cancelled():
        return threading.get_ident() != main_thread
    context = PluginContext(tmp_path / "cpu")
    with fake_source(monkeypatch), use(context, PluginPolicy(cancelled=cancelled)), pytest.raises(InterruptedError):
        gitcode.download_source(URL, tmp_path / "plugin", subpath="plugins/example", ref=COMMIT)


def test_tree_pagination_is_complete_and_bounded(monkeypatch):
    seen = []
    def read(url):
        seen.append(url)
        start = 100 if "page=2" in url else 0
        count = 1 if start else 100
        return {"tree": [blob(f"plugins/example/file{index}") for index in range(start, start + count)]}, {"total_count": "101", "total_page": "2"}
    monkeypatch.setattr(gitcode, "_json", read)
    assert len(gitcode._files(gitcode._api(URL), COMMIT, "plugins/example")) == 101
    assert len(seen) == 2 and "page=2" in seen[-1]


@pytest.mark.parametrize("body,headers", [
    ({"tree": [], "truncated": True}, {}),
    ({"tree": [blob("file")]}, {"total_count": "2", "total_page": "1"}),
    ({"tree": []}, {"total_page": "9999"}),
    ({"tree": []}, {"total_page": "bad"}),
])
def test_incomplete_or_unbounded_trees_fail(monkeypatch, body, headers):
    monkeypatch.setattr(gitcode, "_json", lambda url: (body, headers))
    with pytest.raises(SourceError):
        gitcode._files(gitcode._api(URL), COMMIT, None)


@pytest.mark.parametrize("data,declared,limit,expected", [
    (b"123", "3", 3, None), (b"12", "3", 3, "incomplete"),
    (b"1234", None, 3, "size limit"), (b"1", "4", 3, "size limit"),
])
def test_stream_limits_and_incomplete_content_close_response(monkeypatch, data, declared, limit, expected):
    class Response(io.BytesIO):
        headers = {} if declared is None else {"Content-Length": declared}
        def geturl(self): return "https://api.gitcode.com/file"
    response = Response(data)
    monkeypatch.setattr(gitcode, "_open", lambda request: response)
    if expected:
        with pytest.raises(SourceError, match=expected):
            gitcode._download("https://api.gitcode.com/file", io.BytesIO(), limit=limit)
    else:
        output = io.BytesIO()
        gitcode._download("https://api.gitcode.com/file", output, limit=limit)
        assert output.getvalue() == data
    assert response.closed


@pytest.mark.parametrize("url", ["http://api.gitcode.com/file", "https://untrusted.example/file", "https://user@api.gitcode.com/file"])
def test_redirects_outside_source_are_rejected_before_following(url):
    with pytest.raises(SourceError):
        gitcode._SourceRedirect().redirect_request(None, None, 302, "redirect", {}, url)


@pytest.mark.parametrize("code", [401, 403, 404, 429, 503])
def test_http_failures_report_once_without_transport_fallback(monkeypatch, code):
    error = urllib.error.HTTPError("https://api.gitcode.com/file", code, "failed", {}, io.BytesIO())
    request = Mock(side_effect=error)
    monkeypatch.setattr(gitcode, "_open", request)
    with pytest.raises(SourceError, match="rate limit" if code == 429 else str(code)):
        gitcode._download("https://api.gitcode.com/file", io.BytesIO(), limit=10)
    assert request.call_count == 1


def test_proxy_configuration_is_retained_without_changing_installed_opener(monkeypatch):
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": "http://proxy.invalid:8080"}))
    monkeypatch.setattr(urllib.request, "_opener", opener)
    original = {id(handler): handler.parent for handler in opener.handlers}
    build = Mock(return_value=Mock(open=Mock(return_value="response")))
    monkeypatch.setattr(urllib.request, "build_opener", build)
    # The fixture blocks _open by default; exercise its original implementation.
    import importlib.util
    spec = importlib.util.spec_from_file_location("pymss.plugins._gitcode_opener_test", Path(gitcode.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._open(urllib.request.Request("https://api.gitcode.com/file")) == "response"
    proxies = [handler.proxies for handler in build.call_args.args if isinstance(handler, urllib.request.ProxyHandler)]
    assert proxies == [{"https": "http://proxy.invalid:8080"}]
    assert all(handler.parent is original[id(handler)] for handler in opener.handlers)
