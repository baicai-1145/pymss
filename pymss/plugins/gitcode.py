"""Acquire public GitCode plugin files over HTTPS without a Git executable."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import copy
import hashlib
from http.client import HTTPException
import io
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .sources import SourceError, _relative_parts, _COMMIT_RE

_API_HOST = "api.gitcode.com"
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_ENTRIES = 50_000


def repository(url: str) -> tuple[str, str] | None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "gitcode.com":
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.port not in {None, 443}:
        raise SourceError("Use an HTTPS GitCode repository URL without credentials, query, or fragment.")
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2:
        raise SourceError("Use the repository URL https://gitcode.com/OWNER/REPOSITORY.")
    owner, repo = parts
    repo = repo.removesuffix(".git")
    if any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part) or part in {".", ".."} for part in (owner, repo)):
        raise SourceError("Invalid GitCode repository owner or name.")
    return owner, repo


def _api(url: str) -> str:
    parts = repository(url)
    if parts is None:
        raise SourceError("This source is not a public HTTPS GitCode repository URL.")
    return f"https://{_API_HOST}/api/v5/repos/{parts[0]}/{parts[1]}"


def _check_cancelled() -> None:
    from pymss_plugin_manager.context import current, check_cancelled
    try:
        current()
    except RuntimeError:
        return  # The legacy standalone installer has no management context.
    check_cancelled()


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != _API_HOST or parsed.username or parsed.password
            or parsed.port not in {None, 443}):
        raise SourceError("GitCode redirected outside its download host.")


class _SourceRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        _validate_url(newurl)
        return super().redirect_request(request, response, code, message, headers, newurl)


def _open(request):
    # Retain the host's configured proxy/TLS handlers, without changing its opener.
    installed = urllib.request._opener
    handlers = [copy(handler) for handler in installed.handlers
                if not isinstance(handler, urllib.request.HTTPRedirectHandler)] if installed else []
    return urllib.request.build_opener(*handlers, _SourceRedirect()).open(request, timeout=30)


def _download(url: str, output, *, limit: int, stopped: threading.Event | None = None, consume=None) -> dict:
    _validate_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "pymss", "Accept": "*/*", "Accept-Encoding": "identity"})
    started = time.monotonic()
    try:
        _check_cancelled()
        with _open(request) as response:
            _validate_url(response.geturl())
            headers = dict(response.headers)
            declared = response.headers.get("Content-Length")
            if declared and (int(declared) < 0 or int(declared) > limit):
                raise SourceError("GitCode response exceeds the download size limit.")
            size = 0
            while True:
                _check_cancelled()
                if stopped and stopped.is_set():
                    raise InterruptedError("Plugin source download stopped")
                if time.monotonic() - started > 180:
                    raise SourceError("GitCode download exceeded the allowed time.")
                chunk = response.read(min(64 * 1024, limit - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise SourceError("GitCode response exceeds the download size limit.")
                if consume:
                    consume(len(chunk))
                output.write(chunk)
            if declared is not None and size != int(declared):
                raise SourceError("GitCode returned an incomplete download.")
            return headers
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        if code == 429 or code == 403 and error.headers.get("X-RateLimit-Remaining") == "0":
            raise SourceError("GitCode request rate limit reached; retry later.") from error
        raise SourceError(f"GitCode request failed with HTTP {code}.") from error
    except InterruptedError:
        raise
    except (OSError, ValueError, HTTPException) as error:
        raise SourceError(f"GitCode download failed: {error}") from error


def _json(url: str) -> tuple[object, dict]:
    output = io.BytesIO()
    headers = _download(url, output, limit=_MAX_METADATA_BYTES)
    try:
        return json.loads(output.getvalue()), {key.lower(): value for key, value in headers.items()}
    except ValueError as error:
        raise SourceError("GitCode returned invalid repository metadata.") from error


def resolve_commit(url: str, ref: str | None = None) -> str:
    api = _api(url)
    if ref and _COMMIT_RE.fullmatch(ref.lower()):
        return ref.lower()
    if not ref:
        metadata, _ = _json(api)
        ref = metadata.get("default_branch") if isinstance(metadata, dict) else None
        if not isinstance(ref, str) or not ref:
            raise SourceError("GitCode did not return the repository's default branch.")
    result, _ = _json(f"{api}/commits/{urllib.parse.quote(ref, safe='')}")
    commit = result.get("sha") if isinstance(result, dict) else None
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise SourceError("GitCode did not return a full commit SHA.")
    return commit


def read_file(url: str, path: str, *, commit: str) -> bytes:
    _relative_parts(path)
    if not _COMMIT_RE.fullmatch(commit):
        raise SourceError("A full commit SHA is required to read repository content.")
    output = io.BytesIO()
    _download(f"{_api(url)}/raw/{urllib.parse.quote(path, safe='/')}?ref={commit}", output, limit=_MAX_METADATA_BYTES)
    return output.getvalue()


def _files(api: str, commit: str, subpath: str | None) -> list[tuple[dict, tuple[str, ...]]]:
    selected = _relative_parts(subpath) if subpath else ()
    files, paths, explicit_paths = [], {}, set()
    page, received, expected, pages = 1, 0, None, None
    while True:
        query = urllib.parse.urlencode({"recursive": 1, "per_page": 100, "page": page,
                                        **({"file_path": "/".join(selected)} if selected else {})})
        result, headers = _json(f"{api}/git/trees/{commit}?{query}")
        entries = result.get("tree") if isinstance(result, dict) else None
        if not isinstance(entries, list) or result.get("truncated"):
            raise SourceError("GitCode returned an incomplete repository tree.")
        if page == 1:
            try:
                expected = int(headers["total_count"]) if "total_count" in headers else None
                pages = int(headers["total_page"]) if "total_page" in headers else None
            except (ValueError, TypeError) as error:
                raise SourceError("GitCode returned invalid tree pagination.") from error
            if (expected is not None and not 0 <= expected <= _MAX_ENTRIES
                    or pages is not None and not 1 <= pages <= (_MAX_ENTRIES // 100 + 1)):
                raise SourceError("GitCode repository tree exceeds its entry limit.")
        received += len(entries)
        if received > _MAX_ENTRIES or len(entries) > 100:
            raise SourceError("GitCode repository tree exceeds its entry limit.")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise SourceError("GitCode returned an invalid repository path.")
            parts = _relative_parts(entry["path"])
            if parts[:len(selected)] != selected:
                raise SourceError("GitCode returned a path outside the selected plugin directory.")
            relative = parts[len(selected):]
            kind, mode = entry.get("type"), entry.get("mode")
            if not (kind == "tree" and mode == "040000" or kind == "blob" and mode in {"100644", "100755"}):
                raise SourceError("GitCode plugin contains a link, submodule, or special file.")
            if not isinstance(entry.get("sha"), str) or not _COMMIT_RE.fullmatch(entry["sha"]):
                raise SourceError("GitCode returned an invalid file hash.")
            for length in range(1, len(relative) + 1):
                path = relative[:length]
                key = "/".join(path).casefold()
                expected_kind = kind if length == len(relative) else "tree"
                previous = paths.get(key)
                if previous and previous != (path, expected_kind):
                    raise SourceError("GitCode repository tree contains duplicate or conflicting paths.")
                explicit = length == len(relative)
                if explicit and key in explicit_paths:
                    raise SourceError("GitCode repository tree contains duplicate or conflicting paths.")
                if explicit:
                    explicit_paths.add(key)
                paths[key] = (path, expected_kind)
            if kind == "blob":
                if not relative:
                    raise SourceError("Plugin source must be a directory.")
                files.append((entry, relative))
        if pages is not None and page >= pages or pages is None and len(entries) < 100:
            break
        if not entries:
            raise SourceError("GitCode returned an incomplete repository tree.")
        page += 1
    if expected is not None and received != expected:
        raise SourceError("GitCode returned an incomplete repository tree.")
    if not files:
        raise SourceError("Plugin directory is empty or absent from the GitCode repository.")
    return files


def download_source(url: str, dest: Path, *, subpath: str | None = None, ref: str | None = None,
                    temporary_root: Path | None = None) -> dict:
    if dest.exists():
        raise SourceError(f"Destination already exists: {dest}.")
    api = _api(url)
    commit = resolve_commit(url, ref)
    files = _files(api, commit, subpath)
    dest.mkdir(parents=True)
    stopped, lock = threading.Event(), threading.Lock()
    total = 0

    def consume(size):
        nonlocal total
        with lock:
            total += size
            if total > _MAX_TOTAL_BYTES:
                raise SourceError("GitCode plugin exceeds the total download size limit.")

    def download(entry, relative):
        _check_cancelled()
        target = dest.joinpath(*relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            _download(f"{api}/raw/{urllib.parse.quote(entry['path'], safe='/')}?ref={commit}", output,
                      limit=_MAX_FILE_BYTES, stopped=stopped, consume=consume)
        digest = hashlib.sha1(b"blob " + str(target.stat().st_size).encode() + b"\0")
        with target.open("rb") as content:
            for chunk in iter(lambda: content.read(64 * 1024), b""):
                _check_cancelled()
                digest.update(chunk)
        if digest.hexdigest() != entry["sha"]:
            raise SourceError(f"GitCode file content verification failed: {entry['path']}")
        if entry["mode"] == "100755":
            target.chmod(target.stat().st_mode | 0o111)

    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(copy_context().run, download, entry, relative) for entry, relative in files]
        try:
            for future in as_completed(pending):
                future.result()
        except BaseException:
            stopped.set()
            for future in pending:
                future.cancel()
            raise
    _check_cancelled()
    return {"url": url, "ref": ref, "resolved_commit": commit}
