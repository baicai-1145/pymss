"""Select an available official catalog and pin its packages to that snapshot."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from http.client import HTTPException
import json
import os
import re
import time
import urllib.parse
import urllib.request

from .sources import SourceError


GITHUB_REPOSITORY = "https://github.com/pymss-project/pymss-plugins"
GITCODE_REPOSITORY = "https://gitcode.com/pymss-project/pymss-plugins"
GITHUB_REGISTRY = "https://raw.githubusercontent.com/pymss-project/pymss-plugins/main/registry.json"
GITCODE_REGISTRY = "https://api.gitcode.com/api/v5/repos/pymss-project/pymss-plugins/raw/registry.json?ref=main"
_MAX_METADATA_BYTES = 1024 * 1024
_SOURCE: ContextVar[dict | None] = ContextVar("pymss_official_plugin_source", default=None)
HTTP_PROVIDERS = ("github", "gitcode")


def source_capabilities() -> dict:
    return {"httpProviders": list(HTTP_PROVIDERS)}


def source_info() -> dict | None:
    value = _SOURCE.get()
    return dict(value) if value else None


def is_official_registry(url: str) -> bool:
    return url in {GITHUB_REGISTRY, GITCODE_REGISTRY}


def _json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "pymss", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        resolved = urllib.parse.urlsplit(response.url)
        if resolved.scheme != "https" or resolved.hostname not in {
            "api.github.com", "raw.githubusercontent.com", "api.gitcode.com",
        }:
            raise SourceError("Official plugin catalog redirected outside its source hosts")
        content = response.read(_MAX_METADATA_BYTES + 1)
    if len(content) > _MAX_METADATA_BYTES:
        raise SourceError("Official plugin catalog metadata exceeds its size limit")
    return json.loads(content)


def _probe(provider: str) -> tuple[dict, dict]:
    started = time.monotonic()
    api = "https://api.github.com/repos" if provider == "github" else "https://api.gitcode.com/api/v5/repos"
    revision = _json(f"{api}/pymss-project/pymss-plugins/commits/main")
    commit = revision.get("sha") if isinstance(revision, dict) else None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SourceError("Official plugin source did not return a full commit ID")
    if provider == "github":
        registry = f"https://raw.githubusercontent.com/pymss-project/pymss-plugins/{commit}/registry.json"
        repository = GITHUB_REPOSITORY
    else:
        registry = GITCODE_REGISTRY.removesuffix("main") + commit
        repository = GITCODE_REPOSITORY
    catalog = _json(registry)
    if not isinstance(catalog, dict) or not any(not key.startswith("_") for key in catalog):
        raise SourceError("Official plugin catalog is empty or invalid")
    snapshot = {}
    for name, entry in catalog.items():
        if name.startswith("_"):
            snapshot[name] = entry
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise SourceError(f"Invalid official plugin catalog entry: {name}")
        entry = dict(entry)
        if entry["url"].rstrip("/").removesuffix(".git") in {GITHUB_REPOSITORY, GITCODE_REPOSITORY}:
            entry.update(url=repository, ref=commit)
        snapshot[name] = entry
    return snapshot, {"provider": provider, "repositoryUrl": repository, "commit": commit,
                      "latencyMs": round((time.monotonic() - started) * 1000)}


def fetch_official_catalog(url: str) -> dict:
    """Resolve an explicit provider, or compare usable providers for automatic mode."""
    _SOURCE.set(None)
    if not is_official_registry(url):
        raise SourceError("Unsupported official plugin registry URL")
    preference = "gitcode" if url == GITCODE_REGISTRY else os.environ.get("PYMSS_PLUGINS_SOURCE", "auto")
    if preference not in {"auto", "github", "gitcode"}:
        raise SourceError("Plugin source must be auto, github, or gitcode")
    providers = list(HTTP_PROVIDERS) if preference == "auto" else [preference]
    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        requests = {pool.submit(_probe, provider): provider for provider in providers}
        for request in as_completed(requests):
            provider = requests[request]
            try:
                results[provider] = request.result()
            except (SourceError, OSError, ValueError, HTTPException) as error:
                errors.append(f"{provider}: {error}")
    if not results:
        raise SourceError("Official plugin sources are unavailable: " + "; ".join(sorted(errors)))
    # A reachable canonical source takes precedence over a mirror that has not
    # synchronized that revision, even when the older mirror responds faster.
    if len(results) == 2 and results["github"][1]["commit"] != results["gitcode"][1]["commit"]:
        catalog, info = results["github"]
    else:
        catalog, info = min(results.values(), key=lambda result: (result[1]["latencyMs"], result[1]["provider"]))
    _SOURCE.set({**info, "preference": preference, **source_capabilities()})
    return catalog
