from __future__ import annotations

import io
from http.client import HTTPException
from unittest import mock

import pytest

from pymss.plugins import marketplace
from pymss.plugins import install as install_mod
from pymss.plugins.sources import SourceError


COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def clear_source(monkeypatch):
    monkeypatch.delenv("PYMSS_PLUGINS_SOURCE", raising=False)
    monkeypatch.setattr(install_mod, "_git_available", lambda: True)
    marketplace._SOURCE.set(None)


def result(provider, latency=10, commit=COMMIT):
    return {"pymss.example": {}}, {"provider": provider, "commit": commit, "latencyMs": latency}


def test_auto_selects_the_fastest_usable_source(monkeypatch):
    monkeypatch.setattr(marketplace, "_probe", lambda provider: result(provider, 20 if provider == "github" else 5))
    marketplace.fetch_official_catalog(marketplace.GITHUB_REGISTRY)
    assert marketplace.source_info()["provider"] == "gitcode"


def test_auto_rejects_an_outdated_mirror_even_when_it_is_faster(monkeypatch):
    monkeypatch.setattr(marketplace, "_probe", lambda provider: result(provider, 20 if provider == "github" else 5,
                                                                     COMMIT if provider == "github" else "b" * 40))
    marketplace.fetch_official_catalog(marketplace.GITHUB_REGISTRY)
    assert marketplace.source_info()["provider"] == "github"


@pytest.mark.parametrize("error", [OSError("unavailable"), ValueError("invalid JSON"), HTTPException("incomplete response")])
def test_auto_uses_the_other_source_when_a_catalog_fails(monkeypatch, error):
    def probe(provider):
        if provider == "github":
            raise error
        return result(provider)
    monkeypatch.setattr(marketplace, "_probe", probe)
    marketplace.fetch_official_catalog(marketplace.GITHUB_REGISTRY)
    assert marketplace.source_info()["provider"] == "gitcode"


def test_explicit_source_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv("PYMSS_PLUGINS_SOURCE", "github")
    probe = mock.Mock(side_effect=OSError("network unavailable"))
    monkeypatch.setattr(marketplace, "_probe", probe)
    with pytest.raises(SourceError, match="network unavailable"):
        marketplace.fetch_official_catalog(marketplace.GITHUB_REGISTRY)
    probe.assert_called_once_with("github")
    assert marketplace.source_info() is None


def test_gitcode_is_available_without_git(monkeypatch):
    monkeypatch.setattr(install_mod, "_git_available", lambda: False)
    probe = mock.Mock(side_effect=lambda provider: result(provider))
    monkeypatch.setattr(marketplace, "_probe", probe)
    marketplace.fetch_official_catalog(marketplace.GITHUB_REGISTRY)
    assert {call.args[0] for call in probe.call_args_list} == {"github", "gitcode"}
    assert marketplace.source_info()["httpProviders"] == ["github", "gitcode"]
    probe.reset_mock()
    marketplace.fetch_official_catalog(marketplace.GITCODE_REGISTRY)
    probe.assert_called_once_with("gitcode")


@pytest.mark.parametrize("provider", ["github", "gitcode"])
def test_catalog_and_package_ref_share_the_same_commit(monkeypatch, provider):
    catalog = {
        "_comment": "official catalog",
        "pymss.example": {"url": marketplace.GITHUB_REPOSITORY, "ref": "main", "version": "1.0"},
        "external.plugin": {"url": "https://example.com/plugins", "ref": "stable"},
    }
    requests = []
    def read(url):
        requests.append(url)
        return {"sha": COMMIT} if "/commits/" in url else catalog
    monkeypatch.setattr(marketplace, "_json", read)
    snapshot, info = marketplace._probe(provider)
    assert COMMIT in requests[1]
    assert snapshot["pymss.example"]["ref"] == info["commit"] == COMMIT
    expected = marketplace.GITHUB_REPOSITORY if provider == "github" else marketplace.GITCODE_REPOSITORY
    assert snapshot["pymss.example"]["url"] == expected
    assert snapshot["external.plugin"] == catalog["external.plugin"]
    assert catalog["pymss.example"]["ref"] == "main"


@pytest.mark.parametrize("revision,catalog", [
    ({"sha": "main"}, {}), ({"sha": COMMIT}, []),
    ({"sha": COMMIT}, {"_comment": "empty"}), ({"sha": COMMIT}, {"bad": {}}),
])
def test_invalid_source_metadata_is_rejected(monkeypatch, revision, catalog):
    monkeypatch.setattr(marketplace, "_json", lambda url: revision if "/commits/" in url else catalog)
    with pytest.raises(SourceError):
        marketplace._probe("gitcode")


@pytest.mark.parametrize("body,url", [
    (b"<html>not a catalog</html>", marketplace.GITHUB_REGISTRY),
    (b"x" * (marketplace._MAX_METADATA_BYTES + 1), marketplace.GITHUB_REGISTRY),
    (b"{}", "https://untrusted.example/catalog.json"),
], ids=["html", "oversized", "redirect"])
def test_source_probe_rejects_html_oversized_data_and_untrusted_redirects(monkeypatch, body, url):
    response = io.BytesIO(body)
    response.url = url
    monkeypatch.setattr(marketplace.urllib.request, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises((SourceError, ValueError)):
        marketplace._json(marketplace.GITHUB_REGISTRY)
