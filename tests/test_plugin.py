"""The Claude Code plugin manifests (MEM2-138): valid JSON, the real skill, mem serve's port."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from mem import serve

REPO = Path(__file__).resolve().parents[1]


def _json(rel: str) -> dict:
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def test_plugin_manifest_names_the_existing_skill():
    plugin = _json(".claude-plugin/plugin.json")
    assert plugin["name"] == "mem"
    assert plugin["skills"] == "./skills/"
    assert (REPO / plugin["skills"] / "mem" / "SKILL.md").is_file()
    assert (REPO / plugin["mcpServers"]).resolve() == REPO / ".mcp.json"


def test_mcp_servers_are_the_local_mem_serve_with_the_bearer_token():
    servers = _json(".mcp.json")["mcpServers"]
    assert sorted(servers) == [f"mem-{n}" for n in serve.STORE_NAMES]
    for name in serve.STORE_NAMES:
        entry = servers[f"mem-{name}"]
        url = urlsplit(entry["url"])
        assert entry["type"] == "http"
        assert (url.scheme, url.hostname, url.port) == ("http", "127.0.0.1", serve.DEFAULT_PORT)
        assert url.path == f"/{name}/mcp"
        assert entry["headers"]["Authorization"] == "Bearer ${MEM_TOKEN}"


def test_marketplace_lists_the_plugin_at_the_repo_root():
    market = _json(".claude-plugin/marketplace.json")
    (entry,) = market["plugins"]
    assert entry["name"] == _json(".claude-plugin/plugin.json")["name"]
    assert entry["source"] == "./"
    assert market["owner"]["name"]
