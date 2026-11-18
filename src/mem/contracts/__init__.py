"""Frozen interface contracts (pvd-roadmap §2.3). Changing a file here is a visible diff."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def mcp_tools() -> dict[str, Any]:
    text = resources.files(__name__).joinpath("mcp_tools.json").read_text(encoding="utf-8")
    return json.loads(text)
