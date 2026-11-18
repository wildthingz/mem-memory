"""Check a tool's structured result against the frozen contract (src/mem/contracts)."""

from __future__ import annotations

from typing import Any

from mem import contracts

_TYPES = {
    "string": str, "integer": int, "boolean": bool, "array": list, "object": dict,
    "number": (int, float), "null": type(None),
}  # fmt: skip


def _type_ok(schema: dict[str, Any], value: Any) -> bool:
    t = schema.get("type")
    if t is None:
        return True
    names = t if isinstance(t, list) else [t]
    if isinstance(value, bool) and "boolean" not in names:
        return False
    return any(isinstance(value, _TYPES[n]) for n in names)


def _walk(schema: dict[str, Any], value: Any, path: str, out: list[str]) -> None:
    if not _type_ok(schema, value):
        out.append(f"{path}: expected {schema.get('type')}, got {type(value).__name__}")
        return
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                out.append(f"{path}.{key}: missing")
        if schema.get("additionalProperties") is False:
            out.extend(f"{path}.{k}: not in contract" for k in value if k not in props)
        for k, v in value.items():
            if k in props:
                _walk(props[k], v, f"{path}.{k}", out)
    elif isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(f"{path}: {len(value)} items > maxItems {schema['maxItems']}")
        for i, v in enumerate(value):
            _walk(schema.get("items", {}), v, f"{path}[{i}]", out)
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: {value!r} not in {schema['enum']}")


def structured_violations(tool: str, structured: dict[str, Any]) -> list[str]:
    spec = next(t for t in contracts.mcp_tools()["tools"] if t["name"] == tool)
    out: list[str] = []
    _walk(spec["structuredContent"], structured, tool, out)
    return out
