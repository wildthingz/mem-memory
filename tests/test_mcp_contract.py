import pytest

from mem import contracts


def _tool(name):
    return next(t for t in contracts.mcp_tools()["tools"] if t["name"] == name)


def test_tool_names_are_the_spec_six():
    names = [t["name"] for t in contracts.mcp_tools()["tools"]]
    assert names == ["recall", "get", "remember", "link", "subjects", "forget"]


@pytest.mark.parametrize(
    ("tool", "defaults"),
    [
        ("recall", {"include_history": False, "limit": 8, "budget": 600}),
        ("get", {"max_tokens": 2000}),
        ("remember", {"stated_by_user": False, "pin": False, "distinct_new": False}),
        ("link", {"remove": False}),
        ("subjects", {"limit": 20}),
    ],
)
def test_defaults_match_spec_4_1(tool, defaults):
    props = _tool(tool)["inputSchema"]["properties"]
    assert {k: v["default"] for k, v in props.items() if "default" in v} == defaults


def test_required_params_and_limits():
    assert _tool("recall")["inputSchema"]["required"] == ["query"]
    assert _tool("remember")["inputSchema"]["required"] == ["text", "type", "about"]
    assert _tool("remember")["inputSchema"]["properties"]["about"]["minItems"] == 1
    assert _tool("remember")["inputSchema"]["properties"]["facts"]["maxItems"] == 5
    assert _tool("get")["inputSchema"]["properties"]["ids"]["maxItems"] == 10
    structured = _tool("remember")["structuredContent"]["properties"]
    assert structured["possible_conflicts"]["maxItems"] == 5
    assert structured["status"]["enum"] == ["saved", "exists", "needs_choice", "quarantined"]


def test_recall_structured_row_fields():
    row = _tool("recall")["structuredContent"]["properties"]["rows"]["items"]["required"]
    assert row == ["id", "kind", "type", "date", "title", "about", "p", "flags"]


def test_mcp_tool_contracts():
    """The live server's advertised tools must equal the frozen snapshot."""
    from mem.serve import list_tool_schemas

    assert list_tool_schemas() == contracts.mcp_tools()["tools"]


def test_p3_tools_are_all_live_and_unchanged_in_shape():
    """link, subjects, forget are no longer 'later phase' tools: no `phase` marker, and the
    input shapes the P1 snapshot froze are unchanged (only descriptions may move)."""
    assert all("phase" not in t for t in contracts.mcp_tools()["tools"])
    assert _tool("link")["inputSchema"]["required"] == ["src", "rel", "dst"]
    assert sorted(_tool("link")["inputSchema"]["properties"]) == [
        "dst", "note", "rel", "remove", "src"]  # fmt: skip
    assert _tool("forget")["inputSchema"]["required"] == ["id", "reason"]
    assert _tool("forget")["structuredContent"]["required"] == ["ok", "id", "status"]
    assert _tool("subjects")["inputSchema"]["required"] == []
    assert "P3" not in _tool("forget")["description"]


def test_p3_tools_have_no_later_phase_gate():
    from mem import serve

    assert not hasattr(serve, "LATER_PHASE")


def test_remember_distinct_new_is_optional_and_documented():
    """ADR 0027: the only contract change is one optional boolean."""
    schema = _tool("remember")["inputSchema"]
    assert "distinct_new" not in schema["required"]
    assert schema["properties"]["distinct_new"]["type"] == "boolean"
    assert "new:" in schema["properties"]["distinct_new"]["description"]
