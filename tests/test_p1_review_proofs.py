import json

from mem import doctor
from mem.common import estimate_tokens
from mem.serve import MemServer


def test_get_total_budget_includes_all_ten_headers(store):
    server = MemServer(store)
    ids = [
        server.call_tool(
            "remember",
            {
                "text": f"Preference {i}: tea over coffee.",
                "type": "preference",
                "about": ["new:person:Ana"],
                "source": "personal-preferences",
            },
        )[1]["id"]
        for i in range(10)
    ]
    text, _ = server.call_tool("get", {"ids": ids, "max_tokens": 300})
    assert estimate_tokens(text) <= 300


def test_composite_continuation_does_not_lose_body_when_edges_remain(store):
    saved = store.remember(text="Ana likes tea.", type="fact", about=["new:person:Ana"])
    eid = saved.structured["subjects"][0].split()[0]
    body = "Biography paragraph.\n\n" * 100 + "FINAL BIOGRAPHY SENTINEL"
    store.conn.execute("UPDATE nodes SET body=? WHERE id=?", (body, int(eid[1:])))
    first = store.get([eid], max_tokens=100).structured["items"][0]
    pages = [first["text"]]
    item = first
    for _ in range(100):
        if not item.get("continuation"):
            break
        item = store.get([item["continuation"]], max_tokens=100).structured["items"][0]
        pages.append(item["text"])
    assert "FINAL BIOGRAPHY SENTINEL" in "".join(pages)


def test_doctor_empty_weight_map_is_not_a_model(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    assert doctor.missing_model_files(tmp_path), "No weights exist, but snapshot passes"
