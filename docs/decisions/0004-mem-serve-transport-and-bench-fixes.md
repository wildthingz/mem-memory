# 0004: mem serve, MCP SDK, sessions, and fixes to the P1 suites

Status: accepted · Task: MEM2-11

## Decisions
- **New dependencies:** `mcp` 2.3.0 (DR-1, the MCP Python SDK; pulls starlette) and `uvicorn` 0.54
  (the ASGI server for `--http`). `jsonschema` (also pulled by `mcp`, pinned directly because we import it) validates tool
  arguments against the frozen `inputSchema`.
- **Low-level `Server`, not the decorator API:** the tool list must equal `contracts/mcp_tools.json`
  exactly (`test_mcp_tool_contracts`), so the snapshot is served as is (`inputSchema`, with
  `structuredContent` as MCP `outputSchema`) instead of being derived from Python signatures.
- **Layout:** one MCP `Server` per store, mounted at `/<store>/mcp` in one Starlette app (D4); the
  sub-app lifespans are entered by the parent. Stores are opened in the thread that runs the event
  loop (sqlite connections are thread-bound); tool calls run on the loop thread, so calls are serial.
- **Errors:** a `MemError` (bad input, `link`/`subjects`/`forget` before P3) is a tool result with
  `isError=true` and the message, so Claude can correct itself. Unknown arguments are rejected.
- **Session and host (A12):** session = `Mcp-Session-Id` header (handshake-era clients); host =
  `X-Mem-Host` header, else `--host-label`. The 2026-07-28 MCP protocol has no sessions, so a client
  speaking it logs `session = NULL`. Not verified which protocol Claude Code speaks: the P1 runbook
  has a query that shows it. stdio uses a per-process id; in-process uses `inproc`.
- **Embedder:** `mem.embedder.SentenceTransformerEmbedder` (pinned snapshot, `local_files_only`,
  `HF_HUB_OFFLINE=1`). `mem serve` exits 1 with `mem doctor --fetch` advice if the embedder
  snapshot is missing and creates no store file. `--embedder hash` is the test stub (stdio test).
- **Row text:** the P1 `p` column is the bare RRF score (`.0328`), as the spec's row format and the
  P1 suites require; it was `rrf 0.0328`.
- **`get` budget:** `max_tokens` is split equally over the ids still to do. Sequential allocation let
  early items starve later ones and the whole response overshot (10 ids, 300 tokens → 332).

## Fixes to the p1-bench suites (they contradicted the spec or the frozen contract)
- **B-R6 long `get` item:** it was saved with `remember`, which caps text at 1,200 chars (A1); the
  9 kB fixture is now written straight to the store, like a long imported item.
- **B-R6 ids:** 11 ids in one `get` exceeded the contract's `maxItems: 10`; now 10.
- **B-W6 item 6:** it demanded disjoint ids across stores, but §2.3 says ids are integers per store.
  It now checks isolation (the other store's `get` and `recall` return nothing). Self-check mutant
  `shared_ids` became `cross_store`. **Maintainer: say if disjoint ids were meant; that is a design change.**
- **B-C2:** required an empty recall among corpus queries; P1 has no gate, so none is empty. It
  records `n_empty` and requires only that not all are empty.

## Rejected
- Changing the work store's id base to make ids disjoint: unspecified, and wrong for the §2.3 text.
- `FastMCP`-style decorators: would derive schemas from code and drift from the frozen snapshot.
- Worker threads for tool calls: needs `check_same_thread=False` and a lock for no gain with one user.
