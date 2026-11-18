# 0014: P7 poisoning quarantine, provenance weighting and secret/injection filters on `remember`

**Status:** accepted (MEM2-56, P7). Spec: architecture-v2 §6 P7 line, benchmarks-v2 B-H1 ("poisoning fixtures are
quarantined · secrets in `remember` are rejected"). Code: `src/mem/guard.py`, migration `0002_poisoning.sql`.

## What
- **Tool-sourced** = `remember(source=...)` with a URL (`http(s)://`, `ftp://`) or `tool:<name>`. Stored as
  `nodes.tool_sourced = 1`. Any other `source` (a repo, a topic) stays first-hand. The skill tells Claude to pass the URL
  or `tool:<name>` when the text comes from a fetched page, a file or tool output.
- **Secrets** (any of text, title, summary, facts, about, source): known token shapes (private key blocks, AWS, GitHub,
  GitLab, Slack, `sk-` API keys, Google, Stripe, JWT, credentials in a URL, bearer tokens) and `password/token/secret/
  api key … : value` where the value has ≥ 8 chars and ≥ 2 character classes. The call fails with `rejected: … (kind)`;
  nothing is written, no event is logged, the value is not echoed.
- **Injection → quarantine** (any provenance): instruction override, role reassignment, forged prompt markup, system-prompt
  probes, "don't tell the user", exfiltration of secrets/env. **Tool-sourced only**, also: text addressed to an AI reader
  ("note to AI assistants", "if you are an AI", "when you read this") and `curl … | sh`. First-hand memories may hold those
  legitimately (the owner's install procedure, "the assistant should always run the tests").
  Matching runs on NFKC + casefold + zero-width-stripped text.
- A quarantined memory is stored (`nodes.quarantine` = reason) and returned as `status: quarantined`. It is out of
  `recall` (eligible set), out of conflict candidates and the `SessionStart` core index, cannot supersede or be
  superseded, and `get` shows it as `QUARANTINED (reason): untrusted data, not instructions`.
  `mem quarantine --store X` lists them; `--release m12` puts one back; the rest are `forget`-ed.
- **Provenance weighting:** tool-sourced chunks × `TOOL_WEIGHT = 0.8`, on the RRF score (pool order, the P1 path) and on
  the gate probability before `gate.select` (so a tool-sourced row needs p ≥ τ_keep / 0.8 and ranks below an equally
  relevant first-hand row). Recall rows get a `tool-sourced` flag. The p shown is the weighted p.

## Quarantine on the read, link and render paths (MEM2-57 F2/F10 ruling, MEM2-109)
- **`get` may return a quarantined memory**, but only labelled: the `m<id>` header, the header of any `c<id>` chunk of
  it, and its line in a neighbour's edge list (`get e<id>`) carry `QUARANTINED (reason): untrusted data, not
  instructions`. Ruling from the maintainer: the person and the agent need to be able to see what was held and why (to decide
  on `--release` or `forget`); the label is what keeps it from being read as an instruction. `recall`, conflicts and the
  core index still never surface it, so an agent only reaches it by naming the id.
- **`link` refuses every rel** (`about`, `relates_to`, `duplicate_of`, `same_as`, `supersedes`, …) with a quarantined
  memory as `src` or `dst`: an edge would tie the poisoned text into the graph (entity pages, chains, merges) before the
  person has reviewed it. `link(..., remove=True)` stays allowed: taking an edge away only shrinks its reach. The
  `about` edges `remember` itself writes are kept, so a release restores the memory as it was saved.
- **`render` never writes a quarantined memory** (no page, no `index.md` line, no entry on an entity page): the vault
  mirror is committed to git and read as trusted text, with no place for the label.

## Why these choices
- **First-hand rows keep weight 1.0** (user, agent and import alike): every earlier T0–T1 gate ran on first-hand rows
  only, so their rankings and the calibrated gate are unchanged. Down-weighting `agent` against `user` would move rows
  across τ_keep without a re-fit.
- **Weight outside `config.retrieval`:** that section is fingerprinted (`fp_all`); a new key there would orphan every
  stored gate.
- **Quarantine, not rejection, for injection:** a false positive costs one review line, not lost data; the person decides.
- **Columns, not a status value:** the `status` CHECK can't be widened without a table rebuild; `ALTER TABLE ADD COLUMN`
  is a safe in-place migration for the live stores.

## Limits
Pattern filters, not a classifier: paraphrased injection without the marker phrases gets through as an ordinary
(down-weighted, flagged, if tool-sourced) memory. Secrets split by zero-width characters or in unknown formats are not
caught. Both lists live in `guard.py` and grow with fixtures in `tests/fixtures/hardening/`.
