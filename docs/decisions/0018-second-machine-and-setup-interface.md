# 0018: A full install per machine, remote access over MCP, and the setup interface

- **Status:** Proposed (requested by the owner, MEM2-1 comments of 2026-10-26 16:20)
- **Date:** 2026-10-26 · **Task:** MEM2-232 · **By:** Vault
- Spec: easy-setup-integrations §2.3. Code: `src/mem/setup_steps.py`, `src/mem/install.py`, `docs/multi-machine.md`.

## Context
The owner uses two machines (personal, work). mem keeps one set of SQLite files per server; two writers on copies
corrupt it (`docs/multi-machine.md`). Work notes must not mix into the personal store.

## Decision
1. **Each machine gets a full local install** (`mem install` / `mem setup`) with its own stores, named `mem-personal` and
   `mem-work` by the machine's role. Nothing is synced. Everything runs locally; no cloud service.
2. **Reaching the other machine's store** uses the P5 design: MCP over HTTP with a bearer token, over Tailscale
   (`tailscale serve`, tailnet only, HTTPS from MagicDNS). The server stays bound to 127.0.0.1; there is no public
   exposure and no funnel unless the user opts in to the claude.ai connector.
3. **`mem setup --remote <host>`** registers the other machine's servers in this machine's harness as extra MCP
   entries. The token is exchanged by one of two ways: (a) one command on the serving machine, `mem token add <label>
   --code`, prints a short code that encodes host and token, and `mem setup --remote <host> --code <code>` consumes it;
   (b) the user pastes the token when prompted. The token is read from a prompt or the code, never from argv, and is
   stored mode 0600.
4. **`mem doctor` checks a remote** in this order, stopping at the first failure with the fix: reachability (name resolves
   on the tailnet, TCP and TLS up), token (MCP `initialize` answers 200 with it; 401 means re-issue), latency (round
   trip of `initialize`, reported; a warning above a limit set in the doctor code, not a failure).
5. **The setup interface** (`setup_steps.py`): `mem install`'s plan of actions, plus one `HarnessWriter` per harness
   (`detect`, `configured`, `add`, `installed`, `remove`). Before editing any config file a writer saves it in
   `Backups` (one slot per file, first save wins); uninstall restores the exact bytes and mode, or deletes a file that
   did not exist. All system calls go through the `Runner` from `mem install`, so tests use a fake one. (The first
   draft had a generic `Step` list with `run`; setup was built on the plan and the writers instead, and MEM2-242
   removed the unused `Step`s.)

## Alternatives considered
- **Sync the stores (rsync, Syncthing):** two writers corrupt SQLite; rejected already in `multi-machine.md`.
- **One server, both machines as thin clients:** work data would live on the personal machine, or the reverse. Separate
  stores keep that line.
- **Public HTTPS with OAuth:** needless exposure for two machines the owner controls.

## Consequences
- Cross-machine recall is an MCP call to the other store, so it needs that machine up and on the tailnet.
- Tokens are per client and revocable (`mem token revoke`), effective on the next request.
- Connecting a second machine is still untested end to end (`multi-machine.md` status); the host runbook covers it.
