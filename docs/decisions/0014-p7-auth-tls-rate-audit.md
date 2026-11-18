# 0014: P7 auth: bearer tokens everywhere, TLS by `tailscale serve`, owner-consent OAuth

**Status:** accepted (MEM2-54). Spec: architecture-v2 §6 P7, benchmarks-v2 B-H1. Code: `src/mem/auth.py`,
`mem serve --token-file/--tls/--oauth/--rate-limit/--audit-log`, `mem token`, `bench auth-overhead`.
Runbook: `deliverables/runbooks/p7-auth.md`.

## Choices and why
- **TLS is tailscaled's, not mem's.** `tailscale serve --https=443 http://127.0.0.1:8765` terminates TLS with the
  tailnet's Let's Encrypt certificate and proxies to loopback. `mem serve --tls` binds loopback only, so there is no
  plaintext tailnet socket. Rejected: TLS inside uvicorn (certificate fetch, renewal and file permissions become mem's
  job, for no gain on one host).
- **Loopback is not exempt from the token.** Behind `tailscale serve`, every request comes from 127.0.0.1. Exempting
  loopback would exempt the whole tailnet (and the internet, with funnel). Local Claude Code sessions send the token too.
- **"TLS only" means:** a direct loopback client, or a loopback proxy that sends `X-Forwarded-Proto: https`.
  Anything else gets 403: a non-loopback peer, or a proxied request without https (`tailscale serve --http`). For the
  same reason, `/models` now refuses any request with forwarding headers. Before, `tailscale serve` would have exposed
  it.
- **Tokens:** random `mem_…` (256 bits), shown once. The file keeps `label sha256` (0600) and is re-read when it
  changes, so revoking a token takes effect without a restart. No bcrypt: the tokens are random and long, so a
  slow hash adds nothing.
- **Plaintext `--tailscale` (P5) still works** but warns when it runs without a token. That keeps a P5 unit running
  through an upgrade until the person switches (runbook §2). `--tls` refuses `--tailscale`, a non-loopback `--host`,
  and a missing or empty token file.
- **Rate limit:** a token bucket per client label. A request without a valid token spends from its peer's bucket
  (X-Forwarded-For behind the proxy) before it gets 401, so guessing is limited too. Default 600/min with bursts as
  large: Claude Code makes a few requests a turn. Only the MCP and OAuth routes are limited; `/models` is not (it is
  local-only, and the bench embeds in bulk).
- **Audit log:** JSON lines, 0600, on by default for HTTP (`~/.local/state/mem/audit.jsonl`). One line per request on
  every route, written when the response starts. The line has who, what, the JSON-RPC method and tool, the status,
  the time in ms and the reason for a refusal. It never holds a token, a query string or tool arguments: the
  memories stay in the stores. The `events` table is unchanged: it records tool calls; the audit log records requests.
- **OAuth (optional, for a claude.ai connector):** the MCP SDK's authorization-server routes (DCR, PKCE, refresh,
  revocation) over `OwnerOAuth`. The consent page approves a client only when the person pastes one of their mem
  tokens. There is no password store and no second secret to manage. Issued tokens are kept hashed in
  `~/.config/mem/oauth.json` and survive restarts. Access tokens last 1 h; refresh tokens last 30 d and rotate on use.
  claude.ai calls from Anthropic's cloud, so it needs `tailscale funnel` (public). That is why `--oauth` needs `--tls`,
  and why the runbook keeps funnel off unless the connector is in use.
