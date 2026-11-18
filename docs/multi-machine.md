# Using mem from more than one machine (Tailscale, optional)

Tailscale is optional. A single machine needs none of this: `mem install` binds the server to
`127.0.0.1` and Claude Code on that machine reaches it with a bearer token.

mem is built around one server and one set of SQLite files. A second machine connects to the first
machine's server over your [tailnet](https://tailscale.com); the files never leave it, and two
machines never write to separate copies (two writers would corrupt the database, so syncing the
files is not supported).

> **Status.** The server side below (tokens, HTTPS through `tailscale serve`, rate limit, audit log)
> was exercised on one machine. Connecting a second machine has not been run end to end yet; treat
> that part as untested.

## What protects the endpoint

- The server listens on `127.0.0.1` only. The tailnet reaches it only through `tailscale serve` HTTPS
  (`--tls`), which also gives it a certificate.
- Every MCP request needs a bearer token (`--token-file`), local ones included. One token per client;
  the file keeps only each token's SHA-256.
- Requests are rate limited per client (`--rate-limit`, default 600 a minute) and every request is
  written to an audit log (`~/.local/state/mem/audit.jsonl`, mode 600, no token or argument ever
  logged).
- `/models`, which hands the loaded models to local tools, answers loopback clients only.

## Set up the serving machine

1. In the Tailscale admin console (DNS page), turn on MagicDNS and HTTPS Certificates.
2. One token per client. The label is what the audit log shows. The token is printed once:

   ```bash
   uv run --extra models mem token add laptop
   uv run --extra models mem token list          # labels only, never the tokens
   uv run --extra models mem token revoke laptop # takes effect on the next request, no restart
   ```

3. Add `--tls --token-file %h/.config/mem/tokens` to the `ExecStart` of
   `~/.config/systemd/user/mem.service` (the token file is the one `mem install` created), then:

   ```bash
   systemctl --user daemon-reload && systemctl --user restart mem
   ss -ltnp | grep 8765                          # exactly one listener, 127.0.0.1:8765
   tailscale serve --bg --https=443 http://127.0.0.1:8765
   tailscale serve status                        # https://<machine>.<tailnet>.ts.net (tailnet only)
   ```

`--tls` needs `--token-file` and a MagicDNS name; the server refuses to start with a message saying
which is missing. If `tailscale` is not installed or down, `mem serve` says so and keeps serving
loopback.

## Connect a client

On the serving machine, print the commands for a client (its own host label and token):

```bash
uv run --extra models mem client-config --server <machine>.<tailnet>.ts.net --host-label laptop --tls --token
```

It prints two `claude mcp add ...` lines, for `mem-personal` and `mem-work`, with the header
`X-Mem-Host: laptop` and `Authorization: Bearer $MEM_TOKEN`. On the client, put the token in the
shell without writing it on a command line, then run both lines in that shell:

```bash
read -rs MEM_TOKEN && export MEM_TOKEN
```

The `X-Mem-Host` label becomes the `host` of every event in the log, so you can tell which machine a
call came from. Do not start a label with `bench-`; those are left out of the benchmark counts.

## Check it

An MCP `initialize` over the tailnet name must answer 200 with a good token and 401 without one:

```bash
TS=<machine>.<tailnet>.ts.net
INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'
curl -s -o /dev/null -w '%{http_code}\n' https://$TS/work/mcp -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' -H "Authorization: Bearer $MEM_TOKEN" -d "$INIT"   # 200
curl -s -o /dev/null -w '%{http_code}\n' https://$TS/work/mcp -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' -d "$INIT"                                         # 401
```

A 421 means the Host name differs from the MagicDNS name: add `--allow-host <name>` to `ExecStart`.
A plain `curl` GET without these headers answers 400; that is not the check.

## Optional: a claude.ai connector

claude.ai calls from Anthropic's cloud, not from your tailnet, so this puts the endpoint on the public
internet through `tailscale funnel`. Every request still needs a token; an OAuth token is issued only
after you approve on mem's consent page by pasting one of your mem tokens. It is off unless you add
`--oauth` to `ExecStart` and run `tailscale funnel`. Skip it unless you want it.

## Roll back

Remove `--tls --token-file ...` from the unit (or keep the token file and drop only `--tls`), run
`systemctl --user daemon-reload && systemctl --user restart mem`, and `tailscale serve --https=443 off`.
