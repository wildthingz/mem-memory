# Two machines: home and work

One machine runs mem: the **home** machine, with the GPU, the models and the SQLite files. The **work** machine runs no
models and holds no memories; its coding harness talks to the home machine's server over your tailnet
([Tailscale](https://tailscale.com)). Both machines see the same stores. Copying the files between machines is not supported
(two writers corrupt a database).

> **Status.** The pieces are tested on one machine with a stand-in server. A real home + work run, timed with someone who
> has not read the code, has not been done yet.

What you need: both machines on the same tailnet with MagicDNS and HTTPS certificates turned on (Tailscale admin console,
DNS page), and `mem setup` already done on the home machine ([README](../README.md#quickstart)).

## 1. Home: serve over the tailnet, and make a code

Serving over the tailnet is the one manual step: add `--tls --token-file %h/.config/mem/tokens` to `ExecStart` of
`~/.config/systemd/user/mem.service`, then publish it ([multi-machine.md](multi-machine.md#set-up-the-serving-machine) has the
exact commands and what protects the endpoint):

```bash
systemctl --user daemon-reload && systemctl --user restart mem
tailscale serve --bg --https=443 http://127.0.0.1:8765
mem token add work-laptop --code --host home.your-tailnet.ts.net
```

The last command prints one code that holds the host name and a new token for this client (one token per client; the label
`work-laptop` is what the audit log shows). `mem token revoke work-laptop` cuts it off on the next request.

## 2. Work: add the home machine

Install mem on the work machine the same way as at home but without the models (`uv tool install .`), or any machine with
`mem` on its PATH, then:

```bash
mem setup --remote home.your-tailnet.ts.net --code <the code>
```

Without `--code` it asks for the token (typed hidden). It checks that the host answers and accepts the token before it
changes anything, then stores the token in `~/.config/mem/remote-home.token` (mode 600), lists the host in
`~/.config/mem/remotes.json`, and adds `mem-home-personal` and `mem-home-work` to each harness it finds (Claude Code,
Codex CLI, Gemini CLI, Cursor, VS Code); `mem setup` without `--remote` is for the machine that serves. Run it again after a token change; it skips what is done.

Open the harness and ask it to list its MCP servers or tools: the `mem-home-*` servers show six tools each (`recall`, `get`,
`remember`, `link`, `subjects`, `forget`). Then: *"Remember in my work memory: …"* here, and *"What do you remember about …"*
on the home machine.

## Check and undo

- `mem doctor` on the work machine also checks each remote (the `remote home` line): reachable, token accepted, round-trip time (a warning above
  500 ms). The work machine has no GPU or models, so its `mem doctor` also prints `device` and model failures; they do not matter there, read the `remote` line. Each failure and its fix is in [troubleshooting.md](troubleshooting.md#remote-failures).
- `mem setup --uninstall` removes the remote servers from the harness configs (edited files come back byte for byte) and
  keeps `remotes.json` and the token files; `--purge` deletes them too.
- Roll the home side back with [multi-machine.md](multi-machine.md#roll-back).
