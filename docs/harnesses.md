# Harnesses

mem works in a harness through two things, and both are the same everywhere:

1. **The MCP servers** (`mem-personal`, `mem-work`) are the interface. Nothing about them changes per harness.
2. **The skill** `skills/mem/SKILL.md` says when to recall and when to save. A harness that loads Agent Skills (the
   `SKILL.md` folder format) gets that file in its skills directory. Where it cannot, `mem instructions --harness <name>`
   prints a short text cut from the same file (a test fails if the two drift apart).

```bash
mem mcp-config --harness codex             # print the entry (the token shows as <token>)
mem mcp-config --harness codex --write     # merge it into the harness's file; backs the file up first
mem mcp-config --harness codex --uninstall # put the file back byte for byte
mem instructions --harness claude-desktop  # the text to paste, for a harness without skills
```

Writing keeps your other servers and settings, saves the file first (`Backups`, one slot per file, first save wins) and does
nothing on a second run. `--uninstall` restores the saved bytes; if you changed the file since, it takes out only the
two `mem-*` entries (and keeps any comment you added). JSON files are rewritten with two-space indentation; a file with
comments (Gemini CLI and VS Code read JSONC) is edited in place instead, so its comments and layout stay. mem checks every
edit by reading the result back and never writes a file that would not parse or would say something else; a file it
cannot read or edit safely is left as it is, and `mem setup` skips that harness with a `[warn]` line and goes on with the
others. A symlinked config (a dotfiles repo) stays a link: mem writes its target. The backups in
`~/.config/mem/backups` (mode 0700) keep the mode of the file they copy.

The entries point at the server `mem install` starts (`http://127.0.0.1:8765/<store>/mcp`) and send the label
`X-Mem-Host` and, if `~/.config/mem/token` exists, `Authorization: Bearer <token>`. That token is written into the harness's
file, which is created mode 0600; an existing file keeps its mode. The stdio form (`--transport stdio`) starts one `mem serve`
process per store, which loads the models itself; use it only where HTTP is not possible.

`mem setup` finds the harnesses by their command on `PATH` or their config folder, and, for each one found, merges the servers into its file (`ConfigWriter` in
`src/mem/harnesses.py`, with the backup rules above) and copies the skill. Skills for Codex CLI, Gemini CLI, Cursor and VS Code go to one shared folder, `~/.agents/skills/mem`
(all four read it), Claude Code's to `~/.claude/skills/mem`. Cursor and VS Code also read `~/.claude/skills`, so with both
installed they may list the skill twice; that is harmless, and neither docs page says how duplicates are handled.

Checked on 2026-10-26 against the official docs named in each section. Those pages carry no release number for the
feature unless a section says so; the date is the version.

## Claude Code

- **Config:** `~/.claude.json`, top-level `mcpServers` (user scope; the same file also holds Claude Code's own state, which
  is kept). HTTP entries need `"type": "http"`; an entry with a `url` and no type is read as stdio and fails.

  ```json
  {"mcpServers": {"mem-work": {"type": "http", "url": "http://127.0.0.1:8765/work/mcp",
   "headers": {"X-Mem-Host": "laptop", "Authorization": "Bearer <token>"}}}}
  ```

  stdio: `{"type": "stdio", "command": "uv", "args": ["run", "--project", "<repo>", "mem", "serve", "--store", "work=<path>"]}`.
- **Skill: native.** `~/.claude/skills/mem/SKILL.md` (personal, all projects). Edits are picked up without a restart.
- **Instructions (older versions only):** `CLAUDE.md`.
- `mem install` registers the servers with `claude mcp add` and copies the skill; the writer here does the same by file edit,
  and the Claude Code plugin brings both with no file edit.
- **Docs:** <https://code.claude.com/docs/en/mcp>, <https://code.claude.com/docs/en/skills> (checked 2026-10-26).

## Codex CLI

- **Config:** `~/.codex/config.toml` (`$CODEX_HOME` if set), one `[mcp_servers.<name>]` table per server. Streamable HTTP
  takes `url`, `http_headers` (static map), `env_http_headers` and `bearer_token_env_var`.

  ```toml
  [mcp_servers.mem-work]
  url = "http://127.0.0.1:8765/work/mcp"
  http_headers = { "X-Mem-Host" = "laptop", "Authorization" = "Bearer <token>" }
  ```

  The writer edits the file as text: it appends the two tables, or replaces them if they exist, and leaves every other line.
- **Skill: native.** `~/.agents/skills/mem/SKILL.md` (user level; repo level is `.agents/skills`).
- **Instructions (older versions only):** `AGENTS.md`.
- **Docs:** <https://developers.openai.com/codex/mcp> and <https://developers.openai.com/codex/skills> (both now redirect to
  <https://learn.chatgpt.com/docs/extend/mcp> and <https://learn.chatgpt.com/docs/build-skills>; checked 2026-10-26).

## Gemini CLI

- **Config:** `~/.gemini/settings.json`, top-level `mcpServers`. Streamable HTTP uses `httpUrl` (a plain `url` means SSE) with
  `headers`; stdio uses `command` and `args`, with no `type`.

  ```json
  {"mcpServers": {"mem-work": {"httpUrl": "http://127.0.0.1:8765/work/mcp",
   "headers": {"X-Mem-Host": "laptop", "Authorization": "Bearer <token>"}}}}
  ```
- **Skill: native.** `~/.gemini/skills/` or its alias `~/.agents/skills/` (user), `.gemini/skills/` or `.agents/skills/`
  (workspace). mem uses the alias, so Codex, Cursor and VS Code share the copy; within a tier the alias wins over
  `.gemini/skills`, so no duplicate.
- **Instructions (older versions only):** `GEMINI.md` (`~/.gemini/GEMINI.md` or in the project). The `context.fileName`
  setting can also make it read `AGENTS.md`.
- **Note:** the docs banner says Gemini CLI was replaced by Antigravity CLI for unpaid-tier and Google One users on
  2026-06-18. Not covered here; the skill page is dated "Last updated: Apr 30, 2026".
- **Docs:** <https://geminicli.com/docs/tools/mcp-server/>, <https://geminicli.com/docs/cli/skills/>,
  <https://geminicli.com/docs/cli/gemini-md/> (checked 2026-10-26).

## Cursor

- **Config:** `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project), top-level `mcpServers`. A remote server is
  `url` plus `headers`; stdio is `type: "stdio"` (the docs list it as required, their examples omit it), `command`, `args`.
  Values can use `${env:NAME}`; the writer puts the literal token, since the file is 0600.

  ```json
  {"mcpServers": {"mem-work": {"url": "http://127.0.0.1:8765/work/mcp",
   "headers": {"X-Mem-Host": "laptop", "Authorization": "Bearer <token>"}}}}
  ```
- **Skill: native.** `~/.agents/skills/mem/` (also reads `~/.cursor/skills`, `~/.claude/skills`, `~/.codex/skills`). The skills
  page says only that `/migrate-to-skills` arrived in Cursor 2.4, so skills exist from 2.4 on.
- **Instructions (older versions only):** `.cursor/rules/mem.mdc` with `alwaysApply: true` (only `.mdc` files count as
  rules), or `AGENTS.md` in the project root.
- **Docs:** <https://cursor.com/docs/context/mcp>, <https://cursor.com/docs/context/skills>,
  <https://cursor.com/docs/context/rules> (checked 2026-10-26).

## VS Code (Copilot agent)

- **Config:** the user-profile `mcp.json` (open it with the command *MCP: Open User Configuration*), or `.vscode/mcp.json`
  in a workspace. The top-level key is **`servers`**, not `mcpServers` (`mcpServers` is the portable format). HTTP:
  `type: "http"`, `url`, `headers`; stdio: `type: "stdio"`, `command`, `args`. Secrets can come from `inputs`
  (`promptString`, `password: true`); the writer puts the literal token instead, as for the other harnesses.

  ```json
  {"servers": {"mem-work": {"type": "http", "url": "http://127.0.0.1:8765/work/mcp",
   "headers": {"X-Mem-Host": "laptop", "Authorization": "Bearer <token>"}}}}
  ```

  The writer uses `~/.config/Code/User/mcp.json` (`$XDG_CONFIG_HOME/Code/User` if set; `~/Library/Application Support/Code/User`
  on macOS). The docs name no file path for the user profile; this is VS Code's usual `User` folder and the file the
  command above opens. Check it once with that command.
- **Skill: native.** `~/.agents/skills/mem/` (also `~/.copilot/skills`, `~/.claude/skills`; project: `.github/skills`,
  `.claude/skills`, `.agents/skills`). No setting is needed. `name` must equal the folder name, as `mem` does.
- **Instructions (older versions only):** `.github/copilot-instructions.md` (always applied), or `AGENTS.md`.
- **Docs:** <https://code.visualstudio.com/docs/agents/reference/mcp-configuration>,
  <https://code.visualstudio.com/docs/copilot/customization/mcp-servers>,
  <https://code.visualstudio.com/docs/copilot/customization/agent-skills> (skills page dated 2026-10-21; checked 2026-10-26).

## Claude Desktop

- **Config:** `claude_desktop_config.json`, top-level `mcpServers`, **stdio only** (`command`, `args`):
  macOS `~/Library/Application Support/Claude/`, Windows `%APPDATA%\Claude\`. The docs give no Linux path; the writer uses
  `~/.config/Claude/claude_desktop_config.json`, the folder Claude Desktop's Linux builds use. Restart the app after
  writing. Remote servers go through *Settings > Connectors*, which needs a public HTTPS URL (see claude.ai below), so
  the supported path is local stdio: one `mem serve --store <name>=<path>` process per store.

  ```json
  {"mcpServers": {"mem-work": {"command": "uv", "args": ["run", "--project", "<repo>", "mem", "serve", "--store", "work=<path>"]}}}
  ```

  `mem mcp-config --harness claude-desktop --transport http` refuses with that reason. Stdio opens the stores in a second
  process next to `mem serve --http`; SQLite allows it, but run the stdio form only when the HTTP server is stopped, so two
  processes do not each load the models onto the GPU.
- **Skill: snippet.** Desktop's skills come from a zip uploaded in *Customize > Skills* (the Claude support article on
  skills), not from a folder, so there is nothing for `mem setup` to copy. Run `mem instructions --harness claude-desktop`
  and paste the text into your personal preferences (Settings > Profile; the menu name is from memory of the UI, not from a
  docs page). Or zip `skills/mem` yourself and upload it.
- **Docs:** <https://modelcontextprotocol.io/docs/develop/connect-local-servers>,
  <https://modelcontextprotocol.io/docs/develop/connect-remote-servers>,
  <https://support.claude.com/en/articles/12512180-use-skills-in-claude> (checked 2026-10-26).

## claude.ai (web): unsupported

Custom connectors on claude.ai reach MCP servers from Anthropic's cloud, so the server needs an internet-reachable HTTPS
URL. That breaks the rule that mem runs locally and is not exposed. There is no config writer and no snippet. Use Claude
Desktop with the local stdio server, or Claude Code. (Exposing the server with Tailscale Funnel would work and is
a decision for you, not a default; see ADR 0018.)

**Docs:** <https://modelcontextprotocol.io/docs/develop/connect-remote-servers> (checked 2026-10-26).

## Any other client

`mem mcp-config --harness generic` prints both forms (HTTP with headers, and stdio) in the JSON shape most clients accept;
rename `type`, `url` and `headers` as the client's docs say (`httpUrl`, `serverUrl`, `streamable-http`). For the rules, copy
[`docs/instructions.md`](instructions.md), which is `mem instructions --harness generic` for the current `SKILL.md`; a
test fails when it is out of date.
