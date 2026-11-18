# Backups and encryption

## Encryption at rest

The live stores are plain SQLite files. mem does not encrypt them itself, because `mem serve` starts
unattended and a key would have to sit on the same disk as the store. Use an encrypted disk instead:
full-disk LUKS (dm-crypt), or gocryptfs for the store folder. `mem doctor` reports the `at-rest` row
for the folder holding the stores (`~/.local/share/mem`):

```
[ok  ] at-rest: ... on dm-crypt (...)
[ok  ] at-rest: ... on gocryptfs (...)
[FAIL] at-rest: ... is NOT encrypted at rest (...)
```

It also fails when it cannot tell (no `findmnt` or `lsblk`, tmpfs). On a throwaway machine,
`mem doctor --allow-unencrypted` reports the row as a warning and appends a line to
`~/.local/state/mem/doctor.jsonl`; do not use it on the machine that holds your real memories.

To encrypt the store folder without reinstalling, with gocryptfs (stop the server first):

```bash
systemctl --user stop mem
mkdir -p ~/.mem-cipher && gocryptfs -init ~/.mem-cipher        # keep the passphrase and master key in a password manager
mv ~/.local/share/mem ~/.local/share/mem.plain && mkdir ~/.local/share/mem
gocryptfs ~/.mem-cipher ~/.local/share/mem
cp -a ~/.local/share/mem.plain/. ~/.local/share/mem/
uv run --extra models mem doctor            # expect [ok  ] at-rest: ... on gocryptfs
systemctl --user start mem
shred -u ~/.local/share/mem.plain/*.db* && rm -r ~/.local/share/mem.plain
```

Mount it after every login, before `mem serve` starts; otherwise the server creates new empty stores
in the empty mount point (`mountpoint ~/.local/share/mem` checks). `shred` cannot overwrite in place on
SSDs or copy-on-write filesystems, so the old plaintext may survive in free blocks; only full-disk
encryption from the start removes that risk.

## Encrypted backups

Backups are `VACUUM INTO` snapshots of each store (consistent while `mem serve` writes), encrypted with
AES-256-GCM under a per-file key derived from a key file. A wrong key, a flipped bit, reordered chunks
or a cut-off tail are all rejected. The `.mbk` files can go anywhere (a NAS, a USB disk, the cloud).

1. **Create the key, once.**

   ```bash
   uv run --extra models mem backup-key
   ```

   It writes `~/.config/mem/backup.key` (mode 600, 64 hex characters). **Copy it somewhere off this
   machine now.** Without it no backup can be restored, and a dead disk takes the key with it. Running
   it again refuses, because a new key would orphan every earlier backup. Every command refuses a key
   file that others can read.

2. **Back up.**

   ```bash
   uv run --extra models mem backup
   ls -l ~/.local/share/mem/backups/
   ```

   One file per store, `<store>-<UTC timestamp>.mbk`, mode 600. By default the newest 14 per store are
   kept (`--keep`). Even an empty store is about 1.2 MB, because the vector index allocates its chunks
   up front.

3. **Nightly, with systemd.**

   ```bash
   cp deploy/mem-backup.service deploy/mem-backup.timer ~/.config/systemd/user/
   systemctl --user daemon-reload && systemctl --user enable --now mem-backup.timer
   systemctl --user list-timers mem-backup.timer
   ```

   The timer runs at 03:30 local; a run missed while the machine was off runs at the next boot. The
   unit assumes the repository is at `~/code/mem`; edit its `WorkingDirectory` if it is elsewhere.
   For an off-machine copy, add an `ExecStartPost=` line to your copy of the service that runs
   `rsync` to wherever you keep them. Never keep the key next to the backups.

At most one day of writes is lost on a restore (the vault is its own source: re-import it).

## Restore

```bash
systemctl --user stop mem
cd ~/.local/share/mem && mkdir -p broken && mv personal.db personal.db-wal personal.db-shm broken/ 2>/dev/null; cd -
uv run --extra models mem restore ~/.local/share/mem/backups/personal-<stamp>.mbk --to ~/.local/share/mem/personal.db
uv run --extra models mem doctor --store ~/.local/share/mem/personal.db
systemctl --user start mem
```

`mem restore` decrypts into a **new** file (it refuses to overwrite) and verifies the sha256 and
SQLite's `integrity_check`. Move the old `-wal` and `-shm` files aside with the store: a stale WAL next
to the restored file would be applied to it and corrupt it. On a new machine, first put the key back:
`install -m 600 /dev/stdin ~/.config/mem/backup.key`, paste the 64 hex characters, then Ctrl-D.

### Check that backups work (do this once, then monthly)

Restore each store to a scratch folder and compare its canonical dump with the live one:

```bash
systemctl --user stop mem
uv run --extra models mem backup
D=$(mktemp -d)
for s in personal work; do
  F=$(ls -1 ~/.local/share/mem/backups/$s-*.mbk | tail -1)
  uv run --extra models mem restore "$F" --to $D/$s.db
  uv run --extra models bench dump ~/.local/share/mem/$s.db --out $D/$s-live.jsonl
  uv run --extra models bench dump $D/$s.db --out $D/$s-restored.jsonl
  diff $D/$s-live.jsonl $D/$s-restored.jsonl && echo "$s: 0 diff lines"
done
systemctl --user start mem
rm -r $D        # holds plaintext copies of both stores
```

Pass: `0 diff lines` for both stores. A backup restored with a different key fails with
`cannot decrypt: wrong key or corrupted backup` and writes nothing.

| Message | Meaning | Do |
|---|---|---|
| `cannot read key file ...` | no key at `--key-file` | restore the key from your password manager |
| `... is readable by others (mode ...)` | key file is not mode 600 | `chmod 600 ~/.config/mem/backup.key` |
| `cannot decrypt: wrong key or corrupted backup` | other key, or damaged file | try the right key, otherwise an older `.mbk` |
| `backup is truncated` / `not a mem backup (bad magic)` | incomplete copy, or not a `.mbk` | use another copy |
| `... exists; restore only writes a new file` | `--to` exists | pick a new path, or move the old store aside |
| `snapshot of ... failed integrity_check` | the live store is damaged; no backup was written | keep the last good `.mbk` files (they are not pruned on failure) and restore |
