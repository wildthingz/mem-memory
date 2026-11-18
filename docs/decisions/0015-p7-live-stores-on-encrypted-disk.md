# 0015: Live stores rely on full-disk encryption, not SQLCipher; `mem doctor` fails without it

- **Status:** Accepted (ruling from the maintainer on review MEM2-57, finding F6)
- **Date:** 2026-10-25 · **Task:** MEM2-111 (fix for MEM2-55) · **By:** Vault
- Spec: benchmarks-v2 B-H1 ("the data is unreadable without the key"), architecture-v2 §6 P7.
  Code: `src/mem/doctor.py` `check_at_rest`, `mem doctor --allow-unencrypted`. Runbook: `deliverables/runbooks/p7-backup.md` §0–§1.

> **Update 2026-10-25 (MEM2-164):** encryption at rest is deferred (owner's backlog). `mem doctor`'s at-rest check stays but returns
> **warn** ("deferred (owner's backlog); see p7-backup.md") until the owner sets `security.at_rest_required: true`, which makes it fail as before.

## Context
The two live stores (`personal.db`, `work.db`) are SQLite with sqlite-vec. `mem serve` starts unattended at login,
so whatever key protects the live stores must be available without a person typing it. In MEM2-55 the encryption
choice lived only in the runbook, and `mem doctor`'s `at-rest` row was only a `warn` (exit 0 on a plain disk), so
nothing enforced B-H1.

## Decision
The stores stay plain SQLite on a volume that is encrypted underneath: full-disk LUKS/dm-crypt (or FileVault on a
Mac), or gocryptfs for the store folder. `mem doctor` checks the store folder's volume (`findmnt` + `lsblk`), and the
`at-rest` row is `fail` (exit 1) when the folder is not on an encrypted volume **or when it cannot tell**.
`--allow-unencrypted` (test machines only) turns that into a `warn`. Each time it is used on an unencrypted folder,
it writes a line to `~/.local/state/mem/doctor.jsonl` (mode 600): `ts`, `event: at-rest-opt-out`, `path`, `detail`.

## Alternatives considered
- **SQLCipher for the live stores:** the key would have to sit on the same disk so `mem serve` can start
  unattended. It only protects anything if that disk is encrypted anyway. It also replaces the `sqlite3` driver in
  about 27 places (serve, render, doctor, every `bench` reader) and needs a SQLCipher build that loads sqlite-vec.
  It adds dependency risk and gives no protection that disk encryption doesn't already give.
- **Keep `at-rest` a `warn`:** this is what MEM2-57 F6 found. An unencrypted host passes `mem doctor`, so B-H1 is
  only documented, never checked.
- **Fail without an opt-out:** CI, containers and tmpfs test machines could never get a green `mem doctor`.
  People would learn to ignore its exit code, and that hides the other rows' failures too.

## Consequences
- One mechanism covers the stores, WAL/SHM files, the rendered vault, the audit log and swap, not just the `.db` files.
  No new dependency, and the store format is unchanged.
- Cost: the data is protected only while the machine is off or locked out. Once the volume is unlocked (logged in),
  any process running as the user can read the stores. Someone who copies the `.db` files off a running host gets
  plaintext. SQLCipher with a co-located key would not stop that either.
- Cost: a host without LUKS has to re-install with "Encrypt system" or set up gocryptfs, which has a mount-before-serve
  step after every login (runbook §1), and `shred` cannot erase old plaintext on SSDs or copy-on-write filesystems.
- Cost: detection is Linux-only (`findmnt`/`lsblk`, dm-crypt, gocryptfs/cryfs/ecryptfs). On macOS or any unknown
  setup the row fails as "cannot tell", so FileVault users must use the opt-out until a macOS probe exists.
- Backups are encrypted on their own (AES-256-GCM `.mbk`, runbook §0), so they can leave the encrypted disk.

## Revisit if
The stores must be readable by `mem serve` on a host whose disk cannot be encrypted, the stores are copied to
places outside the encrypted volume (sync, cloud), or a threat appears that needs protection while the user is logged in.
That last one needs a key held outside the disk (TPM, keyring after login) and then SQLCipher.
