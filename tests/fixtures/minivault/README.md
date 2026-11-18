# minivault fixture

`vault/` is the frozen snapshot S1 of the P4 import tests. `config.yaml` holds the per-folder routing
override (Q3). `expected_s1.json` is the hand-worked store dump after importing S1 with `tz="UTC"`
and every file's mtime set to 2026-01-15T12:00:00Z.
Later snapshots (rename, delete, edits, restore) are applied in tests/test_p4_vault_fixture.py. Decision: docs/decisions/0010-p4-import-mapping-routing.md.

`node_modules/pkg/README.md` is skipped by the importer but git-ignored, so the tests create it in
their working copy. Do not run these tests against a real vault.
