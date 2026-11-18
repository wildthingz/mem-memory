
-- keys: schema_version, scope ('personal'|'work'), fingerprint (JSON, §3.3),
--       vec_table ('vec_<fp8>'), gate (JSON: platt_a, platt_b, tau_keep, tau_high, gap,
--       theta_conf, fit_id, fitted_at, fingerprint_hash)
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- Timestamps: UTC 'YYYY-MM-DDTHH:MM:SSZ' everywhere (string order = time order).
-- Date-only input means 00:00 in the configured local time zone, converted to UTC.
CREATE TABLE nodes (
  id           INTEGER PRIMARY KEY,               -- shown as m412 / e45 / n67 / t9
  kind         TEXT NOT NULL CHECK (kind IN ('memory','entity','note','tag')),
  type         TEXT,     -- memory: fact|preference|decision|event|procedure|insight
                         -- entity: person|project|org|topic|place|thing ; note: note|file|chat_session
  name         TEXT NOT NULL,
  summary      TEXT,
  body         TEXT,                              -- verbatim memory text; NULL for notes
  date         TEXT,                              -- what the content is about (event date, note `created`)
  valid_from   TEXT NOT NULL,                     -- valid time start (inclusive)
  valid_to     TEXT,                              -- valid time end (exclusive); derived from the chain (§3.1)
  recorded_at  TEXT NOT NULL,                     -- ingestion time (when the store learned it)
  origin       TEXT NOT NULL CHECK (origin IN ('user','agent','import')),
  source_ref   TEXT,                              -- vault-relative path (import); optional label (memories)
  session      TEXT,                              -- MCP session id of the writing call (provenance)
  host         TEXT,                              -- client host label (provenance)
  content_hash TEXT,
  status       TEXT NOT NULL DEFAULT 'active'
               CHECK (status IN ('active','duplicate','retracted','deleted')),
  status_at    TEXT,
  pinned       INTEGER NOT NULL DEFAULT 0,
  CHECK (valid_to IS NULL OR valid_to >= valid_from)  -- equality = empty interval of a same-instant predecessor (N1)
);
CREATE INDEX nodes_live ON nodes(kind, status, valid_from, valid_to);
CREATE UNIQUE INDEX nodes_mem_hash ON nodes(content_hash) WHERE kind = 'memory' AND status = 'active' AND valid_to IS NULL;  -- open memories only (N2)
CREATE UNIQUE INDEX nodes_source   ON nodes(source_ref) WHERE origin = 'import' AND source_ref IS NOT NULL;
CREATE UNIQUE INDEX nodes_created_name ON nodes(kind, type, name COLLATE NOCASE)
  WHERE kind IN ('entity','tag') AND status = 'active' AND source_ref IS NULL;

-- Relation vocabulary and allowed endpoint kinds (checked by `link`).
CREATE TABLE rels (
  rel TEXT PRIMARY KEY, inverse TEXT,
  src_kinds TEXT NOT NULL, dst_kinds TEXT NOT NULL,   -- comma lists or '*'
  description TEXT NOT NULL
);
-- Seed: about(memory→*), supersedes(memory→memory), duplicate_of(memory→memory),
-- same_as(entity→entity), relates_to(*→*), links_to, embeds, tagged, part_of (import only),
-- plus frontmatter link keys (reports_to, project, attendee) added at import.

-- Edges are structural, not time-sliced: live iff removed_at IS NULL. Time lives on nodes.
CREATE TABLE edges (
  id          INTEGER PRIMARY KEY,
  src         INTEGER NOT NULL REFERENCES nodes(id),
  dst         INTEGER NOT NULL REFERENCES nodes(id),
  rel         TEXT    NOT NULL REFERENCES rels(rel),
  origin      TEXT NOT NULL CHECK (origin IN ('user','agent','import')),
  recorded_at TEXT NOT NULL,
  removed_at  TEXT,
  note        TEXT,
  CHECK (src <> dst)
);
CREATE UNIQUE INDEX edges_live ON edges(src, dst, rel) WHERE removed_at IS NULL;
CREATE INDEX edges_dst ON edges(dst, rel);

CREATE TABLE chunks (
  id           INTEGER PRIMARY KEY,               -- shown as c812 (sections, facts)
  node_id      INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  chunk_key    TEXT NOT NULL,   -- memory: 'body' | 'fact/<i>' | 'name'; note: '<heading path>#<dup n>/p<part>'
  ord          INTEGER NOT NULL,                  -- display order only; never an identifier
  key_kind     TEXT NOT NULL CHECK (key_kind IN ('body','section','fact','name')),
  heading_path TEXT,
  ctx          TEXT,                              -- contextual prefix: title + heading path, or title + summary
  text         TEXT NOT NULL,                     -- verbatim
  tokens       INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  props        TEXT,                              -- JSON: callout types, block ids, line range
  UNIQUE (node_id, chunk_key)
);

-- One row per wikilink occurrence. Logical edges stay deduplicated in `edges`.
CREATE TABLE link_occurrences (
  id       INTEGER PRIMARY KEY,
  edge_id  INTEGER NOT NULL REFERENCES edges(id),
  chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,   -- body link
  field    TEXT,                                              -- frontmatter key (frontmatter link)
  pos      INTEGER NOT NULL,                                  -- char offset in chunk text / list index
  anchor   TEXT, alias TEXT,
  CHECK ((chunk_id IS NULL) <> (field IS NULL))
);
CREATE UNIQUE INDEX occ_body ON link_occurrences(chunk_id, pos) WHERE chunk_id IS NOT NULL;
CREATE INDEX occ_edge ON link_occurrences(edge_id);

CREATE TABLE aliases (
  alias TEXT NOT NULL,
  alias_norm TEXT NOT NULL,                       -- NFKC, casefold, trimmed, single spaces, no [[ ]]
  node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  origin TEXT NOT NULL,                           -- 'import' rows are owned by the importer
  UNIQUE (alias_norm, node_id)                    -- the same alias may name several nodes (→ ask back)
);
CREATE INDEX aliases_norm ON aliases(alias_norm);
CREATE VIRTUAL TABLE aliases_tri USING fts5(alias_norm, content='aliases', content_rowid='rowid', tokenize='trigram');

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  ctx, text, content='chunks', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2');

CREATE TABLE props (
  node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  key TEXT NOT NULL, value TEXT NOT NULL, origin TEXT NOT NULL,
  PRIMARY KEY (node_id, key, value)
);
CREATE INDEX props_kv ON props(key, value);

CREATE TABLE source_files (
  path TEXT PRIMARY KEY, vault TEXT NOT NULL,
  node_id INTEGER NOT NULL REFERENCES nodes(id),
  content_hash TEXT NOT NULL, mtime REAL, imported_at TEXT NOT NULL
);

-- Every tool call (feeds the daily-use gate B-E4 and gold set PG-v2).
CREATE TABLE events (
  id INTEGER PRIMARY KEY, at TEXT NOT NULL,
  session TEXT, host TEXT,
  tool TEXT NOT NULL CHECK (tool IN ('recall','get','remember','link','forget','subjects')),
  args TEXT,          -- JSON request
  result TEXT,        -- JSON: returned ids + p, abstained, status, possible_conflicts ids, side effects
  latency_ms INTEGER
);
CREATE INDEX events_session ON events(session, at);

CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, ctx, text) VALUES (new.id, new.ctx, new.text); END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, ctx, text) VALUES ('delete', old.id, old.ctx, old.text); END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, ctx, text) VALUES ('delete', old.id, old.ctx, old.text);
  INSERT INTO chunks_fts(rowid, ctx, text) VALUES (new.id, new.ctx, new.text); END;
CREATE TRIGGER aliases_ai AFTER INSERT ON aliases BEGIN
  INSERT INTO aliases_tri(rowid, alias_norm) VALUES (new.rowid, new.alias_norm); END;
CREATE TRIGGER aliases_ad AFTER DELETE ON aliases BEGIN
  INSERT INTO aliases_tri(aliases_tri, rowid, alias_norm) VALUES ('delete', old.rowid, old.alias_norm); END;
CREATE TRIGGER aliases_au AFTER UPDATE ON aliases BEGIN
  INSERT INTO aliases_tri(aliases_tri, rowid, alias_norm) VALUES ('delete', old.rowid, old.alias_norm);
  INSERT INTO aliases_tri(rowid, alias_norm) VALUES (new.rowid, new.alias_norm); END;

INSERT INTO rels(rel, inverse, src_kinds, dst_kinds, description) VALUES
  ('about', NULL, 'memory', '*', 'memory is about the target'),
  ('supersedes', 'superseded_by', 'memory', 'memory', 'src replaces dst; dst valid_to is derived'),
  ('duplicate_of', NULL, 'memory', 'memory', 'src is a duplicate of dst'),
  ('same_as', 'same_as', 'entity', 'entity', 'both entities name the same thing'),
  ('relates_to', 'relates_to', '*', '*', 'generic relation'),
  ('links_to', NULL, '*', '*', 'wikilink (import)'),
  ('embeds', NULL, '*', '*', 'embedded note (import)'),
  ('tagged', NULL, '*', 'tag', 'tag assignment (import)'),
  ('part_of', NULL, '*', '*', 'structural containment (import only)');

INSERT INTO meta(key, value) VALUES ('schema_version', '1');
