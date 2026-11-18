-- Derived date annotation (ADR 0021, `ingest.resolve_dates`): the relative time expressions of a
-- chunk resolved against its note or session date, e.g. `yesterday [= 2023-03-09]`. NULL when the
-- switch is off or nothing resolved. The verbatim `text` is never touched; the annotation is its own
-- FTS column (default bm25 weight 1.0, so a store without annotations ranks as before).
ALTER TABLE chunks ADD COLUMN dates TEXT;

DROP TRIGGER chunks_ai;
DROP TRIGGER chunks_ad;
DROP TRIGGER chunks_au;
DROP TABLE chunks_fts;
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  ctx, text, dates, content='chunks', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2');
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, ctx, text, dates) VALUES (new.id, new.ctx, new.text, new.dates); END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, ctx, text, dates) VALUES ('delete', old.id, old.ctx, old.text, old.dates); END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, ctx, text, dates) VALUES ('delete', old.id, old.ctx, old.text, old.dates);
  INSERT INTO chunks_fts(rowid, ctx, text, dates) VALUES (new.id, new.ctx, new.text, new.dates); END;
INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild');
