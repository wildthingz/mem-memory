-- P7 poisoning defence (B-H1, mem.guard). `tool_sourced` = 1 when `remember` got a URL or
-- `tool:<name>` source: the row is second-hand and ranks below first-hand rows. `quarantine` holds
-- the reason a memory was quarantined (prompt injection); such a memory is never recalled and is
-- not in the core index until `mem quarantine --release` sets it back to NULL.
ALTER TABLE nodes ADD COLUMN tool_sourced INTEGER NOT NULL DEFAULT 0 CHECK (tool_sourced IN (0, 1));
ALTER TABLE nodes ADD COLUMN quarantine TEXT;
CREATE INDEX nodes_quarantine ON nodes(id) WHERE quarantine IS NOT NULL;
