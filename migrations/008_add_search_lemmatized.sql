-- Migration 008: Add lemmatized text column and empty tsvector column
--
-- Adds two plain columns (no GENERATED, no index) so the ALTER TABLE
-- statements are instant — no table scan required.
--
--   search_lemmatized      TEXT     — base-form text; filled by lemmatize_documents.py
--   search_tsv_lemmatized  TSVECTOR — filled by lemmatize_documents.py via
--                                     to_tsvector('simple', search_lemmatized)
--
-- The GIN index is created by lemmatize_documents.py using CREATE INDEX
-- CONCURRENTLY after all rows are written (cannot run inside a transaction).
--
-- Run order:
--   1. Apply this migration (run_migrations.py)      — adds empty columns
--   2. python scripts/lemmatize_documents.py         — fills columns + creates index
--   3. Apply migration 009 (run_migrations.py)       — switches hybrid function

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS search_lemmatized TEXT;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS search_tsv_lemmatized TSVECTOR;

INSERT INTO schema_migrations (version)
VALUES ('008_add_search_lemmatized')
ON CONFLICT (version) DO NOTHING;
