-- Migration 006_fix_chunks_and_baseline.sql
--
-- Fix-forward replacement for the chunks table portion of 001_init.sql.
--
-- Why 001 cannot run:
--   001 defines chunks.document_id as UUID, but the live documents.id
--   column is TEXT. The type mismatch prevents the foreign key.
--
-- Why schema_migrations is included:
--   schema_migrations does not exist in the live DB, so the migration
--   runner returns an empty applied-set and retries every migration on
--   every run — including 001, which always fails. Creating the table
--   and marking 001 as applied breaks that loop.

-- 1. Bootstrap the migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT        PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Mark 001 as applied (chunks table is being created here instead)
INSERT INTO schema_migrations (version)
VALUES ('001_init')
ON CONFLICT (version) DO NOTHING;

-- 3. Create chunks table with TEXT document_id to match live documents.id
CREATE TABLE IF NOT EXISTS chunks (
    id                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id           TEXT        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text            TEXT        NOT NULL,
    chunk_text_normalized TEXT,
    locator               TEXT        NOT NULL,
    article_number        TEXT,
    paragraph_number      TEXT,
    law_number            TEXT,
    law_year              TEXT,
    law_reference         TEXT,
    embedding             vector(1536),
    chunk_index           INT         NOT NULL DEFAULT 0,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Indexes (mirrors 001_init.sql)
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_law_reference
    ON chunks (law_reference);

CREATE INDEX IF NOT EXISTS idx_chunks_article_number
    ON chunks (law_reference, article_number);

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index
    ON chunks (document_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_text_search
    ON chunks USING gin(to_tsvector('simple', coalesce(chunk_text_normalized, '')));

-- 5. Record this migration
INSERT INTO schema_migrations (version)
VALUES ('006_fix_chunks_and_baseline')
ON CONFLICT (version) DO NOTHING;
