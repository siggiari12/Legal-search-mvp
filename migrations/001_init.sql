-- Legal Search MVP - Initial Schema
-- Requires: PostgreSQL 14+ with pgvector extension

-- Enable pgvector extension (must be done by superuser or via Supabase dashboard)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Documents Table
-- Stores metadata for each law document
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source information
    source TEXT NOT NULL DEFAULT 'Althingi',
    document_type TEXT NOT NULL DEFAULT 'law',

    -- Law metadata
    title TEXT NOT NULL,
    law_number TEXT NOT NULL,           -- e.g., "33" (number only)
    law_year TEXT NOT NULL,             -- e.g., "1944"
    law_reference TEXT NOT NULL,        -- e.g., "33/1944" (combined)
    publication_date DATE,

    -- Versioning
    version_tag TEXT NOT NULL,          -- e.g., "2024-01-15" or "lagasafn-2024-01"

    -- Full text (canonical form)
    full_text TEXT NOT NULL,
    full_text_normalized TEXT,          -- Lowercased for search

    -- Links
    canonical_url TEXT,

    -- Flexible metadata
    metadata_json JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Chunks Table
-- Stores searchable chunks with embeddings
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Parent document
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Chunk content
    chunk_text TEXT NOT NULL,                    -- Canonical form
    chunk_text_normalized TEXT,                  -- Lowercased for keyword search

    -- Location within document
    locator TEXT NOT NULL,                       -- e.g., "Lög nr. 33/1944 - 1. gr., 2. mgr."
    article_number TEXT,                         -- e.g., "1" (for direct lookup)
    paragraph_number TEXT,                       -- e.g., "2"

    -- Denormalized law reference for faster queries
    law_number TEXT,                             -- e.g., "33"
    law_year TEXT,                               -- e.g., "1944"
    law_reference TEXT,                          -- e.g., "33/1944"

    -- Vector embedding (OpenAI text-embedding-3-small = 1536 dimensions)
    embedding vector(1536),

    -- Chunk ordering within document
    chunk_index INT NOT NULL DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Indexes
-- ============================================================

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_law_reference
    ON documents (law_reference);

CREATE INDEX IF NOT EXISTS idx_documents_law_number_year
    ON documents (law_number, law_year);

CREATE INDEX IF NOT EXISTS idx_documents_version_tag
    ON documents (version_tag);

-- Chunks indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_law_reference
    ON chunks (law_reference);

CREATE INDEX IF NOT EXISTS idx_chunks_article_number
    ON chunks (law_reference, article_number);

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index
    ON chunks (document_id, chunk_index);

-- Full-text search index on normalized text
CREATE INDEX IF NOT EXISTS idx_chunks_text_search
    ON chunks USING gin(to_tsvector('simple', chunk_text_normalized));

-- Vector similarity search index (IVFFlat)
-- Note: This index works best with 100+ rows. For small datasets, exact search is used.
-- Lists parameter should be sqrt(num_rows) for optimal performance
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================
-- Query Logs (Privacy-First)
-- For debugging and basic analytics without storing sensitive data
-- ============================================================
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query metadata (not the actual query text)
    query_hash TEXT NOT NULL,           -- SHA256 prefix of query
    query_length INT NOT NULL,          -- Length for analytics

    -- Search results
    chunk_count INT,                    -- Number of chunks retrieved
    search_type TEXT,                   -- 'direct_reference', 'hybrid', 'vector', 'keyword'

    -- Validation
    validation_passed BOOLEAN,
    retry_count INT DEFAULT 0,
    failure_reason TEXT,                -- If validation failed

    -- Response metadata
    confidence TEXT,                    -- 'high', 'medium', 'low', 'none'
    citation_count INT,

    -- Rate limiting (hashed for privacy)
    ip_hash TEXT,

    -- Timing
    search_duration_ms INT,
    llm_duration_ms INT,
    total_duration_ms INT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for rate limiting lookups
CREATE INDEX IF NOT EXISTS idx_query_logs_ip_hash_time
    ON query_logs (ip_hash, created_at);

-- ============================================================
-- Utility Functions
-- ============================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Migration tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Record this migration
INSERT INTO schema_migrations (version)
VALUES ('001_init')
ON CONFLICT (version) DO NOTHING;
