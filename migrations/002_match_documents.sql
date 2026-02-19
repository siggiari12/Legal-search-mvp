-- Migration 002: match_documents RPC function
--
-- Creates the pgvector similarity search function used by answer_query.py
-- and test_vector_search.py.
--
-- Depends on: documents table with embedding vector(1536) column (from ingestion).

-- Vector index on documents.embedding (IVFFlat, cosine distance).
-- lists=134 ≈ sqrt(17911) for the current corpus size.
CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 134);

-- Similarity search RPC.
-- Returns the top match_count rows ordered by cosine similarity (descending).
-- similarity = 1 - cosine_distance, so 1.0 is identical, 0.0 is orthogonal.
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_count     int DEFAULT 8
)
RETURNS TABLE (
    id              text,
    law_reference   text,
    article_locator text,
    "text"          text,
    similarity      double precision
)
LANGUAGE sql STABLE
AS $$
    SELECT
        id,
        law_reference,
        article_locator,
        "text",
        1 - (embedding <=> query_embedding) AS similarity
    FROM documents
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Record migration
INSERT INTO schema_migrations (version)
VALUES ('002_match_documents')
ON CONFLICT (version) DO NOTHING;
