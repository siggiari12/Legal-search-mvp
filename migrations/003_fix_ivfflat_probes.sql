-- Migration 003: Fix vector search recall by dropping the IVFFlat index
--
-- Root cause: IVFFlat with lists=134 and the default probes=1 causes
-- match_documents to search only 1 of 134 clusters (~134 of 17 911 rows).
-- For queries whose vector falls near a small cluster this returns very few
-- results (observed: as few as 7 rows regardless of match_count) and
-- completely misses relevant documents in other clusters.
--
-- Approaches tried:
--   - SET LOCAL ivfflat.probes = 20 inside a PLPGSQL VOLATILE function:
--     rejected by PostgreSQL ("SET is not allowed in a non-volatile function"
--     when STABLE, and then statement-timeout on Supabase free tier when VOLATILE).
--
-- Final fix: drop the IVFFlat index entirely.
--   At 17 911 rows, a sequential scan for exact nearest-neighbour costs
--   ~10-50 ms and gives perfect recall.  IVFFlat approximate search is only
--   beneficial above ~100 K rows.  The index was premature optimisation that
--   actively degraded recall.

DROP INDEX IF EXISTS idx_documents_embedding;

-- Restore the original STABLE SQL function (no PLPGSQL wrapper needed).
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
VALUES ('003_fix_ivfflat_probes')
ON CONFLICT (version) DO NOTHING;
