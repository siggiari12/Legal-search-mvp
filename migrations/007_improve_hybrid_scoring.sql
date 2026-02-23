-- Migration 007_improve_hybrid_scoring.sql
--
-- Defensive fix: clamp vec_sim_val so floating-point edge cases in pgvector
-- (cosine distance slightly > 1.0) never produce a negative combined score.
--
-- Only the vec_sim_val expression changes. Signature, RETURNS TABLE,
-- weights, FTS logic, CTEs, ordering and limits are identical to 005.

CREATE OR REPLACE FUNCTION public.match_documents_hybrid(
    query_embedding   vector,
    query_text        text,
    match_count       integer DEFAULT 8,
    vector_candidates integer DEFAULT 50,
    fts_candidates    integer DEFAULT 50
)
RETURNS TABLE (
    id              text,
    law_reference   text,
    article_locator text,
    "text"          text,
    similarity      double precision,
    fts_score       double precision,
    vector_sim      double precision
)
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    tsq tsquery;
BEGIN
    IF length(query_text) > 0 THEN
        tsq := websearch_to_tsquery('simple', query_text);
    END IF;

    RETURN QUERY
    WITH
    vec_cands AS (
        SELECT d.id
        FROM   public.documents d
        ORDER  BY d.embedding <=> query_embedding
        LIMIT  vector_candidates
    ),
    fts_cands AS (
        SELECT d.id
        FROM   public.documents d
        WHERE  tsq IS NOT NULL
          AND  d.search_tsv @@ tsq
        ORDER  BY ts_rank_cd(d.search_tsv, tsq) DESC
        LIMIT  fts_candidates
    ),
    all_ids AS (
        SELECT vc.id FROM vec_cands vc
        UNION
        SELECT fc.id FROM fts_cands fc
    ),
    scored AS (
        SELECT
            d.id                                                                        AS doc_id,
            d.law_reference,
            d.article_locator,
            d."text",
            (GREATEST(0.0, 1.0 - (d.embedding <=> query_embedding)))::double precision AS vec_sim_val,
            CASE
                WHEN tsq IS NOT NULL
                THEN ts_rank_cd(d.search_tsv, tsq)::double precision
                ELSE 0.0
            END                                                                         AS fts_raw
        FROM all_ids ai
        JOIN public.documents d ON d.id = ai.id
    ),
    max_fts AS (
        SELECT GREATEST(MAX(s.fts_raw), 1e-10) AS mx
        FROM scored s
    )
    SELECT
        s.doc_id                                                                     AS id,
        s.law_reference,
        s.article_locator,
        s."text",
        (0.7 * s.vec_sim_val + 0.3 * (s.fts_raw / mf.mx))::double precision        AS similarity,
        s.fts_raw                                                                    AS fts_score,
        s.vec_sim_val                                                                AS vector_sim
    FROM scored s
    CROSS JOIN max_fts mf
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Record migration
INSERT INTO schema_migrations (version)
VALUES ('007_improve_hybrid_scoring')
ON CONFLICT (version) DO NOTHING;

-- ----------------------------------------------------------------------------
-- EXPLAIN checks (run manually in Supabase SQL editor)
--
-- Vector path -- expect: Index Scan using idx_documents_embedding_hnsw
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT id FROM public.documents
-- ORDER BY embedding <=> (SELECT embedding FROM public.documents ORDER BY id LIMIT 1)
-- LIMIT 50;
--
-- FTS path -- expect: Bitmap Index Scan on idx_documents_search_tsv
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT id FROM public.documents
-- WHERE search_tsv @@ websearch_to_tsquery('simple', 'uppsögn ráðningarsamnings')
-- ORDER BY ts_rank_cd(search_tsv, websearch_to_tsquery('simple', 'uppsögn ráðningarsamnings')) DESC
-- LIMIT 50;
-- ----------------------------------------------------------------------------
