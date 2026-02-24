-- Migration 009: Switch match_documents_hybrid to use search_tsv_lemmatized
--
-- Changes from 007:
--   - fts_cands and scoring now reference search_tsv_lemmatized instead of search_tsv
--   - query_text is expected to be pre-lemmatised by the Python caller
--     (retrieve_hybrid calls get_lemmatized_text before invoking this RPC)
--
-- Everything else is identical to migration 007:
--   same signature, same weights (0.7/0.3), same HNSW vector path, same ordering.

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
        -- Uses search_tsv_lemmatized so lemmatised query tokens match
        -- lemmatised document tokens regardless of inflected form.
        SELECT d.id
        FROM   public.documents d
        WHERE  tsq IS NOT NULL
          AND  d.search_tsv_lemmatized @@ tsq
        ORDER  BY ts_rank_cd(d.search_tsv_lemmatized, tsq) DESC
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
                THEN ts_rank_cd(d.search_tsv_lemmatized, tsq)::double precision
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

INSERT INTO schema_migrations (version)
VALUES ('009_hybrid_use_lemmatized_tsv')
ON CONFLICT (version) DO NOTHING;
