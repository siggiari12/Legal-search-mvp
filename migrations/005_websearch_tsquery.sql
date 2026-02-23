-- Migration 005: Replace plainto_tsquery with websearch_to_tsquery
--
-- plainto_tsquery produces a strict AND query over every token in the input:
--   "Hvað er þjófnaður samkvæmt hegningarlögum?"
--   → 'hvað' & 'er' & 'þjófnaður' & 'samkvæmt' & 'hegningarlögum'
-- This matches only documents that contain ALL of those tokens, which for
-- natural-language questions is almost never satisfied.
--
-- websearch_to_tsquery uses web-search semantics:
--   • Unquoted adjacent words remain AND — preserving precision for short queries.
--   • The function is lenient about punctuation (?, !, …) — safe for questions.
--   • Critically, callers may use OR syntax ("þjófnaður | hegningarlög") if
--     needed in the future without a schema change.
--   • For the current natural-language queries it degrades gracefully, and it
--     will pick up key content words that plainto_tsquery was rejecting because
--     co-occurring function words were not present in legal article text.
--
-- No scoring weights changed.  No index changes.

-- ── match_documents_fts (updated) ────────────────────────────────────────────

CREATE OR REPLACE FUNCTION match_documents_fts(
    query_text  text,
    match_count int DEFAULT 50
)
RETURNS TABLE (
    id              text,
    law_reference   text,
    article_locator text,
    "text"          text,
    rank            double precision
)
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    tsq tsquery;
BEGIN
    IF length(query_text) > 0 THEN
        tsq := websearch_to_tsquery('simple', query_text);
    END IF;

    IF tsq IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        d.id,
        d.law_reference,
        d.article_locator,
        d."text",
        ts_rank_cd(d.search_tsv, tsq)::double precision AS rank
    FROM   documents d
    WHERE  d.search_tsv @@ tsq
    ORDER  BY rank DESC
    LIMIT  match_count;
END;
$$;


-- ── match_documents_hybrid (updated) ─────────────────────────────────────────

CREATE OR REPLACE FUNCTION match_documents_hybrid(
    query_embedding   vector(1536),
    query_text        text,
    match_count       int DEFAULT 8,
    vector_candidates int DEFAULT 50,
    fts_candidates    int DEFAULT 50
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
        FROM   documents d
        ORDER  BY d.embedding <=> query_embedding
        LIMIT  vector_candidates
    ),
    fts_cands AS (
        SELECT d.id
        FROM   documents d
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
            d.id                                                                 AS doc_id,
            d.law_reference,
            d.article_locator,
            d."text",
            (1 - (d.embedding <=> query_embedding))::double precision            AS vec_sim_val,
            CASE
                WHEN tsq IS NOT NULL
                THEN ts_rank_cd(d.search_tsv, tsq)::double precision
                ELSE 0.0
            END                                                                  AS fts_raw
        FROM all_ids ai
        JOIN documents d ON d.id = ai.id
    ),
    max_fts AS (
        SELECT GREATEST(MAX(s.fts_raw), 1e-10) AS mx
        FROM scored s
    )
    SELECT
        s.doc_id                                                                 AS id,
        s.law_reference,
        s.article_locator,
        s."text",
        (0.7 * s.vec_sim_val + 0.3 * (s.fts_raw / mf.mx))::double precision    AS similarity,
        s.fts_raw                                                                AS fts_score,
        s.vec_sim_val                                                            AS vector_sim
    FROM scored s
    CROSS JOIN max_fts mf
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;
