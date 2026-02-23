-- Migration 004: Add full-text search and server-side hybrid retrieval
--
-- Adds a stored tsvector column on documents (auto-maintained by Postgres),
-- a GIN index for fast FTS queries, and two RPC functions:
--
--   match_documents_fts    — pure FTS retrieval, ordered by ts_rank_cd
--   match_documents_hybrid — combined HNSW vector + FTS retrieval
--
-- Hybrid scoring: combined = 0.70 * vector_sim + 0.30 * (fts_score / max_fts)
-- Returns 'similarity' = combined score for downstream compatibility.
-- Degrades gracefully to pure vector when no query tokens are >= 4 chars.
--
-- Uses 'simple' text-search config (lowercase only, no stemming) — correct
-- for Icelandic since standard Postgres has no Icelandic stemmer.
--
-- No schema_migrations bookkeeping (table may not exist).

-- ── 1. Generated tsvector column ──────────────────────────────────────────────
-- Stored so values are pre-computed and kept in sync automatically.
-- Concatenates article text + law_reference + article_locator for maximum recall.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS search_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('simple',
            coalesce("text",          '') || ' ' ||
            coalesce(law_reference,   '') || ' ' ||
            coalesce(article_locator, '')
        )
    ) STORED;


-- ── 2. GIN index for fast FTS queries ─────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_documents_search_tsv
    ON documents USING GIN (search_tsv);


-- ── 3. match_documents_fts ────────────────────────────────────────────────────
-- Pure FTS retrieval.  Builds an OR-tsquery from query tokens >= 4 chars to
-- maximise recall (individual inflected forms match the OR branches).
-- Returns empty result set when no valid tokens are found.

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
    tsq_text text;
    tsq      tsquery;
BEGIN
    -- Build OR-tsquery from tokens with at least 4 characters
    SELECT string_agg(lexeme, ' | ')
    INTO   tsq_text
    FROM   unnest(to_tsvector('simple', query_text))
    WHERE  length(lexeme) >= 4;

    IF tsq_text IS NULL THEN
        RETURN;  -- no qualifying tokens; return empty
    END IF;

    tsq := to_tsquery('simple', tsq_text);

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


-- ── 4. match_documents_hybrid ─────────────────────────────────────────────────
-- Server-side hybrid: HNSW vector search UNION Postgres FTS, scored and merged.
--
-- Algorithm:
--   1. Take top vector_candidates from HNSW (ORDER BY embedding <=>).
--   2. Take top fts_candidates by ts_rank_cd (OR-query, tokens >= 4 chars).
--   3. Union candidate ids; join back to documents for full row data.
--   4. Score each candidate:
--        vec_sim  = 1 - cosine_distance
--        fts_raw  = ts_rank_cd score (0.0 if no tsq)
--        combined = 0.70 * vec_sim + 0.30 * (fts_raw / max_fts)
--   5. Return top match_count by combined score.
--
-- Returns 'similarity' = combined score (downstream uses this field).
-- Extra fields 'fts_score' and 'vector_sim' are available for debugging.
-- Falls back to pure vector (fts_raw=0 for all) when tsq IS NULL.

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
        tsq := plainto_tsquery('simple', query_text);
    END IF;

    RETURN QUERY
    WITH
    vec_cands AS (
        SELECT d.id
        FROM documents d
        ORDER BY d.embedding <=> query_embedding
        LIMIT vector_candidates
    ),
    fts_cands AS (
        SELECT d.id
        FROM documents d
        WHERE tsq IS NOT NULL
          AND d.search_tsv @@ tsq
        ORDER BY ts_rank_cd(d.search_tsv, tsq) DESC
        LIMIT fts_candidates
    ),
    all_ids AS (
        SELECT vc.id FROM vec_cands vc
        UNION
        SELECT fc.id FROM fts_cands fc
    ),
    scored AS (
        SELECT
            d.id                           AS doc_id,
            d.law_reference,
            d.article_locator,
            d."text",
            (1 - (d.embedding <=> query_embedding))::double precision AS vec_sim_val,
            CASE
                WHEN tsq IS NOT NULL
                THEN ts_rank_cd(d.search_tsv, tsq)::double precision
                ELSE 0.0
            END AS fts_raw
        FROM all_ids ai
        JOIN documents d ON d.id = ai.id
    ),
    max_fts AS (
        SELECT GREATEST(MAX(s.fts_raw), 1e-10) AS mx
        FROM scored s
    )
    SELECT
        s.doc_id                                AS id,
        s.law_reference,
        s.article_locator,
        s."text",
        (0.7 * s.vec_sim_val + 0.3 * (s.fts_raw / mf.mx))::double precision AS similarity,
        s.fts_raw                               AS fts_score,
        s.vec_sim_val                            AS vector_sim
    FROM scored s
    CROSS JOIN max_fts mf
    ORDER BY similarity DESC
    LIMIT match_count;

END;
$$;
