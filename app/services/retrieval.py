"""
retrieval.py — Python wrappers for Supabase retrieval RPCs.

Provides three retrieval functions:

  retrieve_vector        — pure HNSW vector search via match_documents RPC
  retrieve_hybrid        — server-side vector + FTS hybrid via match_documents_hybrid RPC
                           (sync, uses Supabase SDK — for scripts/CLI use)
  retrieve_hybrid_async  — same hybrid logic but async, uses asyncpg pool
                           (for FastAPI service use)

Both hybrid variants add 'fts_score' and 'vector_sim' keys alongside the
standard 'similarity' (= combined score) used downstream.
"""

from __future__ import annotations

from app.services.lemmatize import get_lemmatized_text


def retrieve_vector(sb, embedding: list[float], k: int) -> list[dict]:
    """
    Pure vector retrieval: top-k chunks by cosine similarity.

    Calls match_documents(query_embedding, match_count).

    Returns list of dicts with keys:
        id, law_reference, article_locator, text, similarity
    """
    result = sb.rpc(
        "match_documents",
        {"query_embedding": embedding, "match_count": k},
    ).execute()
    return result.data or []


def retrieve_hybrid(
    sb,
    embedding:  list[float],
    query_text: str,
    top_k:      int = 8,
    vec_k:      int = 50,
    fts_k:      int = 50,
) -> list[dict]:
    """
    Server-side hybrid retrieval combining HNSW vector search with Postgres FTS.

    Calls match_documents_hybrid which:
      - fetches vec_k vector candidates via HNSW cosine distance
      - fetches fts_k FTS candidates via ts_rank_cd (OR-query, 'simple' config,
        tokens >= 4 chars)
      - unions and deduplicates candidate ids
      - scores each: combined = 0.70 * vector_sim + 0.30 * (fts_score / max_fts)
      - returns top_k by combined score descending

    Degrades to pure vector when no query tokens are >= 4 chars (fts_score=0).

    query_text is lemmatised before being sent to the FTS engine so that
    inflected query forms (e.g. "hegningarlögum") match the base forms
    stored in search_tsv (e.g. "hegningarlög").

    Returns list of dicts with keys:
        id, law_reference, article_locator, text,
        similarity  (= combined score, used by assess_retrieval),
        fts_score   (raw ts_rank_cd score, for debugging),
        vector_sim  (cosine similarity, for debugging)

    Requires migration 004_add_fts_and_hybrid.sql to be applied.
    """
    lemmatized_query = get_lemmatized_text(query_text)
    result = sb.rpc(
        "match_documents_hybrid",
        {
            "query_embedding":   embedding,
            "query_text":        lemmatized_query,
            "match_count":       top_k,
            "vector_candidates": vec_k,
            "fts_candidates":    fts_k,
        },
    ).execute()
    return result.data or []


async def retrieve_hybrid_async(
    pool,
    embedding: list[float],
    query_text: str,
    top_k: int = 8,
    vec_k: int = 50,
    fts_k: int = 50,
) -> list[dict]:
    """
    Async version of retrieve_hybrid for use in the FastAPI service.
    Uses an asyncpg pool instead of the Supabase sync SDK.
    Lemmatises query_text via Greynir before passing to FTS.

    Returns list of dicts with keys:
        id, law_reference, article_locator, text,
        similarity  (= combined score),
        fts_score   (raw ts_rank_cd score),
        vector_sim  (cosine similarity)
    """
    lemmatized_query = get_lemmatized_text(query_text)
    # asyncpg passes vectors as strings in the '[x,y,...]' format
    embedding_list = embedding or []
    embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, law_reference, article_locator, "text",
                   similarity, fts_score, vector_sim
            FROM match_documents_hybrid(
                $1::vector, $2::text, $3::integer, $4::integer, $5::integer
            )
            """,
            embedding_str, lemmatized_query, top_k, vec_k, fts_k,
        )

    return [dict(row) for row in rows]
