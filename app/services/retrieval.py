"""
retrieval.py — Python wrappers for Supabase retrieval RPCs.

Provides two retrieval functions:

  retrieve_vector  — pure HNSW vector search via match_documents RPC
  retrieve_hybrid  — server-side vector + FTS hybrid via match_documents_hybrid RPC

Both return a list of dicts.  retrieve_hybrid adds 'fts_score' and 'vector_sim'
keys alongside the standard 'similarity' (= combined score) used downstream.
"""

from __future__ import annotations


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

    Returns list of dicts with keys:
        id, law_reference, article_locator, text,
        similarity  (= combined score, used by assess_retrieval),
        fts_score   (raw ts_rank_cd score, for debugging),
        vector_sim  (cosine similarity, for debugging)

    Requires migration 004_add_fts_and_hybrid.sql to be applied.
    """
    result = sb.rpc(
        "match_documents_hybrid",
        {
            "query_embedding":   embedding,
            "query_text":        query_text,
            "match_count":       top_k,
            "vector_candidates": vec_k,
            "fts_candidates":    fts_k,
        },
    ).execute()
    return result.data or []
