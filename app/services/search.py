"""
Hybrid Search Module

This module implements hybrid retrieval combining:
  A) Vector search (pgvector) for semantic similarity
  B) Keyword/BM25-style search (PostgreSQL full-text + ILIKE) for exact matches

The system must NOT rely only on embeddings/vector search.
Hybrid search is required to improve recall, especially for:
  - Icelandic inflection (word forms)
  - Exact legal references (law numbers, article numbers)
  - Legal terminology

Merge rules:
  1. Exact law-number / article references get highest priority
  2. Otherwise combine vector + keyword results with deterministic reranking
"""

import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

from app.services.canonicalize import canonicalize, canonicalize_for_search


@dataclass
class Chunk:
    """Represents a searchable chunk of legal text."""
    id: str
    document_id: str
    chunk_text: str
    locator: str
    chunk_text_normalized: Optional[str] = None
    article_number: Optional[str] = None
    paragraph_number: Optional[str] = None
    law_number: Optional[str] = None
    law_year: Optional[str] = None
    # Search scores
    vector_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0


class SearchType(Enum):
    """Type of search that produced a result."""
    DIRECT_REFERENCE = "direct_reference"  # Exact law/article lookup
    VECTOR = "vector"                       # Semantic similarity
    KEYWORD = "keyword"                     # Full-text/ILIKE match
    HYBRID = "hybrid"                       # Combined


# Regex patterns for Icelandic legal references
LAW_NUMBER_PATTERNS = [
    r'(\d+)/(\d{4})',                           # 33/1944
    r'nr\.?\s*(\d+)/(\d{4})',                   # nr. 33/1944 or nr 33/1944
    r'lög\s+nr\.?\s*(\d+)/(\d{4})',             # lög nr. 33/1944
    r'l\.?\s*nr\.?\s*(\d+)/(\d{4})',            # l. nr. 33/1944 or l nr 33/1944
    r'laga\s+nr\.?\s*(\d+)/(\d{4})',            # laga nr. 33/1944
]

ARTICLE_PATTERNS = [
    r'(\d+)\.\s*gr\.?',                         # 1. gr. or 1. gr
    r'(\d+)\.\s*grein',                         # 1. grein
]

PARAGRAPH_PATTERNS = [
    r'(\d+)\.\s*mgr\.?',                        # 1. mgr. or 1. mgr
    r'(\d+)\.\s*málsgrein',                     # 1. málsgrein
]


def extract_law_reference(query: str) -> Optional[str]:
    """
    Extract law number from query if present.

    Returns:
        Law number in format "33/1944" or None if not found.
    """
    query_lower = query.lower()

    for pattern in LAW_NUMBER_PATTERNS:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            num, year = match.groups()
            return f"{num}/{year}"

    return None


def extract_article_reference(query: str) -> Optional[str]:
    """
    Extract article number from query if present.

    Returns:
        Article number (e.g., "12") or None if not found.
    """
    for pattern in ARTICLE_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def extract_search_keywords(query: str) -> List[str]:
    """
    Extract keywords from query for keyword search.

    Removes common Icelandic stop words and short words.
    Returns list of keywords for ILIKE matching.
    """
    # Common Icelandic stop words to filter out
    stop_words = {
        'og', 'eða', 'en', 'sem', 'að', 'um', 'til', 'frá', 'með',
        'við', 'í', 'á', 'ég', 'þú', 'hann', 'hún', 'það', 'við',
        'þeir', 'þær', 'þau', 'er', 'var', 'eru', 'voru', 'vera',
        'hvað', 'segir', 'lögum', 'lög', 'grein', 'málsgrein',
        'the', 'and', 'or', 'is', 'are', 'what', 'how',
    }

    # Normalize and split
    normalized = canonicalize_for_search(query.lower())
    words = normalized.split()

    # Filter: remove stop words and very short words
    keywords = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return keywords


class HybridSearcher:
    """
    Hybrid search engine combining vector and keyword search.

    This class provides the search interface for the Legal Search MVP.
    It does NOT depend on a specific database implementation - instead,
    it takes search functions as dependencies for testability.
    """

    def __init__(
        self,
        vector_search_fn,
        keyword_search_fn,
        direct_lookup_fn,
    ):
        """
        Initialize hybrid searcher with search function dependencies.

        Args:
            vector_search_fn: async fn(query_embedding, top_k) -> List[Chunk]
            keyword_search_fn: async fn(keywords, top_k) -> List[Chunk]
            direct_lookup_fn: async fn(law_number, article_number) -> List[Chunk]
        """
        self.vector_search = vector_search_fn
        self.keyword_search = keyword_search_fn
        self.direct_lookup = direct_lookup_fn

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 15,
    ) -> Tuple[List[Chunk], SearchType]:
        """
        Perform hybrid search combining multiple retrieval methods.

        Search priority:
        1. Direct law/article reference lookup (if detected in query)
        2. Hybrid: vector + keyword search with merged results

        Args:
            query: User's search query
            query_embedding: Pre-computed embedding for vector search (optional)
            top_k: Maximum results to return

        Returns:
            Tuple of (results, search_type)
        """
        # Canonicalize query for consistent matching
        normalized_query = canonicalize_for_search(query)

        # Step 1: Check for direct law/article reference
        law_ref = extract_law_reference(query)
        article_ref = extract_article_reference(query)

        if law_ref:
            # Direct lookup has highest priority
            direct_results = await self.direct_lookup(law_ref, article_ref)
            if direct_results:
                return direct_results[:top_k], SearchType.DIRECT_REFERENCE

        # Step 2: Parallel hybrid search (vector + keyword)
        keywords = extract_search_keywords(query)

        # Run searches in parallel
        vector_task = None
        keyword_task = None

        if query_embedding:
            vector_task = asyncio.create_task(
                self._safe_vector_search(query_embedding, top_k)
            )

        if keywords:
            keyword_task = asyncio.create_task(
                self._safe_keyword_search(keywords, top_k)
            )

        # Gather results
        vector_results = []
        keyword_results = []

        if vector_task:
            vector_results = await vector_task
        if keyword_task:
            keyword_results = await keyword_task

        # Step 3: Merge with deterministic reranking
        merged = self._merge_results(
            vector_results,
            keyword_results,
            top_k
        )

        # Determine search type for logging
        if vector_results and keyword_results:
            search_type = SearchType.HYBRID
        elif vector_results:
            search_type = SearchType.VECTOR
        elif keyword_results:
            search_type = SearchType.KEYWORD
        else:
            search_type = SearchType.HYBRID  # No results from either

        return merged, search_type

    async def _safe_vector_search(
        self,
        embedding: List[float],
        top_k: int
    ) -> List[Chunk]:
        """Vector search with error handling."""
        try:
            results = await self.vector_search(embedding, top_k)
            # Assign vector scores (normalized 0-1, descending)
            for i, chunk in enumerate(results):
                chunk.vector_score = 1.0 - (i / max(len(results), 1))
            return results
        except Exception:
            # Log error but don't fail the entire search
            return []

    async def _safe_keyword_search(
        self,
        keywords: List[str],
        top_k: int
    ) -> List[Chunk]:
        """Keyword search with error handling."""
        try:
            results = await self.keyword_search(keywords, top_k)
            # Assign keyword scores (normalized 0-1, descending)
            for i, chunk in enumerate(results):
                chunk.keyword_score = 1.0 - (i / max(len(results), 1))
            return results
        except Exception:
            # Log error but don't fail the entire search
            return []

    def _merge_results(
        self,
        vector_results: List[Chunk],
        keyword_results: List[Chunk],
        top_k: int,
    ) -> List[Chunk]:
        """
        Merge vector and keyword results with deterministic reranking.

        Merge strategy (favoring recall):
        - Union of results (not intersection)
        - Keyword matches get bonus for exact matches
        - Combined score = vector_score + keyword_score + bonuses

        This ensures we don't miss relevant results due to either
        method's weaknesses (embeddings missing inflections, keywords
        missing semantic similarity).
        """
        # Build lookup by chunk ID
        chunks_by_id = {}

        # Add vector results
        for chunk in vector_results:
            chunks_by_id[chunk.id] = chunk

        # Merge keyword results (add scores if already present)
        for chunk in keyword_results:
            if chunk.id in chunks_by_id:
                # Chunk found by both methods - add keyword score
                existing = chunks_by_id[chunk.id]
                existing.keyword_score = chunk.keyword_score
            else:
                # Chunk only found by keyword search
                chunks_by_id[chunk.id] = chunk

        # Calculate combined scores
        for chunk in chunks_by_id.values():
            # Base combined score
            chunk.combined_score = chunk.vector_score + chunk.keyword_score

            # Bonus for appearing in both results (reinforcement)
            if chunk.vector_score > 0 and chunk.keyword_score > 0:
                chunk.combined_score += 0.5  # Reinforcement bonus

        # Sort by combined score (descending)
        merged = sorted(
            chunks_by_id.values(),
            key=lambda c: c.combined_score,
            reverse=True
        )

        return merged[:top_k]


# Standalone functions for simple use cases

def merge_search_results(
    vector_results: List[Chunk],
    keyword_results: List[Chunk],
    top_k: int = 15,
) -> List[Chunk]:
    """
    Standalone merge function for testing or simple integration.

    Same logic as HybridSearcher._merge_results().
    """
    chunks_by_id = {}

    for i, chunk in enumerate(vector_results):
        chunk.vector_score = 1.0 - (i / max(len(vector_results), 1))
        chunks_by_id[chunk.id] = chunk

    for i, chunk in enumerate(keyword_results):
        chunk.keyword_score = 1.0 - (i / max(len(keyword_results), 1))
        if chunk.id in chunks_by_id:
            chunks_by_id[chunk.id].keyword_score = chunk.keyword_score
        else:
            chunks_by_id[chunk.id] = chunk

    for chunk in chunks_by_id.values():
        chunk.combined_score = chunk.vector_score + chunk.keyword_score
        if chunk.vector_score > 0 and chunk.keyword_score > 0:
            chunk.combined_score += 0.5

    merged = sorted(
        chunks_by_id.values(),
        key=lambda c: c.combined_score,
        reverse=True
    )

    return merged[:top_k]
