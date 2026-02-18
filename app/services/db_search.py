"""
Database-Backed Search Service

Connects the hybrid search logic to the real Supabase database.
This module provides the production search implementation.

Usage:
    from app.services.db_search import DatabaseSearcher

    searcher = DatabaseSearcher()
    await searcher.connect()

    results, search_type = await searcher.search(
        query="Hvað segir í stjórnarskránni?",
        query_embedding=[0.1, 0.2, ...]  # From embedding service
    )
"""

from typing import List, Optional, Tuple

from app.db.connection import Database
from app.services.search import (
    Chunk,
    SearchType,
    HybridSearcher,
    extract_law_reference,
    extract_article_reference,
    extract_search_keywords,
)
from app.services.embedding import EmbeddingService


class DatabaseSearcher:
    """
    Production search service backed by Supabase/PostgreSQL.

    Combines:
    - Vector search (pgvector)
    - Keyword search (PostgreSQL ILIKE)
    - Direct law/article lookup
    """

    def __init__(self, db: Optional[Database] = None):
        """
        Initialize database searcher.

        Args:
            db: Database instance. If not provided, creates new one.
        """
        self.db = db or Database()
        self._connected = False
        self._hybrid_searcher: Optional[HybridSearcher] = None
        self._embedding_service: Optional[EmbeddingService] = None

    async def connect(self) -> "DatabaseSearcher":
        """Connect to database."""
        if not self._connected:
            await self.db.connect()
            self._connected = True

            # Create hybrid searcher with database functions
            self._hybrid_searcher = HybridSearcher(
                vector_search_fn=self._vector_search,
                keyword_search_fn=self._keyword_search,
                direct_lookup_fn=self._direct_lookup,
            )

        return self

    async def close(self):
        """Close database connection."""
        if self._connected:
            from app.db.connection import close_db_pool
            await close_db_pool()
            self._connected = False

    def _record_to_chunk(self, record) -> Chunk:
        """Convert database record to Chunk object."""
        return Chunk(
            id=str(record["id"]),
            document_id=str(record["document_id"]),
            chunk_text=record["chunk_text"],
            locator=record["locator"],
            chunk_text_normalized=record.get("chunk_text_normalized"),
            article_number=record.get("article_number"),
            paragraph_number=record.get("paragraph_number"),
            law_number=record.get("law_number"),
            law_year=record.get("law_year"),
        )

    async def _vector_search(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[Chunk]:
        """Perform vector similarity search."""
        records = await self.db.vector_search(embedding, top_k)
        return [self._record_to_chunk(r) for r in records]

    async def _keyword_search(
        self,
        keywords: List[str],
        top_k: int,
    ) -> List[Chunk]:
        """Perform keyword search."""
        records = await self.db.keyword_search(keywords, top_k)
        return [self._record_to_chunk(r) for r in records]

    async def _direct_lookup(
        self,
        law_reference: str,
        article_number: Optional[str],
    ) -> List[Chunk]:
        """Direct lookup by law reference."""
        records = await self.db.direct_lookup(law_reference, article_number)
        return [self._record_to_chunk(r) for r in records]

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 15,
    ) -> Tuple[List[Chunk], SearchType]:
        """
        Perform hybrid search.

        Args:
            query: User's search query
            query_embedding: Pre-computed embedding (optional)
            top_k: Maximum results to return

        Returns:
            Tuple of (results, search_type)
        """
        if not self._connected:
            await self.connect()

        return await self._hybrid_searcher.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    async def search_with_embedding(
        self,
        query: str,
        top_k: int = 15,
    ) -> Tuple[List[Chunk], SearchType]:
        """
        Perform hybrid search, generating embedding automatically.

        Requires OPENAI_API_KEY to be set.

        Args:
            query: User's search query
            top_k: Maximum results to return

        Returns:
            Tuple of (results, search_type)
        """
        if not self._connected:
            await self.connect()

        # Initialize embedding service if needed
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()

        # Generate query embedding
        query_embedding = await self._embedding_service.embed_query(query)

        return await self.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
        )


# Convenience function
async def search(
    query: str,
    query_embedding: Optional[List[float]] = None,
    top_k: int = 15,
) -> Tuple[List[Chunk], SearchType]:
    """
    Perform a search query against the database.

    Args:
        query: Search query
        query_embedding: Pre-computed embedding (optional)
        top_k: Maximum results

    Returns:
        Tuple of (chunks, search_type)
    """
    searcher = DatabaseSearcher()
    await searcher.connect()
    try:
        return await searcher.search(query, query_embedding, top_k)
    finally:
        await searcher.close()
