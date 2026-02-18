"""
Database Connection Module

Provides async PostgreSQL connection pool using asyncpg.
Designed for Supabase but works with any PostgreSQL database.

Usage:
    from app.db.connection import get_db_pool, Database

    # Get the singleton pool
    pool = await get_db_pool()

    # Or use the Database class for operations
    db = Database()
    await db.connect()
    results = await db.fetch("SELECT * FROM documents")
    await db.close()
"""

import os
import json
import asyncio
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager

import asyncpg


# Singleton pool instance
_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


def get_database_url() -> Optional[str]:
    """
    Get database URL from environment.
    Returns None if not configured (for testing without DB).
    """
    return os.environ.get("DATABASE_URL")


async def get_db_pool(
    database_url: Optional[str] = None,
    min_size: int = 2,
    max_size: int = 10,
) -> asyncpg.Pool:
    """
    Get or create the database connection pool.

    Uses singleton pattern - only one pool is created per process.

    Args:
        database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
        min_size: Minimum pool connections
        max_size: Maximum pool connections

    Returns:
        asyncpg.Pool instance

    Raises:
        ValueError: If no database URL is configured
    """
    global _pool

    async with _pool_lock:
        if _pool is not None:
            return _pool

        url = database_url or get_database_url()
        if not url:
            raise ValueError(
                "DATABASE_URL environment variable is not set. "
                "Set it to your PostgreSQL connection string."
            )

        _pool = await asyncpg.create_pool(
            url,
            min_size=min_size,
            max_size=max_size,
            # Disable prepared statement cache for pgbouncer/Supabase pooler compatibility
            statement_cache_size=0,
            # Supabase/pgvector compatibility
            server_settings={
                "search_path": "public",
            },
        )

        return _pool


async def close_db_pool():
    """Close the database connection pool."""
    global _pool

    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None


class Database:
    """
    Database wrapper providing convenient async operations.

    Can use either a shared pool or create its own connection.
    Provides methods for common operations used in the Legal Search MVP.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        """
        Initialize database wrapper.

        Args:
            pool: Existing pool to use. If None, will get/create shared pool.
        """
        self._pool = pool
        self._owns_pool = False

    async def connect(self) -> "Database":
        """
        Connect to the database (get or create pool).

        Returns self for chaining.
        """
        if self._pool is None:
            self._pool = await get_db_pool()
        return self

    async def close(self):
        """Close connection if we own it."""
        if self._owns_pool and self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if self._pool is None:
            await self.connect()

        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args) -> str:
        """
        Execute a query that doesn't return rows.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Status string (e.g., "INSERT 0 1")
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """
        Fetch multiple rows.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of Record objects (dict-like)
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        Fetch a single row.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Record object or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """
        Fetch a single value.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single value from first column of first row
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    # ========================================================
    # Document Operations
    # ========================================================

    async def insert_document(
        self,
        title: str,
        law_number: str,
        law_year: str,
        full_text: str,
        version_tag: str,
        full_text_normalized: Optional[str] = None,
        publication_date: Optional[str] = None,
        canonical_url: Optional[str] = None,
        metadata_json: Optional[Dict] = None,
    ) -> str:
        """
        Insert a new document.

        Returns:
            UUID of inserted document
        """
        query = """
            INSERT INTO documents (
                title, law_number, law_year, law_reference,
                full_text, full_text_normalized, version_tag,
                publication_date, canonical_url, metadata_json
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """

        law_reference = f"{law_number}/{law_year}"

        async with self.acquire() as conn:
            result = await conn.fetchval(
                query,
                title,
                law_number,
                law_year,
                law_reference,
                full_text,
                full_text_normalized,
                version_tag,
                publication_date,
                canonical_url,
                json.dumps(metadata_json or {}),
            )
            return str(result)

    async def get_document_by_law_reference(
        self, law_reference: str
    ) -> Optional[asyncpg.Record]:
        """
        Get document by law reference (e.g., "33/1944").
        """
        query = """
            SELECT * FROM documents
            WHERE law_reference = $1
            ORDER BY created_at DESC
            LIMIT 1
        """
        return await self.fetchrow(query, law_reference)

    # ========================================================
    # Chunk Operations
    # ========================================================

    async def insert_chunk(
        self,
        document_id: str,
        chunk_text: str,
        locator: str,
        chunk_index: int,
        chunk_text_normalized: Optional[str] = None,
        article_number: Optional[str] = None,
        paragraph_number: Optional[str] = None,
        law_number: Optional[str] = None,
        law_year: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Insert a new chunk.

        Returns:
            UUID of inserted chunk
        """
        law_reference = f"{law_number}/{law_year}" if law_number and law_year else None

        # Build query dynamically based on whether embedding is provided
        if embedding is not None:
            query = """
                INSERT INTO chunks (
                    document_id, chunk_text, chunk_text_normalized, locator,
                    chunk_index, article_number, paragraph_number,
                    law_number, law_year, law_reference, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
            """
            args = [
                document_id,
                chunk_text,
                chunk_text_normalized,
                locator,
                chunk_index,
                article_number,
                paragraph_number,
                law_number,
                law_year,
                law_reference,
                embedding,
            ]
        else:
            query = """
                INSERT INTO chunks (
                    document_id, chunk_text, chunk_text_normalized, locator,
                    chunk_index, article_number, paragraph_number,
                    law_number, law_year, law_reference
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """
            args = [
                document_id,
                chunk_text,
                chunk_text_normalized,
                locator,
                chunk_index,
                article_number,
                paragraph_number,
                law_number,
                law_year,
                law_reference,
            ]

        async with self.acquire() as conn:
            result = await conn.fetchval(query, *args)
            return str(result)

    async def update_chunk_embedding(
        self, chunk_id: str, embedding: List[float]
    ) -> None:
        """Update embedding for an existing chunk."""
        query = "UPDATE chunks SET embedding = $1 WHERE id = $2"
        await self.execute(query, embedding, chunk_id)

    async def get_chunks_by_document(
        self, document_id: str
    ) -> List[asyncpg.Record]:
        """Get all chunks for a document."""
        query = """
            SELECT * FROM chunks
            WHERE document_id = $1
            ORDER BY chunk_index
        """
        return await self.fetch(query, document_id)

    # ========================================================
    # Search Operations
    # ========================================================

    async def direct_lookup(
        self,
        law_reference: str,
        article_number: Optional[str] = None,
    ) -> List[asyncpg.Record]:
        """
        Direct lookup by law reference and optional article.

        This is the highest-priority search method for exact references.
        """
        if article_number:
            query = """
                SELECT * FROM chunks
                WHERE law_reference = $1 AND article_number = $2
                ORDER BY chunk_index
            """
            return await self.fetch(query, law_reference, article_number)
        else:
            query = """
                SELECT * FROM chunks
                WHERE law_reference = $1
                ORDER BY chunk_index
            """
            return await self.fetch(query, law_reference)

    async def vector_search(
        self,
        embedding: List[float],
        top_k: int = 15,
    ) -> List[asyncpg.Record]:
        """
        Semantic similarity search using pgvector.

        Uses cosine distance for similarity ranking.
        """
        query = """
            SELECT *,
                   1 - (embedding <=> $1::vector) as similarity
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        return await self.fetch(query, embedding, top_k)

    async def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 15,
    ) -> List[asyncpg.Record]:
        """
        Full-text keyword search using PostgreSQL.

        Combines ILIKE matching with ts_rank for relevance.
        """
        if not keywords:
            return []

        # Build OR conditions for ILIKE matching
        conditions = []
        args = []
        for i, keyword in enumerate(keywords):
            conditions.append(f"chunk_text_normalized ILIKE ${i + 1}")
            args.append(f"%{keyword}%")

        # Add top_k as last argument
        args.append(top_k)

        query = f"""
            SELECT *,
                   (SELECT COUNT(*) FROM unnest(ARRAY[{','.join([f'chunk_text_normalized ILIKE ${i+1}' for i in range(len(keywords))])}]::boolean[]) AS c WHERE c) as match_count
            FROM chunks
            WHERE {' OR '.join(conditions)}
            ORDER BY match_count DESC, chunk_index
            LIMIT ${len(keywords) + 1}
        """

        return await self.fetch(query, *args)

    # ========================================================
    # Query Logging (Privacy-First)
    # ========================================================

    async def log_query(
        self,
        query_hash: str,
        query_length: int,
        chunk_count: Optional[int] = None,
        search_type: Optional[str] = None,
        validation_passed: Optional[bool] = None,
        retry_count: int = 0,
        failure_reason: Optional[str] = None,
        confidence: Optional[str] = None,
        citation_count: Optional[int] = None,
        ip_hash: Optional[str] = None,
        search_duration_ms: Optional[int] = None,
        llm_duration_ms: Optional[int] = None,
        total_duration_ms: Optional[int] = None,
    ) -> str:
        """Log a query for analytics (privacy-preserving)."""
        query = """
            INSERT INTO query_logs (
                query_hash, query_length, chunk_count, search_type,
                validation_passed, retry_count, failure_reason,
                confidence, citation_count, ip_hash,
                search_duration_ms, llm_duration_ms, total_duration_ms
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
        """
        result = await self.fetchval(
            query,
            query_hash,
            query_length,
            chunk_count,
            search_type,
            validation_passed,
            retry_count,
            failure_reason,
            confidence,
            citation_count,
            ip_hash,
            search_duration_ms,
            llm_duration_ms,
            total_duration_ms,
        )
        return str(result)

    async def check_rate_limit(
        self,
        ip_hash: str,
        limit: int = 20,
        window_hours: int = 1,
    ) -> bool:
        """
        Check if IP has exceeded rate limit.

        Returns True if under limit, False if exceeded.
        """
        query = """
            SELECT COUNT(*) FROM query_logs
            WHERE ip_hash = $1
            AND created_at > NOW() - INTERVAL '1 hour' * $2
        """
        count = await self.fetchval(query, ip_hash, window_hours)
        return count < limit

    # ========================================================
    # Utility Methods
    # ========================================================

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception:
            return False

    async def get_stats(self) -> Dict[str, int]:
        """Get basic database statistics."""
        doc_count = await self.fetchval("SELECT COUNT(*) FROM documents")
        chunk_count = await self.fetchval("SELECT COUNT(*) FROM chunks")
        embedded_count = await self.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        )

        return {
            "documents": doc_count or 0,
            "chunks": chunk_count or 0,
            "embedded_chunks": embedded_count or 0,
        }
