"""
Database Smoke Tests

These tests verify basic database connectivity and operations.
Automatically SKIPPED if DATABASE_URL is not set.

Run with:
    pytest tests/test_db_smoke.py -v

Requires:
    - DATABASE_URL environment variable set
    - Migrations applied (python scripts/run_migrations.py)
"""

import asyncio
import os
import sys

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="DATABASE_URL not set — skipping database tests"
)


@pytest_asyncio.fixture
async def db():
    asyncpg = pytest.importorskip("asyncpg", reason="asyncpg not installed")
    from app.db.connection import Database, close_db_pool

    database = Database()
    await database.connect()
    yield database
    await close_db_pool()


class TestConnection:
    @pytest.mark.asyncio
    async def test_can_connect(self, db):
        result = await db.fetchval("SELECT 1")
        assert result == 1

    @pytest.mark.asyncio
    async def test_health_check(self, db):
        healthy = await db.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_tables_exist(self, db):
        for table in ["documents", "chunks", "query_logs"]:
            exists = await db.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = '{table}'
                )
            """)
            assert exists is True, f"{table} table should exist"

    @pytest.mark.asyncio
    async def test_pgvector_enabled(self, db):
        vector_exists = await db.fetchval("""
            SELECT EXISTS (
                SELECT FROM pg_extension
                WHERE extname = 'vector'
            )
        """)
        assert vector_exists is True


class TestDocumentOperations:
    @pytest.mark.asyncio
    async def test_insert_and_retrieve_document(self, db):
        doc_id = await db.insert_document(
            title="Test Law",
            law_number="999",
            law_year="2024",
            full_text="This is test content.",
            version_tag="test-run",
            full_text_normalized="this is test content.",
        )
        try:
            doc = await db.get_document_by_law_reference("999/2024")
            assert doc_id is not None
            assert doc is not None
            assert doc["title"] == "Test Law"
            assert doc["law_reference"] == "999/2024"
        finally:
            await db.execute("DELETE FROM documents WHERE id = $1", doc_id)


class TestChunkOperations:
    @pytest.mark.asyncio
    async def test_insert_and_retrieve_chunk(self, db):
        doc_id = await db.insert_document(
            title="Test Law for Chunks",
            law_number="998",
            law_year="2024",
            full_text="Full text here.",
            version_tag="test-run",
        )
        try:
            chunk_id = await db.insert_chunk(
                document_id=doc_id,
                chunk_text="This is article 1.",
                locator="Lög nr. 998/2024 - 1. gr.",
                chunk_index=0,
                chunk_text_normalized="this is article 1.",
                article_number="1",
                law_number="998",
                law_year="2024",
            )
            chunks = await db.get_chunks_by_document(doc_id)
            assert chunk_id is not None
            assert len(chunks) == 1
            assert chunks[0]["chunk_text"] == "This is article 1."
            assert chunks[0]["article_number"] == "1"
        finally:
            await db.execute("DELETE FROM documents WHERE id = $1", doc_id)


class TestSearchOperations:
    @pytest.mark.asyncio
    async def test_direct_lookup(self, db):
        doc_id = await db.insert_document(
            title="Test Law for Search",
            law_number="997",
            law_year="2024",
            full_text="Full text.",
            version_tag="test-run",
        )
        try:
            await db.insert_chunk(
                document_id=doc_id,
                chunk_text="Article content.",
                locator="Lög nr. 997/2024 - 1. gr.",
                chunk_index=0,
                article_number="1",
                law_number="997",
                law_year="2024",
            )
            results = await db.direct_lookup("997/2024")
            results_with_article = await db.direct_lookup("997/2024", "1")
            assert len(results) == 1
            assert len(results_with_article) == 1
            assert results[0]["chunk_text"] == "Article content."
        finally:
            await db.execute("DELETE FROM documents WHERE id = $1", doc_id)

    @pytest.mark.asyncio
    async def test_keyword_search(self, db):
        doc_id = await db.insert_document(
            title="Test Law for Keywords",
            law_number="996",
            law_year="2024",
            full_text="Full text.",
            version_tag="test-run",
        )
        try:
            await db.insert_chunk(
                document_id=doc_id,
                chunk_text="Mannréttindi eru mikilvæg.",
                locator="Lög nr. 996/2024 - 1. gr.",
                chunk_index=0,
                chunk_text_normalized="mannréttindi eru mikilvæg.",
                article_number="1",
                law_number="996",
                law_year="2024",
            )
            results = await db.keyword_search(["mannréttindi"])
            assert len(results) >= 1
            assert any("mannréttindi" in r["chunk_text_normalized"] for r in results)
        finally:
            await db.execute("DELETE FROM documents WHERE id = $1", doc_id)


class TestQueryLogging:
    @pytest.mark.asyncio
    async def test_log_query(self, db):
        log_id = await db.log_query(
            query_hash="abc123",
            query_length=50,
            chunk_count=5,
            search_type="hybrid",
            validation_passed=True,
            confidence="high",
            citation_count=2,
        )
        try:
            assert log_id is not None
        finally:
            await db.execute("DELETE FROM query_logs WHERE id = $1", log_id)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, db):
        under_limit = await db.check_rate_limit("test-ip-hash-unique-12345", limit=20)
        assert under_limit is True


class TestStatistics:
    @pytest.mark.asyncio
    async def test_get_stats(self, db):
        stats = await db.get_stats()
        assert "documents" in stats
        assert "chunks" in stats
        assert "embedded_chunks" in stats
        assert isinstance(stats["documents"], int)
