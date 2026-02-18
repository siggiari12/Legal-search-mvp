"""
Embedding Service Tests

Tests for OpenAI embedding generation.
Automatically SKIPPED if OPENAI_API_KEY is not set (avoids API costs in CI).

Run with:
    pytest tests/test_embedding.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    OPENAI_API_KEY is None or OPENAI_API_KEY == "sk-...",
    reason="OPENAI_API_KEY not set — skipping embedding tests"
)


@pytest.fixture
def service():
    from app.services.embedding import EmbeddingService
    return EmbeddingService()


@pytest.fixture
def dimension():
    from app.services.embedding import EMBEDDING_DIMENSION
    return EMBEDDING_DIMENSION


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)


class TestBasicEmbedding:
    @pytest.mark.asyncio
    async def test_single_embedding(self, service, dimension):
        embedding = await service.embed("Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) == dimension
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embedding_dimension(self, service):
        embedding = await service.embed("Test text for dimension check")
        assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_icelandic_text(self, service, dimension):
        embedding = await service.embed("Þetta er íslenskur texti með sérstökum stöfum")
        assert len(embedding) == dimension

    @pytest.mark.asyncio
    async def test_legal_text(self, service, dimension):
        legal_text = "1. gr. laga nr. 33/1944 um þjóðfánann kveður á um að íslenskur fáni sé blár"
        embedding = await service.embed(legal_text)
        assert len(embedding) == dimension


class TestBatchEmbedding:
    @pytest.mark.asyncio
    async def test_batch_embedding(self, service, dimension):
        texts = ["First text", "Second text", "Third text"]
        embeddings = await service.embed_batch(texts)
        assert len(embeddings) == 3
        assert all(len(e) == dimension for e in embeddings)

    @pytest.mark.asyncio
    async def test_empty_batch(self, service):
        embeddings = await service.embed_batch([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self, service):
        texts = ["Alpha", "Beta", "Gamma"]
        embeddings = await service.embed_batch(texts)
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]


class TestSimilarity:
    @pytest.mark.asyncio
    async def test_similar_texts_have_similar_embeddings(self, service):
        emb1 = await service.embed("The cat sat on the mat")
        emb2 = await service.embed("A cat was sitting on a mat")
        emb3 = await service.embed("Database query optimization techniques")

        sim_12 = cosine_similarity(emb1, emb2)
        sim_13 = cosine_similarity(emb1, emb3)

        assert sim_12 > sim_13
        assert sim_12 > 0.8


class TestCanonicalization:
    @pytest.mark.asyncio
    async def test_whitespace_normalized(self, service):
        emb1 = await service.embed("hello   world")
        emb2 = await service.embed("hello world")
        sim = cosine_similarity(emb1, emb2)
        assert sim > 0.99


class TestQueryEmbedding:
    @pytest.mark.asyncio
    async def test_query_embedding(self, service, dimension):
        query_emb = await service.embed_query("What does the law say about human rights?")
        assert len(query_emb) == dimension
