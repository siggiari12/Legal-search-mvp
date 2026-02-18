"""
OpenAI Embeddings Service

Generates vector embeddings for text using OpenAI's text-embedding-3-small model.
Supports batch processing with rate limiting.

Usage:
    from app.services.embedding import EmbeddingService

    service = EmbeddingService()

    # Single text
    embedding = await service.embed("Some text")

    # Batch
    embeddings = await service.embed_batch(["Text 1", "Text 2", ...])
"""

import asyncio
import os
from typing import List, Optional

from openai import AsyncOpenAI

from app.services.canonicalize import canonicalize


# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
MAX_BATCH_SIZE = 100  # OpenAI limit per request
MAX_TOKENS_PER_TEXT = 8191  # Model limit


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI API.

    Uses text-embedding-3-small which provides good quality at lower cost.
    Dimension: 1536
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize embedding service.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = EMBEDDING_MODEL
        self.dimension = EMBEDDING_DIMENSION

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed (will be canonicalized)

        Returns:
            List of floats (1536 dimensions)
        """
        # Canonicalize for consistency
        normalized = canonicalize(text)

        if not normalized:
            raise ValueError("Cannot embed empty text")

        # Truncate if too long (rough estimate: 1 token ~ 4 chars)
        max_chars = MAX_TOKENS_PER_TEXT * 4
        if len(normalized) > max_chars:
            normalized = normalized[:max_chars]

        response = await self.client.embeddings.create(
            model=self.model,
            input=normalized,
        )

        return response.data[0].embedding

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = MAX_BATCH_SIZE,
        delay_between_batches: float = 0.1,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Processes in batches to respect API limits.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (max 100)
            delay_between_batches: Seconds to wait between batches

        Returns:
            List of embeddings (same order as input texts)
        """
        if not texts:
            return []

        # Canonicalize and truncate all texts
        max_chars = MAX_TOKENS_PER_TEXT * 4
        normalized_texts = []
        for text in texts:
            normalized = canonicalize(text)
            if len(normalized) > max_chars:
                normalized = normalized[:max_chars]
            normalized_texts.append(normalized if normalized else " ")  # Empty -> space

        all_embeddings = []

        # Process in batches
        for i in range(0, len(normalized_texts), batch_size):
            batch = normalized_texts[i:i + batch_size]

            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Rate limiting delay between batches
            if i + batch_size < len(normalized_texts):
                await asyncio.sleep(delay_between_batches)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Same as embed() but named for clarity in search context.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        return await self.embed(query)


# Convenience function for simple usage
async def get_embedding(text: str, api_key: Optional[str] = None) -> List[float]:
    """
    Generate embedding for a single text.

    Creates a new service instance each call - for repeated use,
    create an EmbeddingService instance instead.

    Args:
        text: Text to embed
        api_key: Optional OpenAI API key

    Returns:
        Embedding vector (1536 dimensions)
    """
    service = EmbeddingService(api_key=api_key)
    return await service.embed(text)


async def get_embeddings_batch(
    texts: List[str],
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        api_key: Optional OpenAI API key

    Returns:
        List of embedding vectors
    """
    service = EmbeddingService(api_key=api_key)
    return await service.embed_batch(texts)
