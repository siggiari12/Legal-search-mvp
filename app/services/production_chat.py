"""
Production Chat Service

Wires together all services for the production Legal Search chat:
- Database-backed hybrid search
- OpenAI embeddings
- Claude LLM responses
- Strict citation validation

Usage:
    from app.services.production_chat import ProductionChatService

    service = ProductionChatService()
    await service.connect()

    response = await service.chat(ChatRequest(query="..."))

    await service.close()
"""

from typing import Optional

from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat import ChatService
from app.services.db_search import DatabaseSearcher
from app.services.embedding import EmbeddingService
from app.services.llm import ClaudeLLM


class ProductionChatService:
    """
    Production-ready chat service with all integrations.

    Connects:
    - Supabase/PostgreSQL for document storage
    - pgvector for semantic search
    - OpenAI for query embeddings
    - Claude for response generation
    - Strict validation for citations
    """

    def __init__(
        self,
        debug: bool = False,
        skip_embeddings: bool = False,
    ):
        """
        Initialize production chat service.

        Args:
            debug: Include debug info in responses
            skip_embeddings: Skip embedding generation (keyword search only)
        """
        self.debug = debug
        self.skip_embeddings = skip_embeddings

        self._searcher: Optional[DatabaseSearcher] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._llm: Optional[ClaudeLLM] = None
        self._chat_service: Optional[ChatService] = None
        self._connected = False

    async def connect(self) -> "ProductionChatService":
        """
        Connect to all services.

        Returns self for chaining.
        """
        if self._connected:
            return self

        # Initialize database searcher
        self._searcher = DatabaseSearcher()
        await self._searcher.connect()

        # Initialize embedding service (optional)
        if not self.skip_embeddings:
            try:
                self._embedding_service = EmbeddingService()
            except ValueError:
                # No OpenAI key - fall back to keyword-only search
                self._embedding_service = None

        # Initialize LLM
        self._llm = ClaudeLLM()

        # Create chat service with dependencies
        self._chat_service = ChatService(
            searcher=self._searcher._hybrid_searcher,
            llm_fn=self._llm,
            embedding_fn=self._embedding_service.embed if self._embedding_service else None,
            debug=self.debug,
        )

        self._connected = True
        return self

    async def close(self):
        """Close all connections."""
        if self._searcher:
            await self._searcher.close()
        self._connected = False

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request.

        Args:
            request: ChatRequest with query

        Returns:
            ChatResponse with answer, citations, or failure
        """
        if not self._connected:
            await self.connect()

        return await self._chat_service.chat(request)

    async def health_check(self) -> dict:
        """
        Check health of all services.

        Returns:
            Dict with service statuses
        """
        status = {
            "database": False,
            "embeddings": False,
            "llm": False,
        }

        # Check database
        if self._searcher:
            try:
                await self._searcher.db.health_check()
                status["database"] = True
            except Exception:
                pass

        # Check embeddings
        if self._embedding_service:
            status["embeddings"] = True

        # Check LLM
        if self._llm:
            status["llm"] = True

        return status


# Convenience function for single queries
async def chat(
    query: str,
    debug: bool = False,
) -> ChatResponse:
    """
    Process a single chat query.

    Creates new connections each call - for repeated use,
    create a ProductionChatService instance.

    Args:
        query: User's question
        debug: Include debug info

    Returns:
        ChatResponse
    """
    service = ProductionChatService(debug=debug)
    try:
        await service.connect()
        return await service.chat(ChatRequest(query=query))
    finally:
        await service.close()
