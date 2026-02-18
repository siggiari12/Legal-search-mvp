"""
LLM Service Tests

Tests for Claude integration and production chat service.
Automatically SKIPPED if ANTHROPIC_API_KEY is not set.

Run with:
    pytest tests/test_llm.py -v
"""

import os
import sys

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

skip_llm = pytest.mark.skipif(
    ANTHROPIC_API_KEY is None or ANTHROPIC_API_KEY == "sk-ant-...",
    reason="ANTHROPIC_API_KEY not set — skipping LLM tests"
)

skip_integration = pytest.mark.skipif(
    ANTHROPIC_API_KEY is None or ANTHROPIC_API_KEY == "sk-ant-..." or DATABASE_URL is None,
    reason="ANTHROPIC_API_KEY or DATABASE_URL not set — skipping integration tests"
)


class TestClaudeLLM:
    pytestmark = skip_llm

    @pytest.fixture
    def llm(self):
        from app.services.llm import ClaudeLLM
        return ClaudeLLM()

    @pytest.mark.asyncio
    async def test_simple_generation(self, llm):
        response = await llm.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2? Reply with just the number.",
        )
        assert "4" in response

    @pytest.mark.asyncio
    async def test_icelandic_response(self, llm):
        response = await llm.generate(
            system_prompt="Þú ert aðstoðarmaður sem svarar á íslensku.",
            user_prompt="Hvað er tveir plús tveir? Svaraðu aðeins með tölunni.",
        )
        assert "4" in response or "fjór" in response.lower()

    @pytest.mark.asyncio
    async def test_json_response(self, llm):
        response = await llm.generate(
            system_prompt="You respond only in JSON format.",
            user_prompt='Return a JSON object with key "result" and value 42.',
        )
        assert "42" in response and "result" in response


class TestProductionChat:
    pytestmark = skip_integration

    @pytest_asyncio.fixture
    async def service(self):
        from app.services.production_chat import ProductionChatService
        svc = ProductionChatService(debug=True, skip_embeddings=True)
        await svc.connect()
        yield svc
        await svc.close()

    @pytest.mark.asyncio
    async def test_health_check(self, service):
        health = await service.health_check()
        assert health["database"] is True
        assert health["llm"] is True

    @pytest.mark.asyncio
    async def test_direct_reference_query(self, service):
        from app.models.schemas import ChatRequest
        response = await service.chat(ChatRequest(
            query="Hvað segir 1. gr. laga nr. 33/1944?"
        ))
        assert response.failure_reason is None or response.failure_reason.value != "internal_error"

    @pytest.mark.asyncio
    async def test_keyword_query(self, service):
        from app.models.schemas import ChatRequest
        response = await service.chat(ChatRequest(
            query="Hvað segja lögin um þjóðfána Íslendinga?"
        ))
        assert response.failure_reason is None or response.failure_reason.value != "internal_error"
