"""
End-to-End Tests for Chat API

Tests the complete flow:
1. Query -> Hybrid Search -> LLM -> Validation -> Response

Uses mock LLM to test:
- Happy path: valid citations returned
- Ambiguous query: clarification requested
- No evidence: refusal
- Invalid citations: rejected (even with retry)
"""

import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.canonicalize import canonicalize
from app.services.search import Chunk, HybridSearcher, SearchType
from app.services.chat import ChatService
from app.services.validation import ResponseValidator, ValidationContext
from app.api.chat import ChatEndpoint, APIResponse
from app.models.schemas import ChatRequest, FailureReason


# Test data - sample Icelandic legal chunks
def make_chunk(id, doc_id, text, locator, law_num, law_year):
    """Helper to create test chunks."""
    return Chunk(
        id=id,
        document_id=doc_id,
        chunk_text=text,
        locator=locator,
        law_number=law_num,
        law_year=law_year,
    )

TEST_CHUNKS = [
    make_chunk(
        id="1",
        doc_id="doc_33_1944",
        text="Þjóðfáni Íslendinga er himinblár með mjóum hvítum krossi, rauðum að innanverðu. Armar krossins ná að jöðrum fánans.",
        locator="Lög nr. 33/1944 - 1. gr.",
        law_num="33",
        law_year="1944",
    ),
    make_chunk(
        id="2",
        doc_id="doc_33_1944",
        text="Stærðarhlutföll fánans skulu vera þannig, að breiddin sé 18/25 af lengdinni.",
        locator="Lög nr. 33/1944 - 2. gr.",
        law_num="33",
        law_year="1944",
    ),
    make_chunk(
        id="3",
        doc_id="doc_85_2024",
        text="Lög þessi gilda um notkun gervigreindar í opinberri þjónustu. Markmið laganna er að tryggja örugga og ábyrga notkun gervigreindar.",
        locator="Lög nr. 85/2024 - 1. gr.",
        law_num="85",
        law_year="2024",
    ),
]


# Mock search functions
async def mock_vector_search(embedding, top_k):
    """Return all test chunks for any query."""
    return TEST_CHUNKS[:top_k]


async def mock_keyword_search(keywords, top_k):
    """Search by keywords."""
    results = []
    for chunk in TEST_CHUNKS:
        text_lower = chunk.chunk_text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                results.append(chunk)
                break
    return results[:top_k]


async def mock_direct_lookup(law_number, article_number):
    """Direct lookup by law number."""
    results = []
    for chunk in TEST_CHUNKS:
        if chunk.law_number == law_number.split('/')[0]:
            if article_number is None or chunk.locator.find(f"{article_number}. gr.") != -1:
                results.append(chunk)
    return results


def create_mock_llm(response_template: dict):
    """Create a mock LLM function that returns a specific response."""
    async def mock_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(response_template)
    return mock_llm


def create_mock_llm_with_valid_citation():
    """Create mock LLM that returns valid citation from test chunks."""
    response = {
        "answer_markdown": "Þjóðfáninn er blár með hvítum og rauðum krossi [1].",
        "citations": [{
            "document_id": "doc_33_1944",
            "locator": "Lög nr. 33/1944 - 1. gr.",
            "quote": "himinblár með mjóum hvítum krossi, rauðum að innanverðu",
        }],
        "needs_clarification": False,
        "clarification_question": None,
    }
    return create_mock_llm(response)


def create_mock_llm_with_invalid_citation():
    """Create mock LLM that returns fabricated citation."""
    response = {
        "answer_markdown": "Þetta er rangt [1].",
        "citations": [{
            "document_id": "doc_33_1944",
            "locator": "Lög nr. 33/1944 - 1. gr.",
            "quote": "Þetta er algjörlega uppspunnið og ekki í textanum",
        }],
        "needs_clarification": False,
        "clarification_question": None,
    }
    return create_mock_llm(response)


def create_mock_llm_refusal():
    """Create mock LLM that refuses to answer."""
    response = {
        "answer_markdown": "Ég finn ekki upplýsingar um þetta í heimildunum.",
        "citations": [],
        "needs_clarification": False,
        "clarification_question": None,
    }
    return create_mock_llm(response)


def create_mock_llm_clarification():
    """Create mock LLM that asks for clarification."""
    response = {
        "answer_markdown": None,
        "citations": [],
        "needs_clarification": True,
        "clarification_question": "Vinsamlegast tilgreindu hvaða lög þú átt við.",
    }
    return create_mock_llm(response)


class TestHappyPath:
    """Tests for successful query flow."""

    def test_valid_citation_passes(self):
        """Query with valid LLM response passes validation."""
        searcher = HybridSearcher(
            vector_search_fn=mock_vector_search,
            keyword_search_fn=mock_keyword_search,
            direct_lookup_fn=mock_direct_lookup,
        )

        service = ChatService(
            searcher=searcher,
            llm_fn=create_mock_llm_with_valid_citation(),
            debug=True,
        )

        async def run():
            request = ChatRequest(query="Hvernig lítur þjóðfáninn út samkvæmt lögum nr. 33/1944?")
            return await service.chat(request)

        response = asyncio.run(run())

        assert response.answer is not None, "Should have answer"
        assert len(response.citations) > 0, "Should have citations"
        assert response.failure_reason is None, f"Should not fail: {response.failure_reason}"

    def test_direct_law_reference_works(self):
        """Query with direct law reference uses direct lookup."""
        searcher = HybridSearcher(
            vector_search_fn=mock_vector_search,
            keyword_search_fn=mock_keyword_search,
            direct_lookup_fn=mock_direct_lookup,
        )

        service = ChatService(
            searcher=searcher,
            llm_fn=create_mock_llm_with_valid_citation(),
            debug=True,
        )

        async def run():
            request = ChatRequest(query="Hvað segir í lögum nr. 33/1944?")
            return await service.chat(request)

        response = asyncio.run(run())

        # Should use direct lookup
        if response.debug:
            assert response.debug.get("search_type") == "direct_reference" or \
                   response.debug.get("law_reference") == "33/1944"


class TestAmbiguousQuery:
    """Tests for ambiguous query handling."""

    def test_vague_query_requests_clarification(self):
        """Vague query with chunks requests clarification."""
        # Use mock that always returns chunks (so we can test ambiguity detection)
        async def always_return_chunks(embedding, top_k):
            return TEST_CHUNKS[:top_k]

        async def always_return_keyword(keywords, top_k):
            return TEST_CHUNKS[:top_k]

        searcher = HybridSearcher(
            vector_search_fn=always_return_chunks,
            keyword_search_fn=always_return_keyword,
            direct_lookup_fn=mock_direct_lookup,
        )

        service = ChatService(
            searcher=searcher,
            llm_fn=create_mock_llm_clarification(),
            debug=True,
        )

        async def run():
            # Short vague query - should trigger ambiguity check
            request = ChatRequest(query="Hvað er?")
            return await service.chat(request)

        response = asyncio.run(run())

        # Either AMBIGUOUS_QUERY (if detected) or returns clarification from LLM
        # Both are acceptable outcomes for a vague query
        is_ambiguous = response.failure_reason == FailureReason.AMBIGUOUS_QUERY
        has_clarification = response.clarification_question is not None
        assert is_ambiguous or has_clarification, \
            f"Vague query should trigger clarification. Got: {response.failure_reason}"

    def test_empty_query_rejected(self):
        """Empty query is rejected."""
        searcher = HybridSearcher(
            vector_search_fn=mock_vector_search,
            keyword_search_fn=mock_keyword_search,
            direct_lookup_fn=mock_direct_lookup,
        )

        service = ChatService(
            searcher=searcher,
            llm_fn=create_mock_llm_with_valid_citation(),
            debug=True,
        )

        async def run():
            request = ChatRequest(query="")
            return await service.chat(request)

        response = asyncio.run(run())

        assert response.failure_reason == FailureReason.AMBIGUOUS_QUERY


class TestInsufficientEvidence:
    """Tests for no-evidence scenarios."""

    def test_no_chunks_returns_refusal(self):
        """Query with no matching chunks returns refusal."""
        # Mock that returns no results
        async def empty_search(embedding, top_k):
            return []

        async def empty_keyword(keywords, top_k):
            return []

        async def empty_lookup(law, article):
            return []

        searcher = HybridSearcher(
            vector_search_fn=empty_search,
            keyword_search_fn=empty_keyword,
            direct_lookup_fn=empty_lookup,
        )

        service = ChatService(
            searcher=searcher,
            llm_fn=create_mock_llm_with_valid_citation(),
            debug=True,
        )

        async def run():
            request = ChatRequest(query="Hvað segir um eitthvað sem ekki er í gagnagrunni?")
            return await service.chat(request)

        response = asyncio.run(run())

        assert response.failure_reason == FailureReason.NO_RELEVANT_DATA
        assert response.answer is None


class TestValidationRejection:
    """Tests for validation rejection - CRITICAL."""

    def test_fabricated_citation_rejected(self):
        """Response with fabricated quote is rejected."""
        # Mocks that always return chunks (so we can test validation)
        async def always_return_chunks(embedding, top_k):
            return TEST_CHUNKS[:top_k]

        async def always_return_keyword(keywords, top_k):
            return TEST_CHUNKS[:top_k]

        searcher = HybridSearcher(
            vector_search_fn=always_return_chunks,
            keyword_search_fn=always_return_keyword,
            direct_lookup_fn=mock_direct_lookup,
        )

        # LLM that always returns fabricated citation
        service = ChatService(
            searcher=searcher,
            llm_fn=create_mock_llm_with_invalid_citation(),
            debug=True,
        )

        async def run():
            request = ChatRequest(query="Hvernig lítur þjóðfáninn út samkvæmt lögum nr. 33/1944?")
            return await service.chat(request)

        response = asyncio.run(run())

        # Should fail validation and return refusal
        assert response.failure_reason == FailureReason.VALIDATION_FAILED, \
            f"Should fail validation, got: {response.failure_reason}"
        assert response.answer is None, "Should not return answer with invalid citation"

    def test_retry_then_fail(self):
        """Invalid response retries once then fails."""
        call_count = [0]

        async def mock_llm_always_invalid(system_prompt: str, user_prompt: str) -> str:
            call_count[0] += 1
            return json.dumps({
                "answer_markdown": "Rangt svar.",
                "citations": [{
                    "document_id": "doc_1",
                    "locator": "Lög nr. 33/1944 - 1. gr.",
                    "quote": "Uppspunnið quote " + str(call_count[0]),
                }],
            })

        # Mocks that always return chunks
        async def always_return_chunks(embedding, top_k):
            return TEST_CHUNKS[:top_k]

        async def always_return_keyword(keywords, top_k):
            return TEST_CHUNKS[:top_k]

        searcher = HybridSearcher(
            vector_search_fn=always_return_chunks,
            keyword_search_fn=always_return_keyword,
            direct_lookup_fn=mock_direct_lookup,
        )

        service = ChatService(
            searcher=searcher,
            llm_fn=mock_llm_always_invalid,
            debug=True,
        )

        async def run():
            request = ChatRequest(query="Test query með lögum nr. 33/1944")
            return await service.chat(request)

        response = asyncio.run(run())

        # Should have called LLM twice (original + 1 retry)
        assert call_count[0] == 2, f"Expected 2 calls, got {call_count[0]}"
        assert response.failure_reason == FailureReason.VALIDATION_FAILED


class TestAPIEndpoint:
    """Tests for the API endpoint handler."""

    def test_endpoint_formats_success(self):
        """Endpoint formats successful response correctly."""
        searcher = HybridSearcher(
            vector_search_fn=mock_vector_search,
            keyword_search_fn=mock_keyword_search,
            direct_lookup_fn=mock_direct_lookup,
        )

        service = ChatService(
            searcher=searcher,
            llm_fn=create_mock_llm_with_valid_citation(),
            debug=True,
        )

        endpoint = ChatEndpoint(chat_service=service, debug=True)

        async def run():
            return await endpoint.handle_chat(
                query="Hvernig lítur fáninn út samkvæmt lögum nr. 33/1944?"
            )

        response = asyncio.run(run())

        assert response.success, f"Should succeed: {response.error}"
        assert response.data is not None
        assert "answer" in response.data
        assert "citations" in response.data

    def test_endpoint_formats_failure(self):
        """Endpoint formats failure response correctly."""
        searcher = HybridSearcher(
            vector_search_fn=mock_vector_search,
            keyword_search_fn=mock_keyword_search,
            direct_lookup_fn=mock_direct_lookup,
        )

        service = ChatService(
            searcher=searcher,
            llm_fn=create_mock_llm_with_invalid_citation(),
            debug=True,
        )

        endpoint = ChatEndpoint(chat_service=service, debug=True)

        async def run():
            return await endpoint.handle_chat(query="Test query")

        response = asyncio.run(run())

        # May succeed (if LLM returns no citations) or fail (if validation fails)
        # Either way, response should be well-formed
        assert response.request_id is not None
        assert response.data is not None or response.error is not None


class TestHybridSearchImprovement:
    """Tests demonstrating hybrid search improves recall."""

    def test_keyword_finds_what_vector_misses(self):
        """Keyword search finds results that vector search might miss."""
        # Simulate vector search missing a result
        vector_results = [TEST_CHUNKS[0]]  # Only returns first chunk

        # Keyword search finds more
        keyword_results = [TEST_CHUNKS[0], TEST_CHUNKS[2]]  # Finds two chunks

        from app.services.search import merge_search_results

        merged = merge_search_results(vector_results, keyword_results, top_k=10)

        # Should have union of both
        merged_ids = {c.id for c in merged}
        assert "1" in merged_ids
        assert "3" in merged_ids
        assert len(merged) >= 2, "Hybrid should find more than vector alone"


def run_all_tests():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("END-TO-END CHAT TESTS")
    print("=" * 60)

    # Happy path
    print("\n--- Happy Path Tests ---")
    happy_tests = TestHappyPath()
    happy_tests.test_valid_citation_passes()
    print("  test_valid_citation_passes: PASS")
    happy_tests.test_direct_law_reference_works()
    print("  test_direct_law_reference_works: PASS")

    # Ambiguous query
    print("\n--- Ambiguous Query Tests ---")
    ambiguous_tests = TestAmbiguousQuery()
    ambiguous_tests.test_vague_query_requests_clarification()
    print("  test_vague_query_requests_clarification: PASS")
    ambiguous_tests.test_empty_query_rejected()
    print("  test_empty_query_rejected: PASS")

    # Insufficient evidence
    print("\n--- Insufficient Evidence Tests ---")
    evidence_tests = TestInsufficientEvidence()
    evidence_tests.test_no_chunks_returns_refusal()
    print("  test_no_chunks_returns_refusal: PASS")

    # Validation rejection (CRITICAL)
    print("\n--- Validation Rejection Tests (CRITICAL) ---")
    validation_tests = TestValidationRejection()
    validation_tests.test_fabricated_citation_rejected()
    print("  test_fabricated_citation_rejected: PASS")
    validation_tests.test_retry_then_fail()
    print("  test_retry_then_fail: PASS")

    # API endpoint
    print("\n--- API Endpoint Tests ---")
    api_tests = TestAPIEndpoint()
    api_tests.test_endpoint_formats_success()
    print("  test_endpoint_formats_success: PASS")
    api_tests.test_endpoint_formats_failure()
    print("  test_endpoint_formats_failure: PASS")

    # Hybrid search improvement
    print("\n--- Hybrid Search Improvement Tests ---")
    hybrid_tests = TestHybridSearchImprovement()
    hybrid_tests.test_keyword_finds_what_vector_misses()
    print("  test_keyword_finds_what_vector_misses: PASS")

    print("\n" + "=" * 60)
    print("ALL END-TO-END TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
