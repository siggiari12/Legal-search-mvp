"""
Integration Test: Hybrid Search End-to-End

This test demonstrates the complete flow:
  Query -> Canonicalize -> Hybrid Search -> Merged Results

Uses mock data to simulate real Icelandic legal text.
Runs in all environments (no external services needed).
"""

import asyncio
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.canonicalize import canonicalize, quote_exists_in_source
from app.services.search import (
    Chunk,
    HybridSearcher,
    SearchType,
    extract_law_reference,
    merge_search_results,
)


# Sample Icelandic legal data (simulating Lagasafn)
SAMPLE_LAWS = {
    "33/1944": {
        "title": "Lög um þjóðfána Íslendinga og ríkisskjaldarmerkið",
        "articles": [
            {
                "number": "1",
                "text": "Þjóðfáni Íslendinga er himinblár með mjóum hvítum krossi, rauðum að innanverðu. Armar krossins ná að jöðrum fánans."
            },
            {
                "number": "2",
                "text": "Stærðarhlutföll fánans skulu vera þannig, að breiddin sé 18/25 af lengdinni."
            },
        ]
    },
    "33/1944-stjórnarskrá": {
        "title": "Stjórnarskrá lýðveldisins Íslands",
        "articles": [
            {
                "number": "65",
                "text": "Allir skulu vera jafnir fyrir lögum og njóta mannréttinda án tillits til kynferðis, trúarbragða, skoðana, þjóðernisuppruna, kynþáttar, litarháttar, efnahags, ætternis og stöðu að öðru leyti."
            },
        ]
    }
}


def create_test_chunks():
    """Create test chunks from sample data."""
    chunks = []
    chunk_id = 1

    for law_number, law_data in SAMPLE_LAWS.items():
        for article in law_data["articles"]:
            normalized_text = canonicalize(article["text"])

            chunk = Chunk(
                id=str(chunk_id),
                document_id=f"doc_{law_number}",
                chunk_text=article["text"],
                chunk_text_normalized=normalized_text,
                locator=f"Lög nr. {law_number} - {article['number']}. gr.",
                article_number=article["number"],
                law_number=law_number,
            )
            chunks.append(chunk)
            chunk_id += 1

    return chunks


TEST_CHUNKS = create_test_chunks()


async def mock_vector_search(embedding, top_k):
    """Simulate vector search."""
    results = [c for c in TEST_CHUNKS if "fáni" in c.chunk_text.lower() or "mannréttind" in c.chunk_text.lower()]
    return results[:top_k]


async def mock_keyword_search(keywords, top_k):
    """Simulate keyword search."""
    results = []
    for chunk in TEST_CHUNKS:
        normalized = chunk.chunk_text_normalized.lower()
        for keyword in keywords:
            if keyword.lower() in normalized:
                results.append(chunk)
                break
    return results[:top_k]


async def mock_direct_lookup(law_number, article_number):
    """Simulate direct law/article lookup."""
    results = []
    for chunk in TEST_CHUNKS:
        if chunk.law_number == law_number:
            if article_number is None or chunk.article_number == article_number:
                results.append(chunk)
    return results


@pytest.fixture
def searcher():
    return HybridSearcher(
        vector_search_fn=mock_vector_search,
        keyword_search_fn=mock_keyword_search,
        direct_lookup_fn=mock_direct_lookup,
    )


class TestDirectLawReference:
    @pytest.mark.asyncio
    async def test_direct_law_reference_uses_direct_lookup(self, searcher):
        query = "Hvað segir í lög nr. 33/1944?"
        results, search_type = await searcher.search(
            query=query,
            query_embedding=[0.1] * 10,
        )
        assert search_type == SearchType.DIRECT_REFERENCE
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_direct_reference_extracts_law_number(self):
        ref = extract_law_reference("Hvað segir í lög nr. 33/1944?")
        assert ref is not None


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_conceptual_query_returns_results(self, searcher):
        query = "mannréttindi jafnrétti"
        results, search_type = await searcher.search(
            query=query,
            query_embedding=[0.1] * 10,
        )
        assert len(results) > 0


class TestQuoteValidationAfterSearch:
    def test_valid_quote_found_in_source(self):
        flag_chunk = [c for c in TEST_CHUNKS if "Þjóðfáni" in c.chunk_text][0]
        llm_quote = "Þjóðfáni Íslendinga er himinblár"
        assert quote_exists_in_source(llm_quote, flag_chunk.chunk_text)

    def test_nbsp_difference_still_matches(self):
        source_with_nbsp = "Þjóðfáni\u00A0Íslendinga er himinblár"
        llm_quote_spaces = "Þjóðfáni Íslendinga er himinblár"
        assert quote_exists_in_source(llm_quote_spaces, source_with_nbsp)

    def test_fabricated_quote_rejected(self):
        flag_chunk = [c for c in TEST_CHUNKS if "Þjóðfáni" in c.chunk_text][0]
        fake_quote = "Þetta er rangt og ekki í textanum"
        assert not quote_exists_in_source(fake_quote, flag_chunk.chunk_text)


class TestMergePrioritizesOverlap:
    def test_overlap_ranked_first(self):
        vector_res = [TEST_CHUNKS[0], TEST_CHUNKS[1]]
        keyword_res = [TEST_CHUNKS[0], TEST_CHUNKS[2]] if len(TEST_CHUNKS) > 2 else [TEST_CHUNKS[0]]
        merged = merge_search_results(vector_res, keyword_res, top_k=10)
        assert merged[0].id == TEST_CHUNKS[0].id
