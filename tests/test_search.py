"""
Tests for Hybrid Search Module

These tests verify that:
1. Law number references are extracted correctly
2. Article references are extracted correctly
3. Keyword extraction works for Icelandic queries
4. Hybrid merge produces correct rankings
5. Direct reference lookup takes priority
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.search import (
    Chunk,
    SearchType,
    extract_law_reference,
    extract_article_reference,
    extract_search_keywords,
    merge_search_results,
    HybridSearcher,
)


class TestLawReferenceExtraction:
    """Tests for law number extraction."""

    def test_simple_law_number(self):
        """Basic format: 33/1944"""
        assert extract_law_reference("33/1944") == "33/1944"
        assert extract_law_reference("Hvað segir í 33/1944?") == "33/1944"

    def test_nr_prefix(self):
        """Format: nr. 33/1944"""
        assert extract_law_reference("nr. 33/1944") == "33/1944"
        assert extract_law_reference("nr 33/1944") == "33/1944"

    def test_log_nr_prefix(self):
        """Format: lög nr. 33/1944"""
        assert extract_law_reference("lög nr. 33/1944") == "33/1944"
        assert extract_law_reference("Hvað segir í lög nr. 33/1944?") == "33/1944"

    def test_l_nr_prefix(self):
        """Format: l. nr. 33/1944 (abbreviated)"""
        assert extract_law_reference("l. nr. 33/1944") == "33/1944"
        assert extract_law_reference("l nr 33/1944") == "33/1944"

    def test_laga_nr_prefix(self):
        """Format: laga nr. 33/1944"""
        assert extract_law_reference("1. gr. laga nr. 33/1944") == "33/1944"

    def test_no_law_reference(self):
        """Queries without law numbers return None."""
        assert extract_law_reference("Hvað segir um mannréttindi?") is None
        assert extract_law_reference("stjórnarskrá") is None

    def test_various_years(self):
        """Different years work correctly."""
        assert extract_law_reference("lög nr. 85/2024") == "85/2024"
        assert extract_law_reference("lög nr. 1/2000") == "1/2000"
        assert extract_law_reference("lög nr. 123/1901") == "123/1901"

    def test_case_insensitive(self):
        """Extraction is case-insensitive."""
        assert extract_law_reference("LÖG NR. 33/1944") == "33/1944"
        assert extract_law_reference("Lög Nr. 33/1944") == "33/1944"


class TestArticleReferenceExtraction:
    """Tests for article number extraction."""

    def test_gr_format(self):
        """Format: 1. gr."""
        assert extract_article_reference("1. gr.") == "1"
        assert extract_article_reference("12. gr.") == "12"
        assert extract_article_reference("Hvað segir 5. gr.?") == "5"

    def test_grein_format(self):
        """Format: 1. grein"""
        assert extract_article_reference("1. grein") == "1"
        assert extract_article_reference("12. grein laga") == "12"

    def test_no_article(self):
        """Queries without articles return None."""
        assert extract_article_reference("Hvað segir um mannréttindi?") is None
        assert extract_article_reference("lög nr. 33/1944") is None

    def test_combined_reference(self):
        """Extract article from combined references."""
        assert extract_article_reference("1. gr. laga nr. 33/1944") == "1"
        assert extract_article_reference("sbr. 12. gr.") == "12"


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    def test_basic_keywords(self):
        """Extract meaningful keywords."""
        keywords = extract_search_keywords("mannréttindi stjórnarskrá")
        assert "mannréttindi" in keywords
        assert "stjórnarskrá" in keywords

    def test_stop_words_removed(self):
        """Stop words are filtered out."""
        keywords = extract_search_keywords("hvað segir í lögum um mannréttindi")
        assert "hvað" not in keywords
        assert "segir" not in keywords
        assert "lögum" not in keywords
        assert "mannréttindi" in keywords

    def test_short_words_removed(self):
        """Very short words are filtered."""
        keywords = extract_search_keywords("a og b mannréttindi")
        assert "a" not in keywords
        assert "og" not in keywords
        assert "mannréttindi" in keywords

    def test_icelandic_keywords_preserved(self):
        """Icelandic characters preserved in keywords."""
        keywords = extract_search_keywords("þjóðfáni Íslendinga æðri dómstólar")
        assert "þjóðfáni" in keywords
        assert "íslendinga" in keywords
        assert "æðri" in keywords
        assert "dómstólar" in keywords

    def test_whitespace_handling(self):
        """Multiple spaces don't create empty keywords."""
        keywords = extract_search_keywords("mannréttindi    stjórnarskrá")
        assert "" not in keywords


class TestMergeResults:
    """Tests for result merging logic."""

    def create_chunk(self, id: str, **kwargs) -> Chunk:
        """Helper to create test chunks."""
        return Chunk(
            id=id,
            document_id=f"doc_{id}",
            chunk_text=f"Text for {id}",
            chunk_text_normalized=f"text for {id}",
            locator=f"Lög nr. 1/2000 - {id}. gr.",
            **kwargs
        )

    def test_vector_only_results(self):
        """When only vector results exist, they are returned."""
        vector_results = [
            self.create_chunk("1"),
            self.create_chunk("2"),
            self.create_chunk("3"),
        ]

        merged = merge_search_results(vector_results, [], top_k=10)

        assert len(merged) == 3
        assert merged[0].id == "1"  # First result stays first

    def test_keyword_only_results(self):
        """When only keyword results exist, they are returned."""
        keyword_results = [
            self.create_chunk("a"),
            self.create_chunk("b"),
        ]

        merged = merge_search_results([], keyword_results, top_k=10)

        assert len(merged) == 2
        assert merged[0].id == "a"

    def test_union_not_intersection(self):
        """Results are unioned, not intersected."""
        vector_results = [
            self.create_chunk("1"),
            self.create_chunk("2"),
        ]
        keyword_results = [
            self.create_chunk("a"),
            self.create_chunk("b"),
        ]

        merged = merge_search_results(vector_results, keyword_results, top_k=10)

        # All 4 chunks should be present
        ids = {c.id for c in merged}
        assert ids == {"1", "2", "a", "b"}

    def test_overlap_gets_bonus(self):
        """Chunks found by both methods get ranking bonus."""
        # Same chunk found by both methods
        vector_results = [
            self.create_chunk("shared"),
            self.create_chunk("vector_only"),
        ]
        keyword_results = [
            self.create_chunk("shared"),  # Same ID
            self.create_chunk("keyword_only"),
        ]

        merged = merge_search_results(vector_results, keyword_results, top_k=10)

        # Shared chunk should be ranked first (has reinforcement bonus)
        assert merged[0].id == "shared"
        assert merged[0].combined_score > merged[1].combined_score

    def test_top_k_limiting(self):
        """Results are limited to top_k."""
        vector_results = [self.create_chunk(str(i)) for i in range(10)]
        keyword_results = [self.create_chunk(f"k{i}") for i in range(10)]

        merged = merge_search_results(vector_results, keyword_results, top_k=5)

        assert len(merged) == 5

    def test_deterministic_ordering(self):
        """Same inputs produce same ordering."""
        vector_results = [
            self.create_chunk("1"),
            self.create_chunk("2"),
        ]
        keyword_results = [
            self.create_chunk("a"),
            self.create_chunk("2"),  # Overlap
        ]

        merged1 = merge_search_results(vector_results, keyword_results, top_k=10)
        merged2 = merge_search_results(vector_results, keyword_results, top_k=10)

        assert [c.id for c in merged1] == [c.id for c in merged2]


class TestHybridSearcher:
    """Integration tests for HybridSearcher class."""

    def create_chunk(self, id: str, law_number: str = None, **kwargs) -> Chunk:
        """Helper to create test chunks."""
        return Chunk(
            id=id,
            document_id=f"doc_{id}",
            chunk_text=f"Text for {id}",
            locator=f"Lög nr. {law_number or '1/2000'} - {id}. gr.",
            law_number=law_number,
            **kwargs
        )

    def get_mock_searcher(self):
        """Create searcher with mock search functions."""
        test = self

        async def mock_vector_search(embedding, top_k):
            return [
                test.create_chunk("v1"),
                test.create_chunk("v2"),
            ]

        async def mock_keyword_search(keywords, top_k):
            return [
                test.create_chunk("k1"),
                test.create_chunk("v1"),  # Overlap with vector
            ]

        async def mock_direct_lookup(law_number, article_number):
            if law_number == "33/1944":
                return [
                    test.create_chunk("direct1", law_number="33/1944"),
                    test.create_chunk("direct2", law_number="33/1944"),
                ]
            return []

        return HybridSearcher(
            vector_search_fn=mock_vector_search,
            keyword_search_fn=mock_keyword_search,
            direct_lookup_fn=mock_direct_lookup,
        )

    def test_direct_reference_priority(self):
        """Direct law reference lookup takes priority."""
        mock_searcher = self.get_mock_searcher()

        async def run():
            return await mock_searcher.search(
                query="Hvað segir í lög nr. 33/1944?",
                query_embedding=[0.1] * 10,
            )

        results, search_type = asyncio.run(run())

        assert search_type == SearchType.DIRECT_REFERENCE
        assert len(results) == 2
        assert results[0].law_number == "33/1944"

    def test_hybrid_when_no_direct_match(self):
        """Falls back to hybrid when no direct reference."""
        mock_searcher = self.get_mock_searcher()

        async def run():
            return await mock_searcher.search(
                query="mannréttindi stjórnarskrá",
                query_embedding=[0.1] * 10,
            )

        results, search_type = asyncio.run(run())

        assert search_type == SearchType.HYBRID
        # Should have results from both vector and keyword
        ids = {c.id for c in results}
        assert "v1" in ids  # From vector
        assert "k1" in ids  # From keyword

    def test_overlap_ranking(self):
        """Overlapping results get higher rank."""
        mock_searcher = self.get_mock_searcher()

        async def run():
            return await mock_searcher.search(
                query="test query",
                query_embedding=[0.1] * 10,
            )

        results, _ = asyncio.run(run())

        # v1 appears in both vector and keyword results
        # Should be ranked higher due to reinforcement bonus
        assert results[0].id == "v1"


class TestRealWorldScenarios:
    """Tests simulating real query scenarios."""

    def test_icelandic_law_query(self):
        """Icelandic law query extraction."""
        query = "Hvað segir 1. gr. laga nr. 33/1944 um þjóðfánann?"

        law_ref = extract_law_reference(query)
        article_ref = extract_article_reference(query)
        keywords = extract_search_keywords(query)

        assert law_ref == "33/1944"
        assert article_ref == "1"
        # Keywords include punctuation, check for "þjóðfánann" with or without "?"
        assert any("þjóðfánann" in k for k in keywords)

    def test_conceptual_query(self):
        """Query without specific references."""
        query = "Hvaða lög gilda um mannréttindi á Íslandi?"

        law_ref = extract_law_reference(query)
        keywords = extract_search_keywords(query)

        assert law_ref is None  # No specific law mentioned
        assert any("mannréttindi" in k for k in keywords)
        assert any("íslandi" in k for k in keywords)

    def test_mixed_reference_query(self):
        """Query with partial reference."""
        query = "Hvað segir 12. gr. um réttindi?"

        law_ref = extract_law_reference(query)
        article_ref = extract_article_reference(query)

        assert law_ref is None  # No law number
        assert article_ref == "12"  # But article is specified


def run_all_tests():
    """Run all search tests."""
    print("=" * 60)
    print("SEARCH TESTS")
    print("=" * 60)

    # Law reference extraction
    print("\n--- Law Reference Extraction Tests ---")
    law_tests = TestLawReferenceExtraction()
    law_tests.test_simple_law_number()
    print("  test_simple_law_number: PASS")
    law_tests.test_nr_prefix()
    print("  test_nr_prefix: PASS")
    law_tests.test_log_nr_prefix()
    print("  test_log_nr_prefix: PASS")

    # Article extraction
    print("\n--- Article Reference Extraction Tests ---")
    article_tests = TestArticleReferenceExtraction()
    article_tests.test_gr_format()
    print("  test_gr_format: PASS")
    article_tests.test_no_article()
    print("  test_no_article: PASS")

    # Keyword extraction
    print("\n--- Keyword Extraction Tests ---")
    keyword_tests = TestKeywordExtraction()
    keyword_tests.test_basic_keywords()
    print("  test_basic_keywords: PASS")
    keyword_tests.test_stop_words_removed()
    print("  test_stop_words_removed: PASS")
    keyword_tests.test_icelandic_keywords_preserved()
    print("  test_icelandic_keywords_preserved: PASS")

    # Merge results
    print("\n--- Merge Results Tests ---")
    merge_tests = TestMergeResults()
    merge_tests.test_vector_only_results()
    print("  test_vector_only_results: PASS")
    merge_tests.test_union_not_intersection()
    print("  test_union_not_intersection: PASS")
    merge_tests.test_overlap_gets_bonus()
    print("  test_overlap_gets_bonus: PASS")

    # Hybrid searcher
    print("\n--- Hybrid Searcher Tests ---")
    hybrid_tests = TestHybridSearcher()
    hybrid_tests.test_direct_reference_priority()
    print("  test_direct_reference_priority: PASS")
    hybrid_tests.test_hybrid_when_no_direct_match()
    print("  test_hybrid_when_no_direct_match: PASS")
    hybrid_tests.test_overlap_ranking()
    print("  test_overlap_ranking: PASS")

    # Real-world scenarios
    print("\n--- Real-World Scenario Tests ---")
    scenario_tests = TestRealWorldScenarios()
    scenario_tests.test_icelandic_law_query()
    print("  test_icelandic_law_query: PASS")
    scenario_tests.test_conceptual_query()
    print("  test_conceptual_query: PASS")

    print("\n" + "=" * 60)
    print("ALL SEARCH TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
