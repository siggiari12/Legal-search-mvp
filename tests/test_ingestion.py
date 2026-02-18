"""
Tests for Ingestion Pipeline

Tests cover:
1. SGML parsing (extracting law number, year, title, articles)
2. Chunk creation with proper locators
3. Canonicalization applied to all text
4. Sanity checks that fail loudly
5. Integration test: full pipeline produces valid documents and chunks
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.canonicalize import canonicalize
from app.ingestion.parser import SGMLParser, build_locator, SGMLParseError
from app.ingestion.pipeline import (
    IngestionPipeline,
    IngestionConfig,
    IngestionError,
    ingest_laws_batch,
)


# Load test fixtures
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

def load_fixture(filename: str) -> str:
    """Load a test fixture file."""
    with open(os.path.join(FIXTURE_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()


class TestSGMLParser:
    """Tests for SGML parsing."""

    def test_parse_law_number_year(self):
        """Extract law number and year from SGML."""
        parser = SGMLParser()
        content = load_fixture('sample_law.sgml')
        parsed = parser.parse(content, "sample_law.sgml")

        assert parsed.law_number == "33", f"Expected '33', got '{parsed.law_number}'"
        assert parsed.law_year == "1944", f"Expected '1944', got '{parsed.law_year}'"

    def test_parse_title(self):
        """Extract title from SGML."""
        parser = SGMLParser()
        content = load_fixture('sample_law.sgml')
        parsed = parser.parse(content, "sample_law.sgml")

        assert "þjóðfána" in parsed.title.lower()
        assert "Íslendinga" in parsed.title

    def test_parse_articles(self):
        """Extract articles from SGML."""
        parser = SGMLParser()
        content = load_fixture('sample_law.sgml')
        parsed = parser.parse(content, "sample_law.sgml")

        assert len(parsed.articles) == 3, f"Expected 3 articles, got {len(parsed.articles)}"

        # Check first article
        assert parsed.articles[0].number == "1"
        assert "himinblár" in parsed.articles[0].text

    def test_parse_paragraphs(self):
        """Extract paragraphs from articles."""
        parser = SGMLParser()
        content = load_fixture('sample_law.sgml')
        parsed = parser.parse(content, "sample_law.sgml")

        # First article should have 2 paragraphs
        first_article = parsed.articles[0]
        assert len(first_article.paragraphs) >= 2, f"Expected >=2 paragraphs, got {len(first_article.paragraphs)}"

    def test_canonicalization_applied(self):
        """Text should be canonicalized during parsing."""
        parser = SGMLParser()

        # SGML with extra whitespace
        sgml = """
        <log>
            <nr>1</nr>
            <ar>2020</ar>
            <heiti>Test    Law   With   Spaces</heiti>
            <grein>
                <nr>1</nr>
                <mgr>Text  with   extra    whitespace.</mgr>
            </grein>
        </log>
        """

        parsed = parser.parse(sgml, "test")

        # Title should have collapsed whitespace
        assert "    " not in parsed.title
        assert "Test Law With Spaces" in parsed.title

        # Article text should be canonicalized
        assert "    " not in parsed.articles[0].text

    def test_parse_second_law(self):
        """Parse a different law structure."""
        parser = SGMLParser()
        content = load_fixture('sample_law_2.sgml')
        parsed = parser.parse(content, "sample_law_2.sgml")

        assert parsed.law_number == "85"
        assert parsed.law_year == "2024"
        assert "gervigreind" in parsed.title.lower()

    def test_icelandic_characters_preserved(self):
        """Icelandic characters should be preserved in parsing."""
        parser = SGMLParser()
        content = load_fixture('sample_law.sgml')
        parsed = parser.parse(content, "sample_law.sgml")

        # Check for Icelandic characters
        assert "þjóðfána" in parsed.title.lower() or "Þjóðfána" in parsed.title
        assert "Íslendinga" in parsed.title


class TestBuildLocator:
    """Tests for locator construction."""

    def test_basic_locator(self):
        """Build basic law locator."""
        loc = build_locator("33", "1944")
        assert loc == "Lög nr. 33/1944"

    def test_locator_with_article(self):
        """Build locator with article number."""
        loc = build_locator("33", "1944", article_number="1")
        assert loc == "Lög nr. 33/1944 - 1. gr."

    def test_locator_with_paragraph(self):
        """Build locator with article and paragraph."""
        loc = build_locator("33", "1944", article_number="1", paragraph_number="2")
        assert loc == "Lög nr. 33/1944 - 1. gr., 2. mgr."


class TestIngestionPipeline:
    """Tests for ingestion pipeline."""

    def test_create_document(self):
        """Pipeline creates document with correct fields."""
        config = IngestionConfig(strict_validation=True)
        pipeline = IngestionPipeline(config)

        parser = SGMLParser()
        content = load_fixture('sample_law.sgml')
        parsed = parser.parse(content, "test")

        doc = pipeline._create_document(parsed)

        assert doc.law_number == "33"
        assert doc.law_year == "1944"
        assert doc.source == "Althingi"
        assert doc.document_type == "law"
        assert "þjóðfána" in doc.title.lower() or "Þjóðfána" in doc.title
        assert doc.id is not None

    def test_create_chunks(self):
        """Pipeline creates chunks for each article."""
        config = IngestionConfig(strict_validation=True)
        pipeline = IngestionPipeline(config)

        parser = SGMLParser()
        content = load_fixture('sample_law.sgml')
        parsed = parser.parse(content, "test")

        doc = pipeline._create_document(parsed)
        chunks = pipeline._create_chunks(doc, parsed)

        # Should have at least one chunk per article
        assert len(chunks) >= 3

        # Each chunk should have a locator with law reference
        for chunk in chunks:
            assert chunk.locator is not None
            assert "33/1944" in chunk.locator

    def test_chunks_are_canonicalized(self):
        """Chunk text should be canonicalized."""
        config = IngestionConfig(strict_validation=True)
        pipeline = IngestionPipeline(config)

        parser = SGMLParser()
        content = load_fixture('sample_law.sgml')
        parsed = parser.parse(content, "test")

        doc = pipeline._create_document(parsed)
        chunks = pipeline._create_chunks(doc, parsed)

        for chunk in chunks:
            # Text should equal its canonicalized form
            assert chunk.chunk_text == canonicalize(chunk.chunk_text)
            # No multiple consecutive spaces
            assert "  " not in chunk.chunk_text

    def test_validation_empty_title_fails(self):
        """Validation should fail for empty title."""
        config = IngestionConfig(strict_validation=True)
        pipeline = IngestionPipeline(config)

        from app.models.schemas import Document, Chunk

        doc = Document.create(
            title="",  # Empty title
            law_number="1",
            law_year="2020",
            full_text="Some text",
        )

        chunks = [Chunk.create(
            document_id=doc.id,
            chunk_text="Test chunk",
            locator="Lög nr. 1/2020 - 1. gr.",
        )]

        try:
            pipeline._validate_ingestion(doc, chunks, "test")
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "empty" in str(e).lower()

    def test_validation_no_chunks_fails(self):
        """Validation should fail when no chunks."""
        config = IngestionConfig(strict_validation=True, min_chunks_per_law=1)
        pipeline = IngestionPipeline(config)

        from app.models.schemas import Document

        doc = Document.create(
            title="Test Law",
            law_number="1",
            law_year="2020",
            full_text="Some text",
        )

        try:
            pipeline._validate_ingestion(doc, [], "test")
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "chunk" in str(e).lower()

    def test_validation_empty_chunk_fails(self):
        """Validation should fail for empty chunk text."""
        config = IngestionConfig(strict_validation=True)
        pipeline = IngestionPipeline(config)

        from app.models.schemas import Document, Chunk

        doc = Document.create(
            title="Test Law",
            law_number="1",
            law_year="2020",
            full_text="Some text",
        )

        chunks = [Chunk.create(
            document_id=doc.id,
            chunk_text="",  # Empty
            locator="Lög nr. 1/2020 - 1. gr.",
        )]

        try:
            pipeline._validate_ingestion(doc, chunks, "test")
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "empty" in str(e).lower()


class TestIngestionIntegration:
    """Integration tests for full ingestion flow."""

    def test_full_ingestion_dry_run(self):
        """Test full ingestion without database storage."""
        config = IngestionConfig(
            strict_validation=True,
            version_tag="test-2024-01-15",
        )

        # Run without store functions (dry run)
        pipeline = IngestionPipeline(config)

        content = load_fixture('sample_law.sgml')

        async def run():
            return await pipeline.ingest_sgml(content, "sample_law.sgml")

        result = asyncio.run(run())

        assert result.success, f"Ingestion failed: {result.errors}"
        assert result.document_id is not None
        assert result.chunk_count >= 3  # At least 3 articles

    def test_batch_ingestion(self):
        """Test batch ingestion of multiple laws."""
        config = IngestionConfig(
            strict_validation=True,
            version_tag="test-batch",
        )

        sgml_files = [
            {"content": load_fixture('sample_law.sgml'), "source_info": "law_33_1944.sgml"},
            {"content": load_fixture('sample_law_2.sgml'), "source_info": "law_85_2024.sgml"},
        ]

        async def run():
            return await ingest_laws_batch(sgml_files, config)

        report = asyncio.run(run())

        assert report.total_laws == 2
        assert report.successful == 2
        assert report.failed == 0
        assert report.total_chunks >= 6  # At least 3 per law


def run_all_tests():
    """Run all ingestion tests."""
    print("=" * 60)
    print("INGESTION TESTS")
    print("=" * 60)

    # Parser tests
    print("\n--- SGML Parser Tests ---")
    parser_tests = TestSGMLParser()
    parser_tests.test_parse_law_number_year()
    print("  test_parse_law_number_year: PASS")
    parser_tests.test_parse_title()
    print("  test_parse_title: PASS")
    parser_tests.test_parse_articles()
    print("  test_parse_articles: PASS")
    parser_tests.test_parse_paragraphs()
    print("  test_parse_paragraphs: PASS")
    parser_tests.test_canonicalization_applied()
    print("  test_canonicalization_applied: PASS")
    parser_tests.test_parse_second_law()
    print("  test_parse_second_law: PASS")
    parser_tests.test_icelandic_characters_preserved()
    print("  test_icelandic_characters_preserved: PASS")

    # Locator tests
    print("\n--- Locator Tests ---")
    locator_tests = TestBuildLocator()
    locator_tests.test_basic_locator()
    print("  test_basic_locator: PASS")
    locator_tests.test_locator_with_article()
    print("  test_locator_with_article: PASS")
    locator_tests.test_locator_with_paragraph()
    print("  test_locator_with_paragraph: PASS")

    # Pipeline tests
    print("\n--- Pipeline Tests ---")
    pipeline_tests = TestIngestionPipeline()
    pipeline_tests.test_create_document()
    print("  test_create_document: PASS")
    pipeline_tests.test_create_chunks()
    print("  test_create_chunks: PASS")
    pipeline_tests.test_chunks_are_canonicalized()
    print("  test_chunks_are_canonicalized: PASS")
    pipeline_tests.test_validation_empty_title_fails()
    print("  test_validation_empty_title_fails: PASS")
    pipeline_tests.test_validation_no_chunks_fails()
    print("  test_validation_no_chunks_fails: PASS")
    pipeline_tests.test_validation_empty_chunk_fails()
    print("  test_validation_empty_chunk_fails: PASS")

    # Integration tests
    print("\n--- Integration Tests ---")
    integration_tests = TestIngestionIntegration()
    integration_tests.test_full_ingestion_dry_run()
    print("  test_full_ingestion_dry_run: PASS")
    integration_tests.test_batch_ingestion()
    print("  test_batch_ingestion: PASS")

    print("\n" + "=" * 60)
    print("ALL INGESTION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
