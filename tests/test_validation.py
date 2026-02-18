"""
Tests for Validation Layer

CRITICAL TESTS - These ensure the system never returns invalid citations.

Tests cover:
1. Valid quotes pass validation
2. Fabricated quotes fail validation
3. NBSP/whitespace differences still pass (due to canonicalization)
4. Empty quotes fail
5. Missing locators fail
6. Invalid locator format fails
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.canonicalize import canonicalize
from app.services.validation import (
    ResponseValidator,
    ValidationContext,
    ValidationResult,
)
from app.models.schemas import Chunk


def create_chunk(id: str, text: str, locator: str, **kwargs) -> Chunk:
    """Helper to create test chunks."""
    return Chunk(
        id=id,
        document_id=kwargs.get("document_id", f"doc_{id}"),
        chunk_text=text,
        locator=locator,
        law_number=kwargs.get("law_number", "33"),
        law_year=kwargs.get("law_year", "1944"),
    )


# Sample chunks for testing
SAMPLE_CHUNKS = [
    create_chunk(
        id="1",
        text="Þjóðfáni Íslendinga er himinblár með mjóum hvítum krossi, rauðum að innanverðu.",
        locator="Lög nr. 33/1944 - 1. gr.",
    ),
    create_chunk(
        id="2",
        text="Stærðarhlutföll fánans skulu vera þannig, að breiddin sé 18/25 af lengdinni.",
        locator="Lög nr. 33/1944 - 2. gr.",
    ),
]


class TestValidQuotes:
    """Tests for valid quote validation."""

    def test_exact_quote_passes(self):
        """Exact quote from source passes validation."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Þjóðfáninn er blár.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "Þjóðfáni Íslendinga er himinblár",
            }]
        }

        result = validator.validate_response(response, context)
        assert result.valid, f"Should pass: {result.error_summary}"

    def test_partial_quote_passes(self):
        """Partial quote (substring) passes validation."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Krossinn er hvítur og rauður.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "hvítum krossi, rauðum að innanverðu",
            }]
        }

        result = validator.validate_response(response, context)
        assert result.valid, f"Should pass: {result.error_summary}"

    def test_multiple_valid_citations_pass(self):
        """Multiple valid citations all pass."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Um fánann.",
            "citations": [
                {
                    "document_id": "doc_1",
                    "locator": "Lög nr. 33/1944 - 1. gr.",
                    "quote": "himinblár",
                },
                {
                    "document_id": "doc_2",
                    "locator": "Lög nr. 33/1944 - 2. gr.",
                    "quote": "breiddin sé 18/25",
                },
            ]
        }

        result = validator.validate_response(response, context)
        assert result.valid, f"Should pass: {result.error_summary}"


class TestInvalidQuotes:
    """Tests for invalid quote detection - CRITICAL."""

    def test_fabricated_quote_fails(self):
        """Fabricated quote that doesn't exist in source fails."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Test answer.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "Þetta er algjörlega uppspunnið og ekki í textanum",
            }]
        }

        result = validator.validate_response(response, context)
        assert not result.valid, "Fabricated quote should fail"
        assert any(e.error_type == "quote_not_found" for e in result.errors)

    def test_modified_quote_fails(self):
        """Quote with modified word fails."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        # Original: "himinblár" -> Modified: "djúpblár"
        response = {
            "answer_markdown": "Test.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "Þjóðfáni Íslendinga er djúpblár",  # Wrong color
            }]
        }

        result = validator.validate_response(response, context)
        assert not result.valid, "Modified quote should fail"

    def test_empty_quote_fails(self):
        """Empty quote fails."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Test.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "",
            }]
        }

        result = validator.validate_response(response, context)
        assert not result.valid, "Empty quote should fail"
        assert any(e.error_type == "empty_quote" for e in result.errors)

    def test_whitespace_only_quote_fails(self):
        """Whitespace-only quote fails."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Test.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "   \n\t  ",
            }]
        }

        result = validator.validate_response(response, context)
        assert not result.valid, "Whitespace-only quote should fail"


class TestCanonicalization:
    """Tests ensuring canonicalization allows minor formatting differences."""

    def test_nbsp_difference_passes(self):
        """Quote with NBSP vs regular space passes due to canonicalization."""
        # Create chunk with NBSP
        chunks = [create_chunk(
            id="1",
            text="Þjóðfáni\u00A0Íslendinga er himinblár",  # NBSP
            locator="Lög nr. 33/1944 - 1. gr.",
        )]

        validator = ResponseValidator()
        context = ValidationContext.from_chunks(chunks)

        # Quote with regular space
        response = {
            "answer_markdown": "Test.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "Þjóðfáni Íslendinga er himinblár",  # Regular space
            }]
        }

        result = validator.validate_response(response, context)
        assert result.valid, f"NBSP difference should pass: {result.error_summary}"

    def test_extra_whitespace_passes(self):
        """Quote with different whitespace passes."""
        # Create chunk with extra spaces
        chunks = [create_chunk(
            id="1",
            text="Þjóðfáni  Íslendinga   er himinblár",  # Extra spaces
            locator="Lög nr. 33/1944 - 1. gr.",
        )]

        validator = ResponseValidator()
        context = ValidationContext.from_chunks(chunks)

        # Quote with single spaces
        response = {
            "answer_markdown": "Test.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "Þjóðfáni Íslendinga er himinblár",
            }]
        }

        result = validator.validate_response(response, context)
        assert result.valid, f"Whitespace difference should pass: {result.error_summary}"

    def test_newline_difference_passes(self):
        """Quote spanning newline in source passes."""
        chunks = [create_chunk(
            id="1",
            text="Þjóðfáni Íslendinga\ner himinblár",  # Newline
            locator="Lög nr. 33/1944 - 1. gr.",
        )]

        validator = ResponseValidator()
        context = ValidationContext.from_chunks(chunks)

        response = {
            "answer_markdown": "Test.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Lög nr. 33/1944 - 1. gr.",
                "quote": "Þjóðfáni Íslendinga er himinblár",  # Single space
            }]
        }

        result = validator.validate_response(response, context)
        assert result.valid, f"Newline difference should pass: {result.error_summary}"


class TestLocatorValidation:
    """Tests for locator validation."""

    def test_empty_locator_fails(self):
        """Empty locator fails."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Test.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "",
                "quote": "himinblár",
            }]
        }

        result = validator.validate_response(response, context)
        assert not result.valid, "Empty locator should fail"
        assert any(e.error_type == "empty_locator" for e in result.errors)

    def test_invalid_locator_format_fails(self):
        """Locator without law reference fails."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Test.",
            "citations": [{
                "document_id": "doc_1",
                "locator": "Something random without law number",
                "quote": "himinblár",
            }]
        }

        result = validator.validate_response(response, context)
        assert not result.valid, "Invalid locator format should fail"
        assert any(e.error_type == "invalid_locator" for e in result.errors)

    def test_valid_locator_formats_pass(self):
        """Various valid locator formats pass."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        valid_locators = [
            "Lög nr. 33/1944",
            "Lög nr. 33/1944 - 1. gr.",
            "Lög nr. 33/1944 - 1. gr., 2. mgr.",
            "33/1944",  # Minimal format
            "sbr. 33/1944",
        ]

        for locator in valid_locators:
            response = {
                "answer_markdown": "Test.",
                "citations": [{
                    "document_id": "doc_1",
                    "locator": locator,
                    "quote": "himinblár",
                }]
            }

            result = validator.validate_response(response, context)
            assert result.valid, f"Locator '{locator}' should be valid"


class TestAnswerWithoutCitations:
    """Tests for answers without citations."""

    def test_answer_without_citations_fails(self):
        """Answer with factual content but no citations fails."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Þjóðfáninn er blár og með krossi.",
            "citations": []
        }

        result = validator.validate_response(response, context)
        assert not result.valid, "Answer without citations should fail"

    def test_refusal_without_citations_passes(self):
        """Refusal response without citations passes."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Ég get ekki svarað þessari spurningu.",
            "citations": []
        }

        result = validator.validate_response(response, context)
        assert result.valid, "Refusal should pass without citations"

    def test_clarification_without_citations_passes(self):
        """Clarification request without citations passes."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Vinsamlegast tilgreindu nánar hvað þú leitar að.",
            "citations": []
        }

        result = validator.validate_response(response, context)
        assert result.valid, "Clarification should pass without citations"


class TestOneFailureRejectsAll:
    """Tests ensuring one failed citation rejects entire response."""

    def test_one_bad_citation_fails_all(self):
        """One invalid citation among valid ones fails entire response."""
        validator = ResponseValidator()
        context = ValidationContext.from_chunks(SAMPLE_CHUNKS)

        response = {
            "answer_markdown": "Test.",
            "citations": [
                {
                    "document_id": "doc_1",
                    "locator": "Lög nr. 33/1944 - 1. gr.",
                    "quote": "himinblár",  # Valid
                },
                {
                    "document_id": "doc_2",
                    "locator": "Lög nr. 33/1944 - 2. gr.",
                    "quote": "Þetta er uppspunnið",  # Invalid
                },
            ]
        }

        result = validator.validate_response(response, context)
        assert not result.valid, "One bad citation should fail entire response"


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("VALIDATION TESTS")
    print("=" * 60)

    # Valid quotes
    print("\n--- Valid Quote Tests ---")
    valid_tests = TestValidQuotes()
    valid_tests.test_exact_quote_passes()
    print("  test_exact_quote_passes: PASS")
    valid_tests.test_partial_quote_passes()
    print("  test_partial_quote_passes: PASS")
    valid_tests.test_multiple_valid_citations_pass()
    print("  test_multiple_valid_citations_pass: PASS")

    # Invalid quotes
    print("\n--- Invalid Quote Tests (CRITICAL) ---")
    invalid_tests = TestInvalidQuotes()
    invalid_tests.test_fabricated_quote_fails()
    print("  test_fabricated_quote_fails: PASS")
    invalid_tests.test_modified_quote_fails()
    print("  test_modified_quote_fails: PASS")
    invalid_tests.test_empty_quote_fails()
    print("  test_empty_quote_fails: PASS")
    invalid_tests.test_whitespace_only_quote_fails()
    print("  test_whitespace_only_quote_fails: PASS")

    # Canonicalization
    print("\n--- Canonicalization Tests ---")
    canon_tests = TestCanonicalization()
    canon_tests.test_nbsp_difference_passes()
    print("  test_nbsp_difference_passes: PASS")
    canon_tests.test_extra_whitespace_passes()
    print("  test_extra_whitespace_passes: PASS")
    canon_tests.test_newline_difference_passes()
    print("  test_newline_difference_passes: PASS")

    # Locator validation
    print("\n--- Locator Validation Tests ---")
    locator_tests = TestLocatorValidation()
    locator_tests.test_empty_locator_fails()
    print("  test_empty_locator_fails: PASS")
    locator_tests.test_invalid_locator_format_fails()
    print("  test_invalid_locator_format_fails: PASS")
    locator_tests.test_valid_locator_formats_pass()
    print("  test_valid_locator_formats_pass: PASS")

    # Answer without citations
    print("\n--- Answer Without Citations Tests ---")
    no_cite_tests = TestAnswerWithoutCitations()
    no_cite_tests.test_answer_without_citations_fails()
    print("  test_answer_without_citations_fails: PASS")
    no_cite_tests.test_refusal_without_citations_passes()
    print("  test_refusal_without_citations_passes: PASS")
    no_cite_tests.test_clarification_without_citations_passes()
    print("  test_clarification_without_citations_passes: PASS")

    # One failure rejects all
    print("\n--- One Failure Rejects All Tests ---")
    reject_tests = TestOneFailureRejectsAll()
    reject_tests.test_one_bad_citation_fails_all()
    print("  test_one_bad_citation_fails_all: PASS")

    print("\n" + "=" * 60)
    print("ALL VALIDATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
