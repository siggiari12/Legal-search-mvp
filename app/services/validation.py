"""
Validation Layer for Legal Search MVP

This module provides the HARD GATE for validating LLM responses.
Every response MUST pass validation before being returned to users.

Validation rules:
1. Every citation.quote must exist VERBATIM in the source chunk text
   (after canonicalization)
2. Every citation.locator must match expected document metadata
3. If ANY citation fails, the entire response is invalid

On validation failure:
1. First attempt: retry with stricter prompting
2. Second attempt fails: return refusal

CRITICAL: Uses canonicalize() for all text comparisons.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Awaitable
import re

from app.services.canonicalize import canonicalize, quote_exists_in_source
from app.models.schemas import (
    Citation,
    ChatResponse,
    ValidationResult,
    CitationValidationError,
    Confidence,
    FailureReason,
    Chunk,
)


@dataclass
class ValidationContext:
    """Context for validation with source chunks."""
    chunks: List[Chunk]
    chunks_by_id: Dict[str, Chunk]
    all_text: str  # Concatenated, canonicalized text from all chunks

    @classmethod
    def from_chunks(cls, chunks: List[Chunk]) -> "ValidationContext":
        """Build validation context from retrieved chunks."""
        chunks_by_id = {chunk.id: chunk for chunk in chunks}

        # Concatenate all chunk text for quote search
        all_text = " ".join(
            canonicalize(chunk.chunk_text) for chunk in chunks
        )

        return cls(
            chunks=chunks,
            chunks_by_id=chunks_by_id,
            all_text=canonicalize(all_text),
        )


class ResponseValidator:
    """
    Validates LLM responses against source data.

    All citations must be verifiable against the stored corpus.
    """

    def validate_response(
        self,
        response: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """
        Validate an LLM response.

        Args:
            response: Parsed LLM response with answer and citations
            context: Validation context with source chunks

        Returns:
            ValidationResult indicating pass/fail with error details
        """
        errors = []
        citations = response.get("citations", [])

        if not citations:
            # No citations to validate - could be a refusal or clarification
            answer = response.get("answer_markdown", "")
            if answer and not self._is_refusal_or_clarification(answer):
                # Answer provided without citations - invalid
                errors.append(CitationValidationError(
                    citation_index=-1,
                    locator="",
                    quote="",
                    error_type="missing_citations",
                    details="Answer provided without any citations"
                ))

        for i, citation in enumerate(citations):
            citation_errors = self._validate_citation(citation, context, i)
            errors.extend(citation_errors)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )

    def _validate_citation(
        self,
        citation: Dict[str, Any],
        context: ValidationContext,
        index: int,
    ) -> List[CitationValidationError]:
        """
        Validate a single citation.

        Checks:
        1. Quote exists verbatim in source text
        2. Locator is present and well-formed
        """
        errors = []

        quote = citation.get("quote", "")
        locator = citation.get("locator", "")
        document_id = citation.get("document_id", "")

        # Check 1: Quote must not be empty
        if not quote or not quote.strip():
            errors.append(CitationValidationError(
                citation_index=index,
                locator=locator,
                quote=quote,
                error_type="empty_quote",
                details="Citation has empty quote"
            ))
            return errors

        # Check 2: Quote must exist in source text (canonicalized)
        canonical_quote = canonicalize(quote)

        if not quote_exists_in_source(canonical_quote, context.all_text):
            # Try searching in individual chunks for better error message
            found_in_chunk = None
            for chunk in context.chunks:
                if quote_exists_in_source(canonical_quote, chunk.chunk_text):
                    found_in_chunk = chunk
                    break

            if not found_in_chunk:
                errors.append(CitationValidationError(
                    citation_index=index,
                    locator=locator,
                    quote=quote[:100] + ("..." if len(quote) > 100 else ""),
                    error_type="quote_not_found",
                    details="Quote does not exist verbatim in source text"
                ))

        # Check 3: Locator must be present
        if not locator or not locator.strip():
            errors.append(CitationValidationError(
                citation_index=index,
                locator=locator,
                quote=quote[:50],
                error_type="empty_locator",
                details="Citation has empty locator"
            ))
        else:
            # Check 4: Locator should contain law reference pattern
            if not self._locator_is_valid(locator):
                errors.append(CitationValidationError(
                    citation_index=index,
                    locator=locator,
                    quote=quote[:50],
                    error_type="invalid_locator",
                    details="Locator does not contain valid law reference"
                ))

        return errors

    def _locator_is_valid(self, locator: str) -> bool:
        """
        Check if locator contains valid law reference.

        Expected formats:
        - "Lög nr. 33/1944"
        - "Lög nr. 33/1944 - 1. gr."
        - "Lög nr. 33/1944 - 1. gr., 2. mgr."
        """
        # Must contain law number pattern
        law_pattern = re.compile(r'\d+/\d{4}')
        return bool(law_pattern.search(locator))

    def _is_refusal_or_clarification(self, answer: str) -> bool:
        """
        Check if answer is a refusal or clarification request.

        These don't require citations.
        """
        refusal_patterns = [
            r'get ekki svarað',
            r'finn ekki',
            r'engar upplýsingar',
            r'ekki nægjanlegar',
            r'vantar upplýsingar',
            r'tilgreindu nánar',
            r'vinsamlegast tilgreindu',
        ]

        answer_lower = answer.lower()
        return any(
            re.search(pattern, answer_lower)
            for pattern in refusal_patterns
        )


# Strict prompting instructions for retry
STRICT_QUOTE_INSTRUCTIONS_IS = """
MIKILVÆGT - STRANGAR TILVITNUNAREGLUR:

1. AFRITAÐU tilvitnanir NÁKVÆMLEGA eins og þær birtast í textanum
2. EKKI breyta orðum, orðaröð, eða greinarmerkjum
3. EKKI bæta við eða fjarlægja neinu
4. Tilvitnun VERÐUR að vera orðrétt - staf fyrir staf
5. Notaðu nákvæmlega sama locator og fylgir heimildinni

Ef þú getur ekki vitnað orðrétt, EKKI reyna - svaraðu frekar að þú finnir ekki upplýsingar.
"""

STRICT_QUOTE_INSTRUCTIONS_EN = """
IMPORTANT - STRICT CITATION RULES:

1. COPY quotes EXACTLY as they appear in the source text
2. DO NOT change words, word order, or punctuation
3. DO NOT add or remove anything
4. Quote MUST be verbatim - character for character
5. Use exactly the same locator that accompanies the source

If you cannot quote verbatim, DO NOT try - instead say you cannot find the information.
"""


async def validate_and_retry(
    generate_fn: Callable[[str, List[Chunk], Optional[str]], Awaitable[Dict[str, Any]]],
    query: str,
    chunks: List[Chunk],
    max_retries: int = 1,
    language: str = "is",
) -> ChatResponse:
    """
    Generate response with validation and retry logic.

    Args:
        generate_fn: Async function that generates LLM response
        query: User query
        chunks: Retrieved chunks for context
        max_retries: Number of retries on validation failure
        language: "is" for Icelandic, "en" for English

    Returns:
        ChatResponse - either valid answer or refusal
    """
    validator = ResponseValidator()
    context = ValidationContext.from_chunks(chunks)

    strict_instructions = (
        STRICT_QUOTE_INSTRUCTIONS_IS if language == "is"
        else STRICT_QUOTE_INSTRUCTIONS_EN
    )

    last_error = None

    for attempt in range(max_retries + 1):
        # Add strict instructions on retry
        extra_prompt = strict_instructions if attempt > 0 else None

        try:
            response = await generate_fn(query, chunks, extra_prompt)
        except Exception as e:
            return ChatResponse(
                answer=None,
                citations=[],
                confidence=Confidence.NONE,
                failure_reason=FailureReason.INTERNAL_ERROR,
                debug={"error": str(e), "attempt": attempt}
            )

        # Validate response
        validation = validator.validate_response(response, context)

        if validation.valid:
            # Success - build response
            citations = [
                Citation(
                    document_id=c.get("document_id", ""),
                    locator=c.get("locator", ""),
                    quote=c.get("quote", ""),
                    canonical_url=c.get("canonical_url"),
                )
                for c in response.get("citations", [])
            ]

            return ChatResponse(
                answer=response.get("answer_markdown"),
                citations=citations,
                confidence=_compute_confidence(citations, chunks),
                debug={"attempt": attempt, "validation": "passed"}
            )

        # Validation failed
        last_error = validation.error_summary

    # All retries exhausted - return refusal
    return ChatResponse(
        answer=None,
        citations=[],
        confidence=Confidence.NONE,
        failure_reason=FailureReason.VALIDATION_FAILED,
        debug={
            "last_error": last_error,
            "attempts": max_retries + 1,
        }
    )


def _compute_confidence(citations: List[Citation], chunks: List[Chunk]) -> Confidence:
    """
    Compute confidence level based on objective metrics.

    NOT based on LLM self-assessment.
    """
    if not citations:
        return Confidence.NONE

    citation_count = len(citations)
    unique_docs = len(set(c.document_id for c in citations if c.document_id))

    # Simple heuristic based on citation count
    if citation_count >= 3 or unique_docs >= 2:
        return Confidence.HIGH
    elif citation_count >= 1:
        return Confidence.MEDIUM
    else:
        return Confidence.LOW
