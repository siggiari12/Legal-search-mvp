"""
citation.py — Shared context-building and citation-validation utilities.

Used by scripts/answer_query.py and scripts/validate_pipeline.py.

build_context(chunks) → (context_str, context_chunks)
    Converts retrieved chunks into an LLM-readable SOURCE block string,
    returning the exact subset of chunks the model will see (after the
    char-budget cut).  Each SOURCE block includes a chunk_id field that
    the model is required to echo back in its citations.

validate_citations(citations, context_chunks) → list[str]
    Checks each citation's chunk_id against the context_chunks map,
    then verifies the quote as a whitespace-normalised substring of
    the stored (possibly truncated) chunk text.
"""

import re
from typing import Any

MAX_SOURCES       = 8
MAX_CONTEXT_CHARS = 12_000
MAX_SOURCE_CHARS  = 1_800
TRUNCATION_MARKER = "\n[...]\n"


def normalize_ws(s: str) -> str:
    """Collapse runs of whitespace (spaces, tabs, newlines) to a single space."""
    return re.sub(r"\s+", " ", s).strip()


def _truncate_source(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    result = ""
    for para in text.split("\n\n"):
        candidate = (result + "\n\n" + para).strip() if result else para
        if len(candidate) > max_chars:
            break
        result = candidate
    if len(result) >= max_chars * 0.6:
        return result
    half = (max_chars - len(TRUNCATION_MARKER)) // 2
    return text[:half] + TRUNCATION_MARKER + text[-half:]


def build_context(
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build the LLM context string from retrieved chunks.

    Each SOURCE block exposes a chunk_id line so the model can echo it
    back in citations, enabling per-chunk grounding validation.

    Returns:
        context_str    — formatted string to pass to the LLM
        context_chunks — subset of chunks actually included, each
                         guaranteed to have a "chunk_id" key for validation
    """
    parts:          list[str]            = []
    context_chunks: list[dict[str, Any]] = []
    total = 0

    for i, chunk in enumerate(chunks[:MAX_SOURCES], 1):
        chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"source_{i}"
        raw_text = (chunk.get("text") or "").strip()
        text     = _truncate_source(raw_text, MAX_SOURCE_CHARS)

        block = (
            f"[SOURCE {i}]\n"
            f"chunk_id: {chunk_id}\n"
            f"law_reference: {chunk.get('law_reference', '')}\n"
            f"article_locator: {chunk.get('article_locator', '')}\n"
            f"text: {text}"
        )

        if total + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 100:
                parts.append(block[:remaining])
                context_chunks.append({"chunk_id": chunk_id, **chunk})
            break

        parts.append(block)
        context_chunks.append({"chunk_id": chunk_id, **chunk})
        total += len(block)

    return "\n\n".join(parts), context_chunks


def validate_citations(
    citations:      list[dict[str, Any]],
    context_chunks: list[dict[str, Any]],
) -> list[str]:
    """
    Validate each citation against the exact chunks the model received.

    Checks (in order):
      1. chunk_id is present and exists in context_chunks.
      2. law_reference exactly matches the chunk's law_reference.
      3. article_locator exactly matches the chunk's article_locator.
      4. quote is non-empty.
      5. quote is a whitespace-normalised substring of that chunk's text.
         (normalisation collapses newlines/spaces so minor line-wrap
          differences in the source don't cause false failures)

    Exact string comparison is used for metadata fields (2–3).
    normalize_ws() is used only for quote matching (5).
    """
    chunk_map: dict[str, dict[str, Any]] = {
        c["chunk_id"]: c
        for c in context_chunks
    }

    errors: list[str] = []

    for idx, cit in enumerate(citations, 1):
        cid     = cit.get("chunk_id", "")
        law_ref = cit.get("law_reference", "")
        locator = cit.get("article_locator", "")
        quote   = cit.get("quote", "")

        if not cid:
            errors.append(f"Cite {idx}: missing chunk_id.")
            continue

        if cid not in chunk_map:
            errors.append(f"Cite {idx}: chunk_id {cid!r} not in context.")
            continue

        chunk = chunk_map[cid]

        if law_ref != chunk.get("law_reference", ""):
            errors.append(
                f"Cite {idx}: law_reference {law_ref!r} does not match "
                f"chunk {cid!r} ({chunk.get('law_reference', '')!r})."
            )

        if locator != chunk.get("article_locator", ""):
            errors.append(
                f"Cite {idx}: article_locator {locator!r} does not match "
                f"chunk {cid!r} ({chunk.get('article_locator', '')!r})."
            )

        if not quote:
            errors.append(f"Cite {idx}: empty quote.")
            continue

        if normalize_ws(quote) not in normalize_ws(chunk.get("text") or ""):
            errors.append(
                f"Cite {idx}: quote not found (whitespace-normalised) "
                f"in chunk {cid!r}."
            )

    return errors
