"""
Tests for app.services.citation — build_context and validate_citations.

Coverage:
  build_context
    - chunk_id sourced from "chunk_id" field
    - chunk_id sourced from "id" field (Supabase RPC response shape)
    - chunk_id falls back to "source_N" when neither field present
    - chunk_id appears in rendered SOURCE block
    - context_chunks entries always have a "chunk_id" key
    - char budget: sources that exceed MAX_CONTEXT_CHARS are excluded

  validate_citations
    - passing case: correct chunk_id, law_reference, article_locator, quote
    - missing chunk_id field
    - chunk_id not in context
    - law_reference mismatch (exact)
    - article_locator mismatch (exact)
    - empty quote
    - quote not found in chunk text
    - quote with different whitespace passes (normalize_ws)
    - multiple errors reported independently

  normalize_ws
    - collapses spaces, tabs, newlines

  Regression: Q3 — Hvenær fyrnist krafa?
    Failure mode: model cited a quote that was a paraphrase / partial reword
    of the source text, not a verbatim substring.

  Regression: Q7 — Hvernig er hlutafélag stofnað?
    Failure mode: model cited 2/1995 - 5. gr. with a chunk_id that was not
    among the retrieved sources (it cited an adjacent article it never saw).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.services.citation import build_context, normalize_ws, validate_citations


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(
    text: str,
    law_reference: str = "1/2000",
    article_locator: str = "Lög nr. 1/2000 - 1. gr.",
    *,
    id: str | None = None,
    chunk_id: str | None = None,
) -> dict:
    chunk = {"text": text, "law_reference": law_reference, "article_locator": article_locator}
    if chunk_id is not None:
        chunk["chunk_id"] = chunk_id
    if id is not None:
        chunk["id"] = id
    return chunk


# ── normalize_ws ──────────────────────────────────────────────────────────────

class TestNormalizeWs:
    def test_collapses_multiple_spaces(self):
        assert normalize_ws("a  b") == "a b"

    def test_collapses_tabs(self):
        assert normalize_ws("a\tb") == "a b"

    def test_collapses_newlines(self):
        assert normalize_ws("a\nb") == "a b"

    def test_collapses_mixed(self):
        assert normalize_ws("a \t\n  b") == "a b"

    def test_strips_leading_trailing(self):
        assert normalize_ws("  hello  ") == "hello"

    def test_empty_string(self):
        assert normalize_ws("") == ""


# ── build_context ─────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_chunk_id_from_chunk_id_field(self):
        chunks = [make_chunk("text here", chunk_id="abc-123")]
        _, ctx_chunks = build_context(chunks)
        assert ctx_chunks[0]["chunk_id"] == "abc-123"

    def test_chunk_id_from_id_field(self):
        """Supabase RPC returns 'id', not 'chunk_id'."""
        chunks = [make_chunk("text here", id="doc-xyz")]
        _, ctx_chunks = build_context(chunks)
        assert ctx_chunks[0]["chunk_id"] == "doc-xyz"

    def test_chunk_id_prefers_chunk_id_over_id(self):
        chunks = [make_chunk("text here", chunk_id="cid-1", id="doc-1")]
        _, ctx_chunks = build_context(chunks)
        assert ctx_chunks[0]["chunk_id"] == "cid-1"

    def test_chunk_id_fallback_to_source_n(self):
        chunks = [make_chunk("text here")]  # no id, no chunk_id
        _, ctx_chunks = build_context(chunks)
        assert ctx_chunks[0]["chunk_id"] == "source_1"

    def test_fallback_index_increments(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        _, ctx_chunks = build_context(chunks)
        assert ctx_chunks[0]["chunk_id"] == "source_1"
        assert ctx_chunks[1]["chunk_id"] == "source_2"

    def test_chunk_id_appears_in_source_block(self):
        chunks = [make_chunk("some text", chunk_id="my-id")]
        context, _ = build_context(chunks)
        assert "chunk_id: my-id" in context

    def test_context_chunks_always_have_chunk_id(self):
        chunks = [
            make_chunk("text a", id="id-1"),
            make_chunk("text b"),          # fallback
            make_chunk("text c", chunk_id="explicit"),
        ]
        _, ctx_chunks = build_context(chunks)
        for c in ctx_chunks:
            assert "chunk_id" in c

    def test_law_reference_in_source_block(self):
        chunks = [make_chunk("text", law_reference="30/2004")]
        context, _ = build_context(chunks)
        assert "law_reference: 30/2004" in context

    def test_article_locator_in_source_block(self):
        chunks = [make_chunk("text", article_locator="Lög nr. 30/2004 - 52. gr.")]
        context, _ = build_context(chunks)
        assert "article_locator: Lög nr. 30/2004 - 52. gr." in context

    def test_empty_chunk_list(self):
        context, ctx_chunks = build_context([])
        assert context == ""
        assert ctx_chunks == []

    def test_char_budget_excludes_overflow(self):
        from app.services.citation import MAX_CONTEXT_CHARS, MAX_SOURCE_CHARS
        # Each chunk contributes ~MAX_SOURCE_CHARS + ~100 bytes of header overhead.
        # Create enough chunks to exceed MAX_CONTEXT_CHARS so the last one is cut off.
        n = (MAX_CONTEXT_CHARS // MAX_SOURCE_CHARS) + 3
        chunks = [make_chunk("x" * MAX_SOURCE_CHARS, id=f"fill-{i}") for i in range(n)]
        _, ctx_chunks = build_context(chunks)
        # Not all chunks fit — budget was exceeded before the last ones.
        assert len(ctx_chunks) < n


# ── validate_citations ────────────────────────────────────────────────────────

class TestValidateCitations:
    def _ctx(self, text="Starfsmanni skal segja upp störfum.", chunk_id="c1",
             law_ref="70/1996", locator="Lög nr. 70/1996 - 43. gr."):
        return [{"chunk_id": chunk_id, "text": text,
                 "law_reference": law_ref, "article_locator": locator}]

    def _cite(self, chunk_id="c1", law_ref="70/1996",
              locator="Lög nr. 70/1996 - 43. gr.",
              quote="Starfsmanni skal segja upp störfum."):
        return {"chunk_id": chunk_id, "law_reference": law_ref,
                "article_locator": locator, "quote": quote}

    # ── passing cases ─────────────────────────────────────────────────────────

    def test_valid_citation_passes(self):
        errors = validate_citations([self._cite()], self._ctx())
        assert errors == []

    def test_quote_with_different_whitespace_passes(self):
        ctx = self._ctx(text="Starfsmanni\nskal segja\nupp störfum.")
        cite = self._cite(quote="Starfsmanni skal segja upp störfum.")
        errors = validate_citations([cite], ctx)
        assert errors == []

    def test_partial_quote_passes(self):
        ctx = self._ctx(text="Starfsmanni skal segja upp störfum samkvæmt lögum.")
        cite = self._cite(quote="segja upp störfum")
        errors = validate_citations([cite], ctx)
        assert errors == []

    def test_no_citations_is_valid(self):
        errors = validate_citations([], self._ctx())
        assert errors == []

    # ── chunk_id failures ──────────────────────────────────────────────────────

    def test_missing_chunk_id_field(self):
        cite = {"law_reference": "70/1996",
                "article_locator": "Lög nr. 70/1996 - 43. gr.",
                "quote": "some text"}
        errors = validate_citations([cite], self._ctx())
        assert any("missing chunk_id" in e for e in errors)

    def test_empty_chunk_id(self):
        cite = self._cite(chunk_id="")
        errors = validate_citations([cite], self._ctx())
        assert any("missing chunk_id" in e for e in errors)

    def test_unknown_chunk_id(self):
        cite = self._cite(chunk_id="does-not-exist")
        errors = validate_citations([cite], self._ctx())
        assert any("not in context" in e for e in errors)

    # ── metadata mismatch failures ────────────────────────────────────────────

    def test_wrong_law_reference(self):
        cite = self._cite(law_ref="99/9999")
        errors = validate_citations([cite], self._ctx())
        assert any("law_reference" in e for e in errors)

    def test_wrong_article_locator(self):
        cite = self._cite(locator="Lög nr. 70/1996 - 1. gr.")
        errors = validate_citations([cite], self._ctx())
        assert any("article_locator" in e for e in errors)

    def test_both_metadata_errors_reported(self):
        cite = self._cite(law_ref="wrong", locator="also-wrong")
        errors = validate_citations([cite], self._ctx())
        law_errs = [e for e in errors if "law_reference" in e]
        loc_errs = [e for e in errors if "article_locator" in e]
        assert len(law_errs) >= 1
        assert len(loc_errs) >= 1

    # ── quote failures ────────────────────────────────────────────────────────

    def test_empty_quote(self):
        cite = self._cite(quote="")
        errors = validate_citations([cite], self._ctx())
        assert any("empty quote" in e for e in errors)

    def test_paraphrased_quote_fails(self):
        ctx = self._ctx(text="Starfsmanni skal segja upp störfum.")
        cite = self._cite(quote="Starfsmaður skal fá uppsögn.")  # paraphrase
        errors = validate_citations([cite], ctx)
        assert any("not found" in e for e in errors)

    def test_fabricated_quote_fails(self):
        ctx = self._ctx(text="Starfsmanni skal segja upp störfum.")
        cite = self._cite(quote="Þetta kemur hvergi fram í lögunum.")
        errors = validate_citations([cite], ctx)
        assert any("not found" in e for e in errors)

    # ── multiple citations ────────────────────────────────────────────────────

    def test_multiple_citations_errors_are_independent(self):
        ctx = [
            {"chunk_id": "c1", "text": "Fyrsta setning.",
             "law_reference": "1/2000", "article_locator": "Lög nr. 1/2000 - 1. gr."},
            {"chunk_id": "c2", "text": "Önnur setning.",
             "law_reference": "2/2001", "article_locator": "Lög nr. 2/2001 - 2. gr."},
        ]
        citations = [
            {"chunk_id": "c1", "law_reference": "1/2000",
             "article_locator": "Lög nr. 1/2000 - 1. gr.", "quote": "Fyrsta setning."},  # OK
            {"chunk_id": "c2", "law_reference": "2/2001",
             "article_locator": "Lög nr. 2/2001 - 2. gr.", "quote": "FABRICATED"},       # bad quote
        ]
        errors = validate_citations(citations, ctx)
        # Only cite 2 should fail
        cite1_errs = [e for e in errors if e.startswith("Cite 1")]
        cite2_errs = [e for e in errors if e.startswith("Cite 2")]
        assert cite1_errs == []
        assert len(cite2_errs) >= 1


# ── Regression: Q3 — Hvenær fyrnist krafa? ───────────────────────────────────

class TestRegressionQ3:
    """
    In the previous pipeline run (validate_out.txt), Q3 produced PROBLEMATIC:
      [ERROR] Cite 3: quote not found verbatim in any context source.
      [ERROR] Cite 5: quote not found verbatim in any context source.

    Root cause: the model paraphrased or slightly reworded the source text
    rather than copying it verbatim.  The new pipeline catches this via
    chunk_id-anchored, whitespace-normalised substring matching.
    """

    SOURCE_TEXT = (
        "Krafa um vátryggingarfjárhæð samkvæmt höfuðstólstryggingum "
        "fyrnist á tíu árum og aðrar kröfur um bætur á fjórum árum."
    )

    def _ctx(self):
        return [{"chunk_id": "30-2004-125", "text": self.SOURCE_TEXT,
                 "law_reference": "30/2004",
                 "article_locator": "Lög nr. 30/2004 - 125. gr."}]

    def test_verbatim_quote_passes(self):
        cite = {"chunk_id": "30-2004-125", "law_reference": "30/2004",
                "article_locator": "Lög nr. 30/2004 - 125. gr.",
                "quote": "fyrnist á tíu árum og aðrar kröfur um bætur á fjórum árum"}
        assert validate_citations([cite], self._ctx()) == []

    def test_paraphrased_quote_fails(self):
        # Simulates what the old pipeline accepted but shouldn't have:
        # a reworded version that isn't actually in the source.
        cite = {"chunk_id": "30-2004-125", "law_reference": "30/2004",
                "article_locator": "Lög nr. 30/2004 - 125. gr.",
                "quote": "Vátryggingarfjárhæðarkröfur fyrnast á tíu árum"}  # paraphrase
        errors = validate_citations([cite], self._ctx())
        assert any("not found" in e for e in errors)

    def test_wrong_chunk_id_fails_even_with_valid_quote(self):
        # Model cited an adjacent article it never saw.
        cite = {"chunk_id": "30-2004-052",  # not in context
                "law_reference": "30/2004",
                "article_locator": "Lög nr. 30/2004 - 52. gr.",
                "quote": "Krafa um bætur fyrnist á fjórum árum."}
        errors = validate_citations([cite], self._ctx())
        assert any("not in context" in e for e in errors)


# ── Regression: Q7 — Hvernig er hlutafélag stofnað? ─────────────────────────

class TestRegressionQ7:
    """
    In the previous pipeline run (validate_out.txt), Q7 produced PROBLEMATIC:
      [ERROR] Cite 1: ('2/1995','Lög nr. 2/1995 - 5. gr.') not in context.
      [ERROR] Cite 2: quote not found verbatim in any context source.

    Root cause: the top retrieved chunk was 2/1995 - 56. gr., but the model
    cited 2/1995 - 5. gr. (an adjacent article it never saw) and fabricated
    a plausible-sounding quote for it.

    With the new pipeline the chunk_id check catches this immediately:
    the chunk_id for 5. gr. will not be present in context.
    """

    RETRIEVED_TEXT = (
        "Hlutafé félagsins við stofnun þess skal vera 500 þúsund krónur að lágmarki."
    )

    def _ctx(self):
        # Only 56. gr. was actually retrieved — 5. gr. was NOT.
        return [{"chunk_id": "2-1995-56", "text": self.RETRIEVED_TEXT,
                 "law_reference": "2/1995",
                 "article_locator": "Lög nr. 2/1995 - 56. gr."}]

    def test_citation_of_retrieved_article_passes(self):
        cite = {"chunk_id": "2-1995-56", "law_reference": "2/1995",
                "article_locator": "Lög nr. 2/1995 - 56. gr.",
                "quote": "Hlutafé félagsins við stofnun þess"}
        assert validate_citations([cite], self._ctx()) == []

    def test_citation_of_unretrieved_article_fails(self):
        # 5. gr. was NOT retrieved — chunk_id for it is not in context.
        cite = {"chunk_id": "2-1995-05", "law_reference": "2/1995",
                "article_locator": "Lög nr. 2/1995 - 5. gr.",
                "quote": "Í stofnsamningi skal ávallt greina nöfn stofnenda."}
        errors = validate_citations([cite], self._ctx())
        assert any("not in context" in e for e in errors)

    def test_fabricated_quote_for_retrieved_article_fails(self):
        # chunk_id is correct but quote is invented.
        cite = {"chunk_id": "2-1995-56", "law_reference": "2/1995",
                "article_locator": "Lög nr. 2/1995 - 56. gr.",
                "quote": "Í stofnsamningi skal ávallt greina nöfn, kennitölu og heimilisföng stofnenda."}
        errors = validate_citations([cite], self._ctx())
        assert any("not found" in e for e in errors)
