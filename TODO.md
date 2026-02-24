# TODO – 2026-02-24 (updated)

## Current System State

### Retrieval

- Hybrid retrieval active (vector + FTS weighted scoring, weights 0.7/0.3)
- `assess_retrieval` gates sufficiency on `vector_sim` (not combined score)
- No threshold deflation from FTS weighting
- FTS uses `search_tsv_lemmatized` (migration 009) with `simple` dictionary
- Both query and document text are lemmatised via Greynir before FTS matching
- 17,911 documents lemmatised and written to `search_lemmatized` +
  `search_tsv_lemmatized`; GIN index created

### Citation System

- `app/services/citation.py`: `normalize_ws`, `build_context`, `validate_citations`
- `chunk_id` anchored: model must echo chunk_id from SOURCE block
- Validation checks: chunk_id present → in context → law_reference exact →
  article_locator exact → quote non-empty → whitespace-normalised substring match
- 37/37 tests passing

### Validation Results (Hybrid, query + document lemmatisation)

- GOOD: 6
- ACCEPTABLE: 1  (Q4 – hjúskapur; no relevant article retrieved)
- PROBLEMATIC: 1  (Q7 – hlutafélag; right article retrieved, quote paraphrased)
- Declined: 0
- Citation pass: 7/8

---

## Open Issues

### Q7 – Quote paraphrasing ("Hvernig er hlutafélag stofnað?")

- `2/1995 - 4. gr.` IS retrieved (ranks 6–8, outside top-5 display)
- Model cites correct chunk_id and metadata
- But quote does not verbatim match chunk text → `validate_citations` blocks it
- Likely cause: model constructs quote from memory / truncates with `...`
- **Candidate fixes:**
  - Strengthen SYSTEM_PROMPT: "copy the exact words character-for-character"
  - Inspect chunk text for `2/1995 - 4. gr.` to confirm quote mismatch vs chunk content

### Q4 – Retrieval coverage ("Hvernig er hjúskapur stofnaður?")

- No relevant article surfaced; answer fell back to general knowledge
- Confidence LOW, no citations produced (ACCEPTABLE, not PROBLEMATIC)
- **Candidate fixes:**
  - Check whether the relevant article (hjúskaparlög) is in the DB at all
  - Try `--k 12` to widen retrieval net
  - Check embeddings for hjúskapur-related chunks

---

## Next Steps (priority order)

1. **Harden verbatim-quote instruction in SYSTEM_PROMPT** (Q7 fix)
2. **Inspect `2/1995 - 4. gr.` chunk text** to confirm the paraphrase gap
3. **Investigate Q4 retrieval** — query DB for hjúskapur-related documents
4. **Re-run validation** after prompt hardening to measure improvement
