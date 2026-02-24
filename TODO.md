# TODO – 2026-02-24

## Current System State

### Retrieval

- Hybrid retrieval active (vector + FTS weighted scoring)
- `assess_retrieval` now gates sufficiency using `vector_sim`
- Combined similarity is used for stats display only
- No more threshold deflation from FTS weighting
- Vector-only and hybrid both functioning
- FTS currently uses PostgreSQL `simple` dictionary
- **Observed issue:** FTS scores are `0.0000` for 7/8 natural-language Icelandic queries
- **Likely cause:** no stemming / morphological normalization (literal token matching only)

### Citation System

- `app/services/citation.py` created
- `normalize_ws`, `build_context`, `validate_citations` implemented
- `chunk_id` required in all citations
- Citation validation checks:
  - `chunk_id` present
  - `chunk_id` exists in context
  - `law_reference` exact match
  - `article_locator` exact match
  - quote non-empty
  - whitespace-normalised substring match
- 37/37 tests passing
- Regression cases Q3 and Q7 covered

### Validation Results (Hybrid after fix)

- GOOD: 4
- ACCEPTABLE: 3
- PROBLEMATIC: 1
- Declined: 0
- Citation pass: 7/8
- **Remaining issue (Q7):** retrieval did not surface correct article; model cited adjacent article content but grounding caught it

---

## Tomorrow – Investigation Focus

**Primary task:**
Investigate Icelandic FTS mismatch and decide architecture direction.

**Questions to resolve:**

- Should hybrid remain, or revert to vector-only?
- Should FTS weight be reduced (e.g., 0.1–0.15)?
- Should query normalization be introduced before FTS?
- Is proper Icelandic stemming feasible in Supabase/Postgres?
- Would FTS be better used only as exact-term reranker rather than weighted hybrid?
