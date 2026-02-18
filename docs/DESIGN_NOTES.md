# Design Notes: Legal Search MVP

This document describes the core components and design decisions for the Legal Search MVP.

---

## Engineering Doctrine

1. The system must never answer without verifiable evidence from the stored legal corpus.
2. Exact citations are mandatory; unverifiable quotes invalidate the entire response.
3. When validation fails, the system retries once with stricter constraints, then refuses.
4. Correct refusal is preferable to a partially correct answer.
5. All legal structure (law numbers, articles, paragraphs) must come from parsed source data, never inferred by the model.
6. Text normalization must be deterministic and consistent across ingestion, storage, retrieval, and validation.
7. Search must favor recall over confidence, using hybrid methods to avoid false negatives.
8. The model may summarize or rephrase, but claims require direct citations.
9. System failures must be explicit, auditable, and user-visible, not silent.
10. The MVP optimizes for trust and correctness first; usability and speed second.

---

## 1. Canonical Text Normalization

### The Rule

A **single canonicalization function** (`canonicalize()`) is applied at ALL text processing points:

```python
def canonicalize(text: str) -> str:
    # 1. Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)

    # 2. Replace all Unicode spaces with ASCII space (NBSP, etc.)
    for space_char in UNICODE_SPACES:
        text = text.replace(space_char, ' ')

    # 3. Collapse whitespace to single space
    text = re.sub(r'\s+', ' ', text)

    # 4. Strip leading/trailing
    text = text.strip()

    # NOTE: Case and punctuation are NOT changed
    return text
```

### Where It's Applied

| Stage | Function | Purpose |
|-------|----------|---------|
| **Ingestion** | `canonicalize(raw_text)` | Normalize before storage |
| **Storage** | `chunk.chunk_text` | DB stores canonical form |
| **Retrieval** | `canonicalize_for_search(query)` | Normalize search queries |
| **LLM Context** | `canonicalize(chunk.text)` | Context sent to LLM |
| **Validation** | `canonicalize(quote)`, `canonicalize(source)` | Compare quotes |

### What Is NOT Changed

- **Case**: "ABC" stays "ABC"
- **Punctuation**: "1. gr." stays "1. gr."
- **Icelandic characters**: þ, ð, æ, ö preserved exactly

---

## 2. Hybrid Search Strategy

### Why Hybrid?

Vector-only search has weaknesses:
- **Icelandic inflection**: "mannréttindi" may not match "mannréttindum"
- **Exact references**: "33/1944" needs exact matching
- **Legal terminology**: Domain-specific terms may have weak embeddings

### Search Flow

```
Query
  │
  ├──▶ Extract Law Reference (regex: "33/1944")
  │         │
  │    Found?──▶ Direct DB Lookup (highest priority)
  │         │
  │    Not Found
  │         │
  └──▶ Parallel Hybrid Search
            │
      ┌─────┴─────┐
      │           │
  Vector      Keyword
  Search      Search
      │           │
      └─────┬─────┘
            │
      Merge (UNION)
            │
      Deterministic
       Reranking
```

### Merge Strategy

```python
for chunk in results:
    combined_score = vector_score + keyword_score

    # Bonus if found by both methods
    if vector_score > 0 and keyword_score > 0:
        combined_score += 0.5  # Reinforcement bonus
```

**Key Decision**: Use UNION (not intersection) to maximize recall.

---

## 3. Validation + Retry Behavior

### Validation Rules

Every LLM response is validated before returning to user:

1. **Quote Verification**: Each `citation.quote` must exist VERBATIM in source text (after canonicalization)
2. **Locator Check**: Each `citation.locator` must contain valid law reference pattern (e.g., "33/1944")
3. **One Bad = All Bad**: If ANY citation fails, entire response is invalid

### Retry Flow

```
LLM Response
     │
     ▼
  Validate
     │
  ┌──┴──┐
  │     │
Pass   Fail
  │     │
  │     ▼
  │  Retry with
  │  stricter prompt
  │     │
  │     ▼
  │  Validate
  │     │
  │  ┌──┴──┐
  │  │     │
  │ Pass  Fail
  │  │     │
  └──┴──┐  │
        │  │
     Return │
     Answer │
           │
        Return
        Refusal
```

### Strict Retry Prompt (Icelandic)

```
MIKILVÆGT - STRANGAR TILVITNUNAREGLUR:
1. AFRITAÐU tilvitnanir NÁKVÆMLEGA eins og þær birtast í textanum
2. EKKI breyta orðum, orðaröð, eða greinarmerkjum
3. EKKI bæta við eða fjarlægja neinu
4. Tilvitnun VERÐUR að vera orðrétt - staf fyrir staf
```

---

## 4. Ingestion Pipeline

### SGML Parsing Strategy

Lagasafn SGML is not guaranteed well-formed XML. Our parser:

1. Uses BeautifulSoup with **lenient HTML parser** (not strict XML)
2. Extracts structure with fallbacks:
   - Try `<grein>` tags first
   - Fall back to `X. gr.` pattern matching
3. **Fails loudly** on structural errors
4. Keeps locators **conservative** - only include what's certain

### Locator Construction

Locators are built ONLY from parsed structure, never inferred:

```python
def build_locator(law_number, law_year, article_number=None, paragraph_number=None):
    base = f"Lög nr. {law_number}/{law_year}"
    if article_number:
        base += f" - {article_number}. gr."
        if paragraph_number:
            base += f", {paragraph_number}. mgr."
    return base
```

### Version Tag Strategy

Each ingestion run has a `version_tag`:

```python
version_tag = "2024-01-15"  # Date-based
# or
version_tag = "lagasafn-2024-01"  # Release-based
```

This enables:
- Tracking which ingestion produced a document
- Rolling back to previous versions
- Debugging citation failures

### Sanity Checks

Ingestion fails if:
- Document title is empty
- Law number/year is missing
- Fewer than N chunks per law
- Any chunk has empty text after canonicalization
- Any chunk missing locator

---

## 5. Chat Flow

### Request Processing

```
POST /chat { query }
     │
     ▼
Extract References (law_number, article_number)
     │
     ▼
Generate Query Embedding (if available)
     │
     ▼
Hybrid Search (vector + keyword)
     │
     ▼
Check: Enough evidence?
     │
  ┌──┴──┐
  │     │
 Yes    No ──▶ Return NO_RELEVANT_DATA
  │
  ▼
Check: Query ambiguous?
     │
  ┌──┴──┐
  │     │
 No    Yes ──▶ Return AMBIGUOUS_QUERY
  │
  ▼
Build LLM Context (chunks + locators)
     │
     ▼
Call LLM (strict JSON schema)
     │
     ▼
Validate Citations
     │
  ┌──┴──┐
  │     │
Pass   Fail ──▶ Retry (1x) ──▶ Fail ──▶ Return VALIDATION_FAILED
  │
  ▼
Return Answer + Citations
```

### Confidence Scoring

Confidence is computed **deterministically**, not by LLM:

```python
def compute_confidence(citations, chunks):
    if not citations:
        return "none"

    citation_count = len(citations)
    unique_docs = len(set(c.document_id for c in citations))

    if citation_count >= 3 or unique_docs >= 2:
        return "high"
    elif citation_count >= 1:
        return "medium"
    else:
        return "low"
```

---

## 6. Failure States

The system has distinct failure types with Icelandic user messages:

| Failure Type | Icelandic Message | When |
|-------------|-------------------|------|
| `ambiguous_query` | "Spurningin er of almenn" | Query too vague |
| `no_relevant_data` | "Engar heimildir fundust" | No matching chunks |
| `validation_failed` | "Ekki tókst að staðfesta svar" | Citations invalid |
| `rate_limited` | "Of margar fyrirspurnir" | Rate limit hit |
| `internal_error` | "Kerfisvilla" | System error |

---

## 7. Logging (Privacy-First)

Logs are necessary for debugging but must protect privacy:

**What IS logged:**
- Request ID
- Query length (not content)
- Query hash (SHA256 prefix)
- Chunk count retrieved
- Validation pass/fail
- Retry count
- Failure reason

**What is NOT logged:**
- Full query text
- IP addresses
- User identifiers
- Full response text

**Retention**: 7 days by default, configurable.

---

## 8. File Structure

```
legal-search-mvp/
├── app/
│   ├── api/
│   │   └── chat.py          # FastAPI endpoint
│   ├── ingestion/
│   │   ├── parser.py        # SGML parser
│   │   └── pipeline.py      # Ingestion flow
│   ├── models/
│   │   └── schemas.py       # Data models
│   ├── services/
│   │   ├── canonicalize.py  # THE normalization module
│   │   ├── chat.py          # Chat service
│   │   ├── search.py        # Hybrid search
│   │   └── validation.py    # Citation validation
│   └── main.py              # FastAPI app
├── tests/
│   ├── fixtures/            # Test SGML files
│   ├── test_canonicalize.py # Normalization tests
│   ├── test_chat_e2e.py     # End-to-end tests
│   ├── test_ingestion.py    # Ingestion tests
│   ├── test_search.py       # Search tests
│   └── test_validation.py   # Validation tests
└── docs/
    └── DESIGN_NOTES.md      # This file
```

---

## 9. Test Coverage

### Critical Tests (MUST PASS)

| Test | Purpose |
|------|---------|
| `test_fabricated_quote_fails` | Ensure fake quotes are rejected |
| `test_nbsp_difference_passes` | Ensure canonicalization handles NBSP |
| `test_one_bad_citation_fails_all` | Ensure strict validation |
| `test_retry_then_fail` | Ensure retry works then fails |
| `test_icelandic_characters_preserved` | Ensure þðæö survive |

### Running Tests

```bash
python3 tests/test_canonicalize.py
python3 tests/test_ingestion.py
python3 tests/test_validation.py
python3 tests/test_search.py
python3 tests/test_chat_e2e.py
```

---

## 10. Acceptance Criteria

Before MVP is complete:

- [x] `canonicalize()` handles Icelandic + NBSP + whitespace
- [x] Hybrid search combines vector + keyword
- [x] SGML parser extracts law_number/year/title/articles
- [x] Validation rejects fabricated quotes
- [x] Retry logic works (1 retry then refuse)
- [x] Chat endpoint returns answer only after validation
- [x] Ambiguous queries return clarification request
- [x] No evidence returns explicit refusal
- [x] All tests pass
