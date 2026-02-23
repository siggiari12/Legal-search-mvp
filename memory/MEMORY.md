# Legal Search MVP — Persistent Memory

## Project
Icelandic legal search engine (RAG) over Lagasafn (Althingi law corpus).
Stack: Python/FastAPI, Supabase/pgvector, Claude API, OpenAI embeddings.
PRD: legal-plan.prd  |  Working dir: C:\Users\Sigurður Ari\Desktop\Legal-search-mvp

## Key Architecture
- `app/ingestion/parser.py`   — SGML parser (core: `parse_sgml()` + `parse_file()`)
- `app/ingestion/pipeline.py` — DB ingestion pipeline (uses `SGMLParser` wrapper)
- `app/services/canonicalize.py` — SINGLE text normalization function; use everywhere
- `app/api/chat.py`           — API endpoints
- `app/services/production_chat.py` — hardened chat service
- `data/Raw/lagasafn_156b_sgml/` — raw SGML files (~700+ files)
- `data/processed/`           — parsed JSON output (one file per SGML)
- `scripts/parse_one.py`      — CLI: parse one SGML file to JSON

## SGML Encoding & Structure (Lagasafn 156b)
- Encoding: **Windows-1252 (cp1252)** — must open with `encoding='cp1252'`
- BeautifulSoup html.parser lowercases all tags
- Three structural patterns:
  A) Modern laws (post-~1900): `<law><head><lyr><lno><ldt><title>` + `<body><chapter><gr><gn><mgr>`
  B) Medieval (Jónsbók 1281): no `<gr>` tags, uses `<chapter><p>` blocks
  C) Very old / single-paragraph: just `<p>` in `<body>`
- Filename encodes law: `YYYYNNN.sgml` → year=YYYY, number=NNN (strip leading zeros)
- Multi-part ancient laws: `1281000.400.sgml`, `1281000.401.sgml`, etc.
- Tags to STRIP from text: `<fnpart>`, `<fnn>` (footnote refs), `<nr>` (paragraph numbering in fixture format)
- Inline footnote markers (`&#133;` → U+2026 ellipsis) come through fine after cp1252 decode

## JSON Schema (schema_version "1")
Fields: schema_version, source_file, law_number, law_year, law_reference,
        title, publication_date, articles[], parse_warnings[], article_count
Each article: number, locator, chapter, paragraphs[]
Each paragraph: number, text, locator
Locator format: "Lög nr. {N}/{YYYY} - {art}. gr.[, {para}. mgr.]"

## Test Fixtures
tests/fixtures/sample_law.sgml and sample_law_2.sgml use SYNTHETIC format:
`<log>`, `<nr>`, `<ar>`, `<heiti>`, `<grein>` — different from real SGML!
Parser handles both formats. 18/18 tests pass.

## Current Status (as of 2026-02-18)
- Phases 1–8 fully implemented in code
- Phase 9 (deployment) not started
- parser.py rewritten from real SGML inspection
- data/processed/1944033.json verified correct (80 articles, 33/1944)
- Next: embed+load data into Supabase DB
