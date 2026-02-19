# Quality Validation Report — Lagasafn 156b Parsed Corpus

**Date:** 2026-02-19
**Corpus:** `data/Raw/lagasafn_156b_sgml/` → `data/Processed/laws/`
**Parser version:** `app/ingestion/parser.py` (commit `fbd6fc5`)

---

## 1. Corpus Metrics

| Metric | Value |
|---|---|
| SGML files in snapshot | 910 |
| JSON files produced | **795** |
| Empty-body tombstones (skipped) | 115 |
| Total articles | **18,300** |
| Total paragraphs | **46,089** |
| Average articles per law | 23.0 |
| Min articles (any law) | 1 |
| Max articles (any law) | 258 |

**Top 5 laws by article count**

| File | Law reference | Articles |
|---|---|---|
| `2022080.json` | 80/2022 | 258 |
| `1940019.json` | 19/1940 | 257 |
| `1985034.json` | 34/1985 | 239 |
| `2008088.json` | 88/2008 | 210 |
| `2005088.json` | 88/2005 | 194 |

---

## 2. Structural Validation

795 files were validated against the following criteria: non-empty title, non-empty law reference, non-empty articles list, unique article numbers within each law, ascending article order, non-empty paragraph text, and no leftover SGML tags in article body text.

| Check | Pass | Violations |
|---|---|---|
| Non-empty title | 795 | 0 |
| Law reference present | 795 | 0 |
| Articles list non-empty | 795 | 0 |
| Unique article numbers per law | 794 | **1** |
| Articles in ascending order | 795 | 0 |
| Non-empty paragraph text | 795 | 0 |
| No leftover SGML in article text | 795 | 0 |

**One violation — `2007100.json` (Almannatryggingar, 100/2007):**
The source SGML contains a `55. gr.` and a `55. gr. a.` (a lettered sub-article). The parser's digit-extraction regex strips the `a` suffix, producing two articles both numbered `55`. Both articles contain valid, substantive text. **Recommendation:** use the `locator` field as the primary key during DB ingestion rather than the raw `number` string.

---

## 3. Encoding Audit

All 795 files were scanned for common mojibake signatures.

| Pattern | Occurrences |
|---|---|
| `Ã` (Latin-1 double-encode) | 0 |
| `\ufffd` (Unicode replacement char) | 0 |
| `Â` (stray non-break prefix) | 0 |

Spot-check of `1944033.json` (Constitution, 33/1944) confirmed correct Unicode codepoints for Icelandic characters: Í (U+00CD), ð (U+00F0), þ (U+00FE), ó (U+00F3). The cp1252 → UTF-8 conversion pipeline is clean across the entire corpus.

---

## 4. Known Anomalies

These are expected corpus characteristics, not parser bugs. No corrective action is required before ingestion.

### 4.1 `<fnn>` footnote markers in titles (62 files)

The metadata extractor uses `get_text()` without tag filtering, so footnote reference tags such as `<FNN>1)</FNN>` bleed into the `title` field. Example: `"Lög um veiðigjald<fnn>1)</fnn>"`. The article body text is unaffected. Recommended handling: strip `<fnn>…</fnn>` patterns from title strings during DB load.

### 4.2 Leading bracket in titles (46 files)

Amended or repealed law titles sometimes begin with `[`, e.g. `[Lög um landhelgi…]`. This is Althingi's editorial convention and is preserved faithfully. No action required.

### 4.3 Ellipsis-only paragraphs (762 paragraphs, 254 laws)

Althingi marks repealed article text with the ellipsis character (`…`, U+2026). The parser faithfully captures these. They represent 1.7% of all paragraphs. Three laws (`1975011.201.json`, `1976020.200.json`, `1978003.201.json`) consist entirely of repealed articles. These files can be excluded from embeddings as they contain no searchable content.

---

## 5. Tombstone Verification

Of the 115 SGML files with empty `<BODY>` elements (catalog tombstones referencing the 1990 print edition only), **zero** produced a JSON output file. The parser correctly raises `ValueError: No articles extracted` for all of them.

---

## 6. Verdict

> **SAFE TO PROCEED TO DB INGESTION**

The corpus is structurally sound and correctly encoded. Two items require attention during the ingestion phase:

1. **Strip `<fnn>` tags from the `title` field** before writing to the database.
2. **Use `locator` as the article primary key**, not `number`, to avoid the single known duplicate (`55` / `55a` in `2007100.json`).
3. **Optionally exclude** the 3 fully-repealed laws and filter out ellipsis-only paragraphs from the embedding pipeline, as they add no retrieval value.
