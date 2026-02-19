#!/usr/bin/env python3
"""
build_chunks.py — Article-level chunking for Lagasafn law corpus.

Reads every parsed law JSON file from an input directory and writes one
JSONL record per article to a single output file.  Ellipsis-only articles
(entirely repealed text, represented as '…') are silently skipped.

Usage:
    python scripts/build_chunks.py
    python scripts/build_chunks.py --limit 5
    python scripts/build_chunks.py --output path/to/output.jsonl

Output record fields:
    chunk_id           deterministic key: "{law_reference}::{article_locator}"
    source             "law"
    law_reference      e.g. "33/1944"
    law_title          title with <fnn> footnote markers stripped
    article_locator    canonical locator string, e.g. "Lög nr. 33/1944 - 1. gr."
    article_number_int leading integer of article number, or null
    chapter            chapter label if present, else null
    paragraph_count    number of paragraphs in the article
    text               substantive paragraph texts joined with "\\n\\n"
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

# Matches <fnn>…</fnn> and <FNN>…</FNN> (footnote ref tags that bleed into titles)
_FNN_RE = re.compile(r"<[Ff][Nn][Nn][^>]*>.*?</[Ff][Nn][Nn]>", re.DOTALL)


def _strip_fnn(text: str) -> str:
    """Remove <fnn>…</fnn> footnote markers from a metadata string."""
    return _FNN_RE.sub("", text).strip()


def _is_content_empty(text: str) -> bool:
    """
    Return True when text carries no real content — i.e. it contains no
    Unicode letter or digit.  The ellipsis character '…' (U+2026) and plain
    dots, spaces, and punctuation do not count as content.
    """
    return not re.search(r"[^\W_]", text, re.UNICODE)


def _article_number_int(number_str):
    """
    Extract the leading integer from an article number string, or None.

    Handles falsy input (None, empty string) by returning None immediately.
    Strips leading whitespace and bracket characters before matching,
    so '[55a]' -> 55, ' 14' -> 14, 'III' -> None.
    """
    if not number_str:
        return None
    cleaned = str(number_str).lstrip(" \t[")
    m = re.match(r"\d+", cleaned)
    return int(m.group()) if m else None


# ── Core ──────────────────────────────────────────────────────────────────────

def build_chunks(input_dir: Path, output_path: Path, limit: int | None) -> None:
    """
    Iterate over law JSON files, emit one chunk per substantive article.

    Args:
        input_dir:   Directory containing parsed law JSON files.
        output_path: Destination JSONL file (created/overwritten).
        limit:       If set, stop after processing this many law files.
    """
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: no .json files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    if limit is not None:
        json_files = json_files[:limit]

    total_laws                     = 0
    chunks_written                 = 0
    chunks_skipped_empty           = 0
    chunks_skipped_missing_locator = 0
    chunks_skipped_short_text      = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for law_path in json_files:
            law = json.loads(law_path.read_text(encoding="utf-8"))
            total_laws += 1

            law_ref = law.get("law_reference", "").strip()
            if not law_ref:
                # Skip entire law — no reference means no meaningful chunk_id
                chunks_skipped_missing_locator += len(law.get("articles", []))
                continue

            law_title = _strip_fnn(law.get("title", ""))

            for article in law.get("articles", []):
                # ── Resolve article locator ───────────────────────────────
                article_locator = article.get("locator")
                if not article_locator:
                    # Fallback: use raw article number as locator
                    article_locator = article.get("number")
                if not article_locator:
                    chunks_skipped_missing_locator += 1
                    continue

                # ── Build text: filter ellipsis-only paragraphs first ─────
                paragraphs = article.get("paragraphs", [])
                substantive = [
                    p["text"]
                    for p in paragraphs
                    if p.get("text", "").strip() and not _is_content_empty(p["text"])
                ]

                if not substantive:
                    chunks_skipped_empty += 1
                    continue

                text = "\n\n".join(substantive)

                # Drop suspiciously short text (< 10 chars)
                if len(text.strip()) < 10:
                    chunks_skipped_short_text += 1
                    continue

                # Use source_file stem as the chunk_id anchor — law_reference is
                # shared across multi-part laws (e.g. Jónsbók 1281000.400/401)
                # so it cannot guarantee uniqueness.  source_file is always unique.
                source_stem = law.get("source_file", law_path.name).replace(".sgml", "")
                chunk = {
                    "chunk_id":           f"{source_stem}::{article_locator}::{article.get('number', '')}",
                    "source":             "law",
                    "law_reference":      law_ref,
                    "law_title":          law_title,
                    "article_locator":    article_locator,
                    "article_number_int": _article_number_int(article.get("number")),
                    "chapter":            article.get("chapter"),
                    "paragraph_count":    len(paragraphs),
                    "text":               text,
                }
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunks_written += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total_skipped = (
        chunks_skipped_empty
        + chunks_skipped_missing_locator
        + chunks_skipped_short_text
    )

    print(f"Input           : {input_dir}  ({len(json_files)} files)")
    print(f"Output          : {output_path}")
    print(f"Laws processed  : {total_laws}")
    print(f"Chunks written  : {chunks_written}")
    print(f"Chunks skipped  : {total_skipped}")
    print(f"  ellipsis-only       : {chunks_skipped_empty}")
    print(f"  missing locator     : {chunks_skipped_missing_locator}")
    print(f"  text too short (<10): {chunks_skipped_short_text}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build article-level JSONL chunks from parsed law JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", default="data/Processed/laws", metavar="DIR",
        help="Directory of parsed law JSON files (default: data/Processed/laws)",
    )
    p.add_argument(
        "--output", default="data/Processed/chunks_laws.jsonl", metavar="FILE",
        help="Output JSONL file (default: data/Processed/chunks_laws.jsonl)",
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N law files then stop",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    build_chunks(
        input_dir=Path(args.input),
        output_path=Path(args.output),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
