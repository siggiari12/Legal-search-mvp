#!/usr/bin/env python3
"""
parse_all.py — Bulk SGML-to-JSON build step for Lagasafn.

Reads every .sgml file in an input directory, parses it with
app/ingestion/parser.parse_file, and writes the result as UTF-8 JSON
to the output directory.  A single-file failure never stops the run;
failures are collected and reported at the end.

Usage:
    python scripts/parse_all.py --input data/Raw/lagasafn_156b_sgml

Common flags:
    --input  DIR   Directory containing .sgml files          (required)
    --output DIR   Where to write .json files                (default: data/processed/laws)
    --limit  N     Stop after N files (useful for smoke tests)
    --force        Re-parse even when the output file already exists
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.parser import parse_file


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bulk-parse Lagasafn SGML files to JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", required=True, metavar="DIR",
        help="Directory containing .sgml files",
    )
    p.add_argument(
        "--output", default="data/processed/laws", metavar="DIR",
        help="Output directory for .json files (default: data/processed/laws)",
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Parse at most N files then stop",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-parse files even if the output already exists",
    )
    return p


# ── Core ──────────────────────────────────────────────────────────────────────

def run(
    input_dir: Path,
    output_dir: Path,
    limit: int | None,
    force: bool,
) -> int:
    """
    Parse all .sgml files in input_dir and write JSON to output_dir.
    Returns exit code (0 = all succeeded, 1 = at least one failure).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sgml_files = sorted(input_dir.glob("*.sgml"))
    if not sgml_files:
        print(f"ERROR: no .sgml files found in {input_dir}", file=sys.stderr)
        return 1

    if limit is not None:
        sgml_files = sgml_files[:limit]

    total       = len(sgml_files)
    succeeded   = 0
    skipped     = 0
    failures    = []        # list of (filename, error_message)
    total_articles   = 0
    total_paragraphs = 0

    start = time.monotonic()
    width = len(str(total))   # for zero-padded progress counter

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {total}" + (f"  (limit {limit})" if limit else ""))
    print()

    for i, sgml_path in enumerate(sgml_files, 1):
        out_path = output_dir / sgml_path.name.replace(".sgml", ".json")

        # Skip if output exists and --force not set
        if out_path.exists() and not force:
            skipped += 1
            print(f"  [{i:{width}}/{total}] SKIP  {sgml_path.name}")
            continue

        print(f"  [{i:{width}}/{total}] ..    {sgml_path.name}", end="", flush=True)

        try:
            result = parse_file(sgml_path)
        except Exception as exc:
            failures.append((sgml_path.name, str(exc)))
            print(f"\r  [{i:{width}}/{total}] FAIL  {sgml_path.name}  ({exc})")
            continue

        # Write JSON
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            failures.append((sgml_path.name, f"write error: {exc}"))
            print(f"\r  [{i:{width}}/{total}] FAIL  {sgml_path.name}  (write error: {exc})")
            continue

        n_articles  = result["article_count"]
        n_paragraphs = sum(len(a["paragraphs"]) for a in result["articles"])
        n_warnings  = len(result["parse_warnings"])

        total_articles   += n_articles
        total_paragraphs += n_paragraphs
        succeeded += 1

        warn_tag = f"  {n_warnings}w" if n_warnings else ""
        print(
            f"\r  [{i:{width}}/{total}] OK    {sgml_path.name}"
            f"  {n_articles}art {n_paragraphs}para{warn_tag}"
        )

    elapsed = time.monotonic() - start

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PARSE REPORT")
    print("=" * 60)
    print(f"  Total files : {total}")
    print(f"  Succeeded   : {succeeded}")
    print(f"  Skipped     : {skipped}  (output existed, no --force)")
    print(f"  Failed      : {len(failures)}")
    print(f"  Articles    : {total_articles}")
    print(f"  Paragraphs  : {total_paragraphs}")
    print(f"  Elapsed     : {elapsed:.1f}s")

    if failures:
        print()
        print("FAILURES:")
        for name, msg in failures:
            print(f"  {name}: {msg}")

    print("=" * 60)

    return 1 if failures else 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = build_arg_parser().parse_args()
    exit_code = run(
        input_dir  = Path(args.input),
        output_dir = Path(args.output),
        limit      = args.limit,
        force      = args.force,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
