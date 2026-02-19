#!/usr/bin/env python3
"""
parse_one.py — Parse a single Lagasafn SGML file to JSON.

Usage (from project root):
    python scripts/parse_one.py data/Raw/lagasafn_156b_sgml/1944033.sgml

Output:
    data/processed/1944033.json   (UTF-8, pretty-printed)

The script reads the SGML file with cp1252 encoding, runs the parser, and
writes the result to data/processed/.  No database connection is made.
"""

import json
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.parser import parse_file


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_one.py <path-to-sgml-file>", file=sys.stderr)
        sys.exit(1)

    sgml_path = Path(sys.argv[1])
    if not sgml_path.exists():
        print(f"ERROR: file not found: {sgml_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing:  {sgml_path.name}")

    try:
        result = parse_file(sgml_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Preserve multi-part suffixes:  1281000.400.sgml → 1281000.400.json
    out_name = sgml_path.name.replace(".sgml", ".json")
    out_path = out_dir / out_name

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    # Print a human-readable summary
    for w in result["parse_warnings"]:
        print(f"  WARN: {w}")
    print(f"  Title:    {result['title']}")
    print(f"  Ref:      {result['law_reference']}")
    print(f"  Articles: {result['article_count']}")
    print(f"  Written:  {out_path}")


if __name__ == "__main__":
    main()
