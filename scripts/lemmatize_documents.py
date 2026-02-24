#!/usr/bin/env python3
"""
lemmatize_documents.py — Fill documents.search_lemmatized using Greynir lemmatisation.

Reads every document from Supabase, lemmatises the concatenation of its
text + law_reference + article_locator (matching the original search_tsv
input), and writes the result to both plain columns added by migration 008:

  search_lemmatized      TEXT     — base-form text (set by this script)
  search_tsv_lemmatized  TSVECTOR — to_tsvector('simple', search_lemmatized)
                                    (computed server-side in the same UPDATE)

After all rows are written, creates a GIN index CONCURRENTLY on
search_tsv_lemmatized (must run outside any transaction).

match_documents_hybrid uses search_tsv_lemmatized after migration 009 is applied.

Prerequisites:
    Migration 008 applied (search_lemmatized column exists).

Usage:
    python scripts/lemmatize_documents.py
    python scripts/lemmatize_documents.py --batch 200
    python scripts/lemmatize_documents.py --dry-run
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from dotenv import load_dotenv

from app.services.lemmatize import get_lemmatized_text

FETCH_QUERY = """
    SELECT id, "text", law_reference, article_locator
    FROM   documents
    ORDER  BY id
"""

UPDATE_QUERY = """
    UPDATE documents
    SET    search_lemmatized     = $1,
           search_tsv_lemmatized = to_tsvector('simple', $1)
    WHERE  id = $2
"""

CREATE_INDEX_SQL = """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_search_tsv_lemmatized
    ON public.documents USING GIN (search_tsv_lemmatized)
"""


def build_source_text(row: asyncpg.Record) -> str:
    """Concatenate the same fields used to build search_tsv."""
    return " ".join(filter(None, [
        row["text"]            or "",
        row["law_reference"]   or "",
        row["article_locator"] or "",
    ]))


async def run(batch_size: int, dry_run: bool) -> None:
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set in .env", file=sys.stderr)
        sys.exit(1)

    # DIRECT_URL bypasses pgbouncer — required for CREATE INDEX CONCURRENTLY.
    # Falls back to DATABASE_URL if not set (works when DATABASE_URL is a
    # direct connection, e.g. Supabase session pooler or raw postgres://).
    direct_url = os.getenv("DIRECT_URL") or database_url

    print("Connecting to database...")
    conn = await asyncpg.connect(database_url, statement_cache_size=0)

    try:
        # ── 1. Fetch all documents ─────────────────────────────────────────
        print("Fetching documents...", end=" ", flush=True)
        rows = await conn.fetch(FETCH_QUERY)
        total = len(rows)
        print(f"{total} rows")

        if total == 0:
            print("No documents found. Nothing to do.")
            return

        # ── 2. Lemmatise (CPU-bound, no async needed) ──────────────────────
        print(f"Lemmatising {total} documents...")
        t0 = time.perf_counter()
        updates: list[tuple[str, str]] = []

        for i, row in enumerate(rows, 1):
            source = build_source_text(row)
            lemmatized = get_lemmatized_text(source)
            updates.append((lemmatized, row["id"]))

            if i % 100 == 0 or i == total:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                print(
                    f"  {i:>4}/{total}  "
                    f"{elapsed:>5.1f}s elapsed  "
                    f"~{remaining:.0f}s remaining  "
                    f"({rate:.0f} docs/s)"
                )

        elapsed_total = time.perf_counter() - t0
        print(f"Lemmatisation complete in {elapsed_total:.1f}s")

        # ── 3. Persist (batched executemany) ──────────────────────────────
        if dry_run:
            print(f"\nDRY RUN — skipping database writes.")
            print(f"Sample (first 3):")
            for lemmatized, doc_id in updates[:3]:
                print(f"  [{doc_id[:30]}]  {lemmatized[:80]}")
            return

        print(f"\nWriting to database in batches of {batch_size}...")
        written = 0
        t1 = time.perf_counter()

        for start in range(0, len(updates), batch_size):
            chunk = updates[start : start + batch_size]
            await conn.executemany(UPDATE_QUERY, chunk)
            written += len(chunk)
            print(f"  {written}/{total} written", end="\r", flush=True)

        print(f"\nDone. {written} rows updated in {time.perf_counter() - t1:.1f}s")

        # ── 4. GIN index (must run outside a transaction) ─────────────────
        # CREATE INDEX CONCURRENTLY requires a direct Postgres connection,
        # not a pgbouncer transaction-pooler connection.
        print("\nCreating GIN index CONCURRENTLY on search_tsv_lemmatized ...")
        print("  (this may take a minute on large tables)")
        t2 = time.perf_counter()
        if direct_url != database_url:
            print("  Using DIRECT_URL for index creation (bypasses pgbouncer)")
            idx_conn = await asyncpg.connect(direct_url, statement_cache_size=0)
            try:
                await idx_conn.execute(CREATE_INDEX_SQL)
            finally:
                await idx_conn.close()
        else:
            await conn.execute(CREATE_INDEX_SQL)
        print(f"  Index created in {time.perf_counter() - t2:.1f}s")
        print()
        print("Next step: apply migration 009 via run_migrations.py")
        print("  python scripts/run_migrations.py")

    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill documents.search_lemmatized using Greynir lemmatisation."
    )
    parser.add_argument(
        "--batch", type=int, default=100, metavar="N",
        help="Rows per executemany call (default: 100)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Lemmatise but do not write to the database"
    )
    args = parser.parse_args()

    asyncio.run(run(args.batch, args.dry_run))


if __name__ == "__main__":
    main()
