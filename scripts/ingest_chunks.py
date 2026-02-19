#!/usr/bin/env python3
"""
ingest_chunks.py — Embed and upsert law article chunks into Supabase.

Reads article chunks from a JSONL file, generates text embeddings using
OpenAI text-embedding-3-small, and upserts each chunk into the Supabase
'documents' table.  Safe to re-run — upsert is idempotent on chunk_id.

Usage:
    python scripts/ingest_chunks.py
    python scripts/ingest_chunks.py --limit 50
    python scripts/ingest_chunks.py --input path/to/chunks.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from supabase import create_client
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE      = 100
TABLE_NAME      = "documents"
DEFAULT_INPUT   = "data/Processed/chunks_laws.jsonl"
MAX_RETRIES     = 6   # max wait: 1+2+4+8+16+32 = 63 s


# ── Environment ───────────────────────────────────────────────────────────────

def load_env() -> tuple[str, str]:
    """
    Load .env and validate required variables.
    Returns (supabase_url, supabase_key).
    Exits immediately if any variable is missing.
    """
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    required = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        print("ERROR: Missing required environment variables:")
        for key in missing:
            print(f"  {key}")
        print("\nCopy .env.example to .env and paste your real keys.")
        sys.exit(1)

    return os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")


# ── Chunk loading ─────────────────────────────────────────────────────────────

def count_lines(path: Path) -> int:
    """Count lines efficiently without loading the file into memory."""
    total = 0
    with open(path, "rb") as f:
        for _ in f:
            total += 1
    return total


def load_chunks(path: Path, limit: int | None) -> Generator[dict, None, None]:
    """Yield parsed chunk dicts from a JSONL file one at a time."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using text-embedding-3-small.
    Retries on RateLimitError with exponential backoff (up to MAX_RETRIES).
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            tqdm.write(
                f"  [rate limit] retrying in {wait}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)


# ── Insertion ─────────────────────────────────────────────────────────────────

def insert_batch(
    supabase,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """
    Upsert a batch of chunks with their embeddings into the documents table.
    Returns the number of rows upserted.
    """
    rows = [
        {
            "id":                 chunk["chunk_id"],
            "source":             chunk["source"],
            "law_reference":      chunk["law_reference"],
            "law_title":          chunk["law_title"],
            "article_locator":    chunk["article_locator"],
            "article_number_int": chunk.get("article_number_int"),
            "chapter":            chunk.get("chapter"),
            "text":               chunk["text"],
            "embedding":          embedding,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    supabase.table(TABLE_NAME).upsert(rows, on_conflict="id").execute()
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load env and initialise clients
    supabase_url, supabase_key = load_env()
    openai_client   = OpenAI()
    supabase_client = create_client(supabase_url, supabase_key)

    # Count total for progress bar (fast binary scan)
    total = count_lines(input_path)
    if args.limit is not None:
        total = min(total, args.limit)

    processed = 0
    inserted  = 0
    skipped   = 0

    batch_chunks: list[dict] = []

    print(f"Input  : {input_path}  ({total} chunks)")
    print(f"Table  : {TABLE_NAME}")
    print(f"Model  : {EMBEDDING_MODEL}")
    print()

    with tqdm(total=total, unit="chunk") as bar:

        def flush(batch: list[dict]) -> int:
            embeddings = embed_batch(openai_client, [c["text"] for c in batch])
            return insert_batch(supabase_client, batch, embeddings)

        for chunk in load_chunks(input_path, args.limit):
            bar.update(1)

            if not chunk.get("text", "").strip():
                skipped += 1
                continue

            batch_chunks.append(chunk)
            processed += 1

            if len(batch_chunks) >= BATCH_SIZE:
                inserted   += flush(batch_chunks)
                batch_chunks = []

        # Flush remaining partial batch
        if batch_chunks:
            inserted += flush(batch_chunks)

    print()
    print("=" * 40)
    print("INGEST SUMMARY")
    print("=" * 40)
    print(f"  Processed : {processed}")
    print(f"  Inserted  : {inserted}")
    print(f"  Skipped   : {skipped}  (empty text)")
    print("=" * 40)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Embed and upsert law article chunks into Supabase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", default=DEFAULT_INPUT, metavar="FILE",
        help=f"JSONL chunk file (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N chunks (for smoke tests)",
    )
    return p


if __name__ == "__main__":
    main()
