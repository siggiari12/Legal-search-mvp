#!/usr/bin/env python3
"""
test_vector_search.py — Diagnostic vector similarity search against Supabase.

Requires a match_documents(query_embedding vector, match_count int) function
in the Supabase project.

Usage:
    python scripts/test_vector_search.py --query "hvenær má segja starfsmanni upp"
    python scripts/test_vector_search.py --query "..." --limit 10
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from supabase import create_client


def load_env() -> tuple[str, str]:
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

    required = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        print("ERROR: Missing required environment variables:")
        for key in missing:
            print(f"  {key}")
        sys.exit(1)

    return os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")


def embed_query(client: OpenAI, text: str) -> list[float]:
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    except OpenAIError as e:
        print(f"ERROR: OpenAI embedding failed: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic vector similarity search against Supabase documents table.",
    )
    parser.add_argument("--query", required=True, metavar="TEXT",
                        help="Query text to embed and search")
    parser.add_argument("--limit", type=int, default=5, metavar="N",
                        help="Number of results to return (default: 5)")
    args = parser.parse_args()

    supabase_url, supabase_key = load_env()
    openai_client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase_client = create_client(supabase_url, supabase_key)

    print(f"Query : {args.query}")
    print()

    embedding = embed_query(openai_client, args.query)

    if len(embedding) != 1536:
        print(f"ERROR: Unexpected embedding dimension: {len(embedding)} (expected 1536).")
        sys.exit(1)

    try:
        result = supabase_client.rpc(
            "match_documents",
            {"query_embedding": embedding, "match_count": args.limit},
        ).execute()
    except Exception as e:
        print(f"ERROR: match_documents RPC failed: {e}")
        print("Ensure the match_documents function exists in your Supabase project.")
        sys.exit(1)

    rows = result.data or []

    if not rows:
        print("No results returned.")
        return

    print(f"Top {len(rows)} results:")
    print("─" * 60)
    for i, row in enumerate(rows, 1):
        sim     = row.get("similarity")
        sim_str = f"{sim:.4f}" if sim is not None else "n/a"
        snippet = (row.get("text") or "")[:300].replace("\n", " ")
        print(f"[{i}] similarity : {sim_str}")
        print(f"    law_ref    : {row.get('law_reference', '')}")
        print(f"    locator    : {row.get('article_locator', '')}")
        print(f"    text       : {snippet}")
        print()


if __name__ == "__main__":
    main()
