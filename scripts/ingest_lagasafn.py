#!/usr/bin/env python3
"""
Lagasafn Ingestion Script

Parses SGML files from Lagasafn (Icelandic law collection) and loads them
into the database with vector embeddings.

Usage:
    # Ingest a single file
    python scripts/ingest_lagasafn.py path/to/law.sgml

    # Ingest a directory
    python scripts/ingest_lagasafn.py path/to/lagasafn/

    # Dry run (parse but don't insert)
    python scripts/ingest_lagasafn.py path/to/law.sgml --dry-run

    # Skip embedding generation (faster, for testing)
    python scripts/ingest_lagasafn.py path/to/law.sgml --skip-embeddings

Requires:
    - DATABASE_URL environment variable
    - OPENAI_API_KEY environment variable (unless --skip-embeddings)
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.db.connection import Database, close_db_pool
from app.ingestion.parser import SGMLParser
from app.services.embedding import EmbeddingService
from app.services.canonicalize import canonicalize


async def ingest_file(
    file_path: Path,
    db: Optional[Database],
    embedding_service: Optional[EmbeddingService],
    version_tag: str,
    dry_run: bool = False,
) -> dict:
    """
    Ingest a single SGML file.

    Returns:
        Dict with ingestion stats
    """
    print(f"\nProcessing: {file_path.name}")

    # Read file
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1")

    # Parse SGML
    parser = SGMLParser()
    try:
        parsed = parser.parse(content, str(file_path))
    except Exception as e:
        print(f"  ERROR parsing: {e}")
        return {"status": "error", "error": str(e)}

    print(f"  Title: {parsed.title}")
    print(f"  Law: {parsed.law_number}/{parsed.law_year}")
    print(f"  Articles: {len(parsed.articles)}")

    if dry_run:
        print("  [DRY RUN] Would insert document and chunks")
        return {
            "status": "dry_run",
            "law_reference": f"{parsed.law_number}/{parsed.law_year}",
            "articles": len(parsed.articles),
        }

    # Build full text from articles
    full_text_parts = []
    for a in parsed.articles:
        para_texts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in a.paragraphs]
        full_text_parts.append(f"{a.number}. gr.\n{chr(10).join(para_texts)}")
    full_text = "\n\n".join(full_text_parts)

    # Create document
    doc_id = await db.insert_document(
        title=parsed.title,
        law_number=parsed.law_number,
        law_year=parsed.law_year,
        full_text=canonicalize(full_text),
        full_text_normalized=canonicalize(full_text).lower(),
        version_tag=version_tag,
    )
    print(f"  Document ID: {doc_id}")

    # Create chunks
    chunk_ids = []
    chunk_texts = []

    for idx, article in enumerate(parsed.articles):
        # Combine paragraphs into chunk text
        para_texts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in article.paragraphs]
        chunk_text = "\n".join(para_texts)
        chunk_text_canonical = canonicalize(chunk_text)

        # Build locator
        locator = f"Lög nr. {parsed.law_number}/{parsed.law_year} - {article.number}. gr."

        chunk_id = await db.insert_chunk(
            document_id=doc_id,
            chunk_text=chunk_text_canonical,
            chunk_text_normalized=chunk_text_canonical.lower(),
            locator=locator,
            chunk_index=idx,
            article_number=article.number,
            law_number=parsed.law_number,
            law_year=parsed.law_year,
        )
        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk_text_canonical)

    print(f"  Chunks created: {len(chunk_ids)}")

    # Generate embeddings
    if embedding_service and chunk_texts:
        print("  Generating embeddings...")
        try:
            embeddings = await embedding_service.embed_batch(chunk_texts)
            print(f"  Embeddings generated: {len(embeddings)}")

            # Update chunks with embeddings
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                await db.update_chunk_embedding(chunk_id, embedding)
            print("  Embeddings stored")
        except Exception as e:
            print(f"  WARNING: Failed to generate embeddings: {e}")

    return {
        "status": "success",
        "law_reference": f"{parsed.law_number}/{parsed.law_year}",
        "document_id": doc_id,
        "chunks": len(chunk_ids),
    }


async def main(
    paths: List[str],
    dry_run: bool = False,
    skip_embeddings: bool = False,
):
    """Main ingestion function."""
    print("=" * 60)
    print("LAGASAFN INGESTION")
    print("=" * 60)

    # Collect files to process
    files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() in (".sgml", ".xml", ".sgm"):
            files.append(path)
        elif path.is_dir():
            files.extend(path.glob("**/*.sgml"))
            files.extend(path.glob("**/*.sgm"))
            files.extend(path.glob("**/*.xml"))
        else:
            print(f"WARNING: Skipping {path} (not a file or directory)")

    if not files:
        print("ERROR: No SGML files found")
        sys.exit(1)

    print(f"\nFiles to process: {len(files)}")
    print(f"Dry run: {dry_run}")
    print(f"Skip embeddings: {skip_embeddings}")

    # Version tag for this ingestion run
    version_tag = datetime.now().strftime("%Y-%m-%d-%H%M")
    print(f"Version tag: {version_tag}")

    # Initialize services
    db = None
    embedding_service = None

    if not dry_run:
        # Check DATABASE_URL
        if not os.environ.get("DATABASE_URL"):
            print("ERROR: DATABASE_URL not set")
            sys.exit(1)

        db = Database()
        await db.connect()
        print("\nConnected to database")

        if not skip_embeddings:
            if not os.environ.get("OPENAI_API_KEY"):
                print("WARNING: OPENAI_API_KEY not set - skipping embeddings")
            else:
                embedding_service = EmbeddingService()
                print("Embedding service initialized")

    # Process files
    results = []
    success_count = 0
    error_count = 0

    try:
        for file_path in files:
            result = await ingest_file(
                file_path=file_path,
                db=db,
                embedding_service=embedding_service,
                version_tag=version_tag,
                dry_run=dry_run,
            )
            results.append(result)

            if result["status"] == "success":
                success_count += 1
            elif result["status"] == "error":
                error_count += 1

    finally:
        if db:
            await close_db_pool()

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total files: {len(files)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    if dry_run:
        print("\n[DRY RUN] No data was inserted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Lagasafn SGML files into database"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="SGML file(s) or directory to ingest"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files but don't insert into database"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (faster for testing)"
    )

    args = parser.parse_args()

    asyncio.run(main(
        paths=args.paths,
        dry_run=args.dry_run,
        skip_embeddings=args.skip_embeddings,
    ))
