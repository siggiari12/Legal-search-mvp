#!/usr/bin/env python3
"""
Database Migration Runner

Runs SQL migrations against the PostgreSQL database.
Uses DATABASE_URL or DATABASE_URL_DIRECT environment variable.

Usage:
    python scripts/run_migrations.py
    python scripts/run_migrations.py --dry-run

Migrations are idempotent (safe to run multiple times).
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncpg
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


def get_database_url() -> str:
    """Get database URL, preferring direct connection for migrations."""
    url = os.environ.get("DATABASE_URL_DIRECT") or os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL or DATABASE_URL_DIRECT environment variable not set.")
        print("Set it to your PostgreSQL connection string.")
        print("Example: postgresql://postgres:password@localhost:5432/legal_search")
        sys.exit(1)
    return url


def get_migration_files() -> list[Path]:
    """Get sorted list of migration files."""
    if not MIGRATIONS_DIR.exists():
        print(f"ERROR: Migrations directory not found: {MIGRATIONS_DIR}")
        sys.exit(1)

    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        print(f"No migration files found in {MIGRATIONS_DIR}")
        sys.exit(0)

    return files


async def get_applied_migrations(conn: asyncpg.Connection) -> set[str]:
    """Get set of already-applied migration versions."""
    # Check if schema_migrations table exists
    exists = await conn.fetchval("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'schema_migrations'
        )
    """)

    if not exists:
        return set()

    rows = await conn.fetch("SELECT version FROM schema_migrations")
    return {row["version"] for row in rows}


async def run_migration(
    conn: asyncpg.Connection,
    migration_file: Path,
    dry_run: bool = False
) -> bool:
    """
    Run a single migration file.

    Returns True if migration was applied, False if skipped.
    """
    version = migration_file.stem  # e.g., "001_init"
    sql = migration_file.read_text()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {migration_file.name}")

    if dry_run:
        print(f"  Would execute {len(sql)} characters of SQL")
        return True

    try:
        # Execute the migration
        await conn.execute(sql)
        print(f"  Applied: {version}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        raise


async def main(dry_run: bool = False):
    """Run all pending migrations."""
    database_url = get_database_url()
    migration_files = get_migration_files()

    print("=" * 60)
    print("Legal Search MVP - Database Migrations")
    print("=" * 60)
    print(f"\nDatabase: {database_url[:50]}...")
    print(f"Migrations directory: {MIGRATIONS_DIR}")
    print(f"Found {len(migration_files)} migration file(s)")

    if dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***")

    # Connect to database
    try:
        conn = await asyncpg.connect(database_url)
    except Exception as e:
        print(f"\nERROR: Could not connect to database: {e}")
        sys.exit(1)

    try:
        # Get already-applied migrations
        applied = await get_applied_migrations(conn)
        if applied:
            print(f"\nAlready applied: {', '.join(sorted(applied))}")

        # Run pending migrations
        pending = [f for f in migration_files if f.stem not in applied]

        if not pending:
            print("\nNo pending migrations to apply.")
            return

        print(f"\nPending migrations: {len(pending)}")

        for migration_file in pending:
            await run_migration(conn, migration_file, dry_run)

        print("\n" + "=" * 60)
        if dry_run:
            print("DRY RUN COMPLETE - No changes made")
        else:
            print("ALL MIGRATIONS APPLIED SUCCESSFULLY")
        print("=" * 60)

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run database migrations for Legal Search MVP"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
