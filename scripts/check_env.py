#!/usr/bin/env python3
"""
check_env.py — Verify required environment variables are set.

Loads .env via python-dotenv and checks that all keys needed for
DB ingestion are present.  Does not call any external API.

Usage:
    python scripts/check_env.py
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv is not installed.")
    print("       Run: pip install -r requirements.txt")
    sys.exit(1)

# Load .env from project root (one level up from scripts/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

REQUIRED = {
    "SUPABASE_URL":   "Supabase project URL",
    "SUPABASE_KEY":   "Supabase anon/public key",
    "OPENAI_API_KEY": "OpenAI API key",
}

missing = [var for var in REQUIRED if not os.getenv(var)]

if missing:
    print("ENVIRONMENT CHECK FAILED")
    print()
    for var in missing:
        print(f"  MISSING  {var}  ({REQUIRED[var]})")
    print()
    print("Copy .env.example to .env and paste your real keys.")
    sys.exit(1)

print("ENVIRONMENT CHECK PASSED")
print()
for var, description in REQUIRED.items():
    value = os.getenv(var)
    masked = value[:8] + "..." if len(value) > 8 else "***"
    print(f"  OK  {var:<20} {masked}  ({description})")
