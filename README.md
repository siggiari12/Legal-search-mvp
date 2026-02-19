# Legal Search MVP

A closed-book AI chat system for Icelandic legal information (Lagasafn). The system answers questions only from stored legal corpus with verifiable citations.

## Features

- **Strict Validation**: All citations must exist verbatim in source text
- **Hybrid Search**: Vector (pgvector) + keyword search for better recall
- **Icelandic Support**: Full support for Icelandic characters (þ, ð, æ, ö)
- **Privacy-First Logging**: Query hashes only, no raw queries stored

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
DATABASE_URL=postgresql://...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Set Up Database (Supabase)

#### Option A: Supabase Cloud (Recommended)

1. Create a new project at [supabase.com](https://supabase.com)

2. Enable pgvector extension:
   - Go to **Database** > **Extensions**
   - Search for `vector` and enable it

3. Get your connection string:
   - Go to **Settings** > **Database**
   - Copy the **Connection string** (URI format)
   - Use the "Transaction" pooler URL for normal operations
   - Use the "Session" or direct URL for migrations

4. Set in `.env`:
   ```
   DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
   DATABASE_URL_DIRECT=postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
   ```

#### Option B: Local PostgreSQL

1. Install PostgreSQL 14+ with pgvector:
   ```bash
   # macOS with Homebrew
   brew install postgresql@14 pgvector

   # Ubuntu/Debian
   sudo apt install postgresql-14 postgresql-14-pgvector
   ```

2. Create database:
   ```bash
   createdb legal_search
   psql legal_search -c "CREATE EXTENSION vector;"
   ```

3. Set in `.env`:
   ```
   DATABASE_URL=postgresql://postgres:password@localhost:5432/legal_search
   ```

### 4. Run Migrations

```bash
python scripts/run_migrations.py
```

To preview changes without applying:
```bash
python scripts/run_migrations.py --dry-run
```

### 5. Run Tests

```bash
# Run all tests (without database)
python tests/test_canonicalize.py
python tests/test_search.py
python tests/test_validation.py
python tests/test_ingestion.py

# Run database smoke tests (requires DATABASE_URL)
python tests/test_db_smoke.py
```

### 6. Start the Server

```bash
uvicorn app.main:app --reload
```

## Project Structure

```
legal-search-mvp/
├── app/
│   ├── api/           # FastAPI endpoints
│   ├── db/            # Database connection
│   ├── ingestion/     # SGML parser and pipeline
│   ├── models/        # Pydantic schemas
│   ├── services/      # Core business logic
│   │   ├── canonicalize.py  # Text normalization
│   │   ├── chat.py          # Chat service
│   │   ├── search.py        # Hybrid search
│   │   └── validation.py    # Citation validation
│   └── main.py
├── migrations/        # SQL migrations
├── scripts/           # Utility scripts
├── tests/             # Test suite
└── docs/              # Documentation
```

## Engineering Principles

1. Never answer without verifiable evidence from stored corpus
2. Exact citations are mandatory; unverifiable quotes invalidate the response
3. When validation fails, retry once with stricter constraints, then refuse
4. Correct refusal is preferable to a partially correct answer
5. Legal structure must come from parsed source data, never inferred

See [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) for full design documentation.

## API Endpoints

### POST /api/chat

```json
{
  "query": "Hvað segir í stjórnarskránni um mannréttindi?"
}
```

Response:
```json
{
  "answer_markdown": "...",
  "citations": [
    {
      "document_id": "uuid",
      "locator": "Lög nr. 33/1944 - 1. gr.",
      "quote": "Exact quote from the law..."
    }
  ],
  "confidence": "high"
}
```

### GET /api/health

Returns database and service health status.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `DATABASE_URL_DIRECT` | No | Direct connection for migrations |
| `OPENAI_API_KEY` | Yes | For embeddings (text-embedding-3-small) |
| `ANTHROPIC_API_KEY` | Yes | For Claude chat responses |
| `RATE_LIMIT_PER_HOUR` | No | Default: 20 queries/hour per IP |
| `LOG_LEVEL` | No | Default: INFO |

## License

Private - All rights reserved.

### Handling Legacy Encoding (Windows-1252)

The official Alþingi SGML snapshot is encoded in Windows-1252 rather than UTF-8.

This initially caused corrupted Icelandic characters (þ, ð, æ) during parsing.
The ingestion pipeline now:

- Detects and reads SGML files using cp1252
- Converts content to clean UTF-8 internally
- Outputs normalized UTF-8 JSON

This ensures reliable text processing and prevents silent data corruption in downstream embeddings and retrieval.
