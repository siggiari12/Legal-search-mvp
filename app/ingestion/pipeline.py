"""
Ingestion Pipeline for Legal Search MVP

This module handles the complete ingestion flow:
1. Parse SGML files using SGMLParser
2. Create Document records
3. Create Chunk records (with canonicalized text and locators)
4. Run sanity checks
5. Store in database

CRITICAL: All text is canonicalized before storage.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Callable, Dict, Any
import uuid

from app.services.canonicalize import canonicalize
from app.models.schemas import (
    Document,
    Chunk,
    ParsedLaw,
    ParsedArticle,
    IngestionResult,
    IngestionReport,
)
from app.ingestion.parser import SGMLParser, build_locator, SGMLParseError


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""
    min_chunks_per_law: int = 1  # Minimum chunks required per law
    max_chunk_length: int = 2000  # Max characters per chunk (split if larger)
    version_tag: Optional[str] = None  # Version tag for this ingestion run
    source: str = "Althingi"
    strict_validation: bool = True
    debug: bool = False


class IngestionError(Exception):
    """Raised when ingestion fails validation."""
    pass


class IngestionPipeline:
    """
    Pipeline for ingesting legal documents into the database.

    Usage:
        pipeline = IngestionPipeline(
            config=IngestionConfig(version_tag="2024-01-15"),
            store_document_fn=db.store_document,
            store_chunks_fn=db.store_chunks,
        )
        result = pipeline.ingest_sgml(sgml_content, "law_33_1944.sgml")
    """

    def __init__(
        self,
        config: IngestionConfig,
        store_document_fn: Optional[Callable] = None,
        store_chunks_fn: Optional[Callable] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Ingestion configuration
            store_document_fn: Async function to store document (or None for dry run)
            store_chunks_fn: Async function to store chunks (or None for dry run)
        """
        self.config = config
        self.store_document = store_document_fn
        self.store_chunks = store_chunks_fn
        self.parser = SGMLParser(strict=config.strict_validation)

    async def ingest_sgml(
        self,
        sgml_content: str,
        source_info: str = "unknown"
    ) -> IngestionResult:
        """
        Ingest a single SGML file.

        Args:
            sgml_content: Raw SGML string
            source_info: Identifier for logging (e.g., filename)

        Returns:
            IngestionResult with success/failure info
        """
        errors = []
        warnings = []

        # Step 1: Parse SGML
        try:
            parsed_law = self.parser.parse(sgml_content, source_info)
            warnings.extend(self.parser.warnings)
        except SGMLParseError as e:
            return IngestionResult(
                success=False,
                errors=[f"Parse error: {str(e)}"],
            )

        # Step 2: Create Document
        document = self._create_document(parsed_law)

        # Step 3: Create Chunks
        chunks = self._create_chunks(document, parsed_law)

        # Step 4: Run sanity checks
        try:
            self._validate_ingestion(document, chunks, source_info)
        except IngestionError as e:
            return IngestionResult(
                success=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings,
            )

        # Step 5: Store in database (if functions provided)
        if self.store_document and self.store_chunks:
            try:
                await self.store_document(document)
                await self.store_chunks(chunks)
            except Exception as e:
                return IngestionResult(
                    success=False,
                    document_id=document.id,
                    errors=[f"Storage error: {str(e)}"],
                    warnings=warnings,
                )

        return IngestionResult(
            success=True,
            document_id=document.id,
            chunk_count=len(chunks),
            warnings=warnings,
        )

    def _create_document(self, parsed_law: ParsedLaw) -> Document:
        """
        Create a Document from parsed law data.
        """
        # Canonicalize all text fields
        title = canonicalize(parsed_law.title)
        full_text = canonicalize(parsed_law.full_text)

        # Build canonical URL (best effort)
        canonical_url = self._build_canonical_url(
            parsed_law.law_number,
            parsed_law.law_year
        )

        return Document.create(
            title=title,
            law_number=parsed_law.law_number,
            law_year=parsed_law.law_year,
            full_text=full_text,
            source=self.config.source,
            document_type="law",
            version_tag=self.config.version_tag or datetime.now(timezone.utc).isoformat(),
            canonical_url=canonical_url,
            metadata_json=parsed_law.metadata,
        )

    def _create_chunks(
        self,
        document: Document,
        parsed_law: ParsedLaw
    ) -> List[Chunk]:
        """
        Create Chunks from parsed articles.

        Each article becomes one chunk (unless too long, then split by paragraph).
        """
        chunks = []

        for article in parsed_law.articles:
            article_chunks = self._chunk_article(
                document=document,
                article=article,
                law_number=parsed_law.law_number,
                law_year=parsed_law.law_year,
            )
            chunks.extend(article_chunks)

        return chunks

    def _chunk_article(
        self,
        document: Document,
        article: ParsedArticle,
        law_number: str,
        law_year: str,
    ) -> List[Chunk]:
        """
        Create chunk(s) from a single article.

        If article is short enough, create one chunk.
        If article is too long and has paragraphs, split by paragraph.
        If article is too long without paragraphs, split by character limit.
        """
        chunks = []

        # Canonicalize article text
        article_text = canonicalize(article.text)

        # If article fits in one chunk
        if len(article_text) <= self.config.max_chunk_length:
            locator = build_locator(law_number, law_year, article.number)

            chunks.append(Chunk.create(
                document_id=document.id,
                chunk_text=article_text,
                locator=locator,
                article_number=article.number,
                law_number=law_number,
                law_year=law_year,
            ))
            return chunks

        # Article too long - split by paragraphs if available
        if article.paragraphs:
            for para in article.paragraphs:
                para_text = canonicalize(para["text"])
                if not para_text:
                    continue

                locator = build_locator(
                    law_number,
                    law_year,
                    article.number,
                    para["number"]
                )

                chunks.append(Chunk.create(
                    document_id=document.id,
                    chunk_text=para_text,
                    locator=locator,
                    article_number=article.number,
                    paragraph_number=para["number"],
                    law_number=law_number,
                    law_year=law_year,
                ))
        else:
            # No paragraphs - split by character limit (conservative locator)
            for i, text_chunk in enumerate(self._split_text(article_text)):
                locator = build_locator(law_number, law_year, article.number)
                if i > 0:
                    # Add part number for subsequent chunks
                    locator += f" (hluti {i + 1})"

                chunks.append(Chunk.create(
                    document_id=document.id,
                    chunk_text=text_chunk,
                    locator=locator,
                    article_number=article.number,
                    law_number=law_number,
                    law_year=law_year,
                ))

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Split long text into smaller chunks.
        Tries to split at sentence boundaries.
        """
        max_len = self.config.max_chunk_length
        chunks = []
        current_chunk = ""

        # Split by sentences (roughly)
        sentences = text.replace('. ', '.|').split('|')

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_len:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _build_canonical_url(self, law_number: str, law_year: str) -> str:
        """
        Build canonical URL for a law.
        """
        # Althingi URL format (may need adjustment based on actual URLs)
        return f"https://www.althingi.is/lagas/{law_year}{law_number.zfill(3)}.html"

    def _validate_ingestion(
        self,
        document: Document,
        chunks: List[Chunk],
        source_info: str
    ) -> None:
        """
        Run sanity checks on ingested data.
        Raises IngestionError if validation fails.
        """
        errors = []

        # Check 1: Document has non-empty title
        if not document.title or document.title == canonicalize(""):
            errors.append("Document title is empty")

        # Check 2: Document has valid law number/year
        if not document.law_number or document.law_number == "0":
            errors.append("Invalid law number")
        if not document.law_year or document.law_year == "0000":
            errors.append("Invalid law year")

        # Check 3: Minimum chunk count
        if len(chunks) < self.config.min_chunks_per_law:
            errors.append(
                f"Too few chunks: {len(chunks)} < {self.config.min_chunks_per_law}"
            )

        # Check 4: No empty chunk text
        empty_chunks = [c for c in chunks if not c.chunk_text or not c.chunk_text.strip()]
        if empty_chunks:
            errors.append(f"{len(empty_chunks)} chunks have empty text")

        # Check 5: All chunks have locators
        missing_locators = [c for c in chunks if not c.locator]
        if missing_locators:
            errors.append(f"{len(missing_locators)} chunks missing locators")

        # Check 6: Locators contain law reference
        for chunk in chunks:
            if chunk.locator and document.law_reference not in chunk.locator:
                errors.append(
                    f"Chunk locator '{chunk.locator}' missing law reference '{document.law_reference}'"
                )
                break  # Just report first occurrence

        if errors and self.config.strict_validation:
            raise IngestionError(f"{source_info}: {'; '.join(errors)}")


async def ingest_laws_batch(
    sgml_files: List[Dict[str, str]],
    config: IngestionConfig,
    store_document_fn: Optional[Callable] = None,
    store_chunks_fn: Optional[Callable] = None,
) -> IngestionReport:
    """
    Batch ingest multiple SGML files.

    Args:
        sgml_files: List of {"content": str, "source_info": str}
        config: Ingestion configuration
        store_document_fn: Async function to store document
        store_chunks_fn: Async function to store chunks

    Returns:
        IngestionReport with summary statistics
    """
    pipeline = IngestionPipeline(
        config=config,
        store_document_fn=store_document_fn,
        store_chunks_fn=store_chunks_fn,
    )

    report = IngestionReport(
        total_laws=len(sgml_files),
        version_tag=config.version_tag,
    )

    for sgml_file in sgml_files:
        result = await pipeline.ingest_sgml(
            sgml_content=sgml_file["content"],
            source_info=sgml_file.get("source_info", "unknown"),
        )

        if result.success:
            report.successful += 1
            report.total_chunks += result.chunk_count
        else:
            report.failed += 1
            report.errors.append({
                "source": sgml_file.get("source_info"),
                "errors": result.errors,
            })

        report.warnings.extend(result.warnings)

    return report
