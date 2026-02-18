"""
Data Models for Legal Search MVP

These models define the structure for:
- Documents (laws)
- Chunks (searchable pieces)
- API requests/responses
- Validation results
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
import uuid


# =============================================================================
# Document & Chunk Models
# =============================================================================

@dataclass
class Document:
    """
    Represents a legal document (law) in the system.
    """
    id: str
    source: str  # e.g., "Althingi"
    document_type: str  # e.g., "law"
    title: str
    law_number: str  # e.g., "33"
    law_year: str  # e.g., "1944"
    full_text: str  # Canonicalized full text
    publication_date: Optional[str] = None
    version_tag: Optional[str] = None  # Ingestion timestamp or release tag
    canonical_url: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    @property
    def law_reference(self) -> str:
        """Returns law reference in format '33/1944'."""
        return f"{self.law_number}/{self.law_year}"

    @classmethod
    def create(
        cls,
        title: str,
        law_number: str,
        law_year: str,
        full_text: str,
        **kwargs
    ) -> "Document":
        """Factory method to create a new document with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            source=kwargs.get("source", "Althingi"),
            document_type=kwargs.get("document_type", "law"),
            title=title,
            law_number=law_number,
            law_year=law_year,
            full_text=full_text,
            publication_date=kwargs.get("publication_date"),
            version_tag=kwargs.get("version_tag"),
            canonical_url=kwargs.get("canonical_url"),
            metadata_json=kwargs.get("metadata_json"),
            created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
        )


@dataclass
class Chunk:
    """
    Represents a searchable chunk of legal text.
    Each chunk is a portion of a document (typically an article or paragraph).
    """
    id: str
    document_id: str
    chunk_text: str  # Canonicalized text
    locator: str  # e.g., "Lög nr. 33/1944 - 1. gr., 2. mgr."
    article_number: Optional[str] = None
    paragraph_number: Optional[str] = None
    law_number: Optional[str] = None
    law_year: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    # Search scores (populated during retrieval)
    vector_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0

    @classmethod
    def create(
        cls,
        document_id: str,
        chunk_text: str,
        locator: str,
        **kwargs
    ) -> "Chunk":
        """Factory method to create a new chunk with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_text=chunk_text,
            locator=locator,
            article_number=kwargs.get("article_number"),
            paragraph_number=kwargs.get("paragraph_number"),
            law_number=kwargs.get("law_number"),
            law_year=kwargs.get("law_year"),
            embedding=kwargs.get("embedding"),
            created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
        )


# =============================================================================
# Ingestion Models
# =============================================================================

@dataclass
class ParsedArticle:
    """Represents a parsed article from SGML."""
    number: str
    text: str
    paragraphs: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ParsedLaw:
    """Represents a fully parsed law from SGML."""
    law_number: str
    law_year: str
    title: str
    articles: List[ParsedArticle]
    full_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of ingesting a single law."""
    success: bool
    document_id: Optional[str] = None
    chunk_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class IngestionReport:
    """Summary report for a batch ingestion run."""
    total_laws: int = 0
    successful: int = 0
    failed: int = 0
    total_chunks: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    version_tag: Optional[str] = None


# =============================================================================
# API Models
# =============================================================================

class FailureReason(str, Enum):
    """Types of failures in chat responses."""
    AMBIGUOUS_QUERY = "ambiguous_query"
    NO_RELEVANT_DATA = "no_relevant_data"
    VALIDATION_FAILED = "validation_failed"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"


class Confidence(str, Enum):
    """Confidence levels (computed deterministically, not by LLM)."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class Citation:
    """A citation to a legal source."""
    document_id: str
    locator: str
    quote: str  # Must be verbatim from source
    canonical_url: Optional[str] = None


@dataclass
class ChatRequest:
    """Input for chat endpoint."""
    query: str
    filters: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Output from chat endpoint."""
    answer: Optional[str]  # Markdown answer, or None if refused
    citations: List[Citation]
    confidence: Confidence
    failure_reason: Optional[FailureReason] = None
    clarification_question: Optional[str] = None
    # Debug info (only in debug mode)
    debug: Optional[Dict[str, Any]] = None


# =============================================================================
# Validation Models
# =============================================================================

@dataclass
class CitationValidationError:
    """Details of a failed citation validation."""
    citation_index: int
    locator: str
    quote: str
    error_type: str  # "quote_not_found", "locator_mismatch", "document_not_found"
    details: str


@dataclass
class ValidationResult:
    """Result of validating an LLM response."""
    valid: bool
    errors: List[CitationValidationError] = field(default_factory=list)

    @property
    def error_summary(self) -> str:
        """Human-readable summary of errors."""
        if self.valid:
            return "All citations valid"
        summaries = [f"[{e.error_type}] {e.locator}: {e.details}" for e in self.errors]
        return "; ".join(summaries)
