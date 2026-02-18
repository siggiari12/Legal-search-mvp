"""
Configuration Module

Centralizes all application configuration and environment variables.
Uses pydantic-settings for validation and type coercion.

Usage:
    from app.config import settings

    # Access settings
    db_url = settings.database_url
    rate_limit = settings.rate_limit_per_hour
"""

import os
from typing import Optional
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string (pooled for normal operations)"
    )
    database_url_direct: Optional[str] = Field(
        default=None,
        description="Direct PostgreSQL connection (for migrations)"
    )

    # AI Services
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for embeddings"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude"
    )

    # Supabase (optional, for SDK usage)
    supabase_url: Optional[str] = Field(
        default=None,
        description="Supabase project URL"
    )
    supabase_key: Optional[str] = Field(
        default=None,
        description="Supabase anon key"
    )

    # Application Settings
    rate_limit_per_hour: int = Field(
        default=20,
        description="Maximum queries per IP per hour"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # Search Settings
    search_top_k: int = Field(
        default=15,
        description="Default number of search results"
    )
    embedding_dimension: int = Field(
        default=1536,
        description="OpenAI embedding dimension"
    )

    # Validation Settings
    max_retries: int = Field(
        default=1,
        description="Max validation retries before refusing"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def has_database(self) -> bool:
        """Check if database is configured."""
        return self.database_url is not None

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI is configured."""
        return self.openai_api_key is not None

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic is configured."""
        return self.anthropic_api_key is not None


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience alias
settings = get_settings()
