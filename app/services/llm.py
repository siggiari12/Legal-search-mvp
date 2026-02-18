"""
LLM Service - Anthropic Claude Integration

Provides the Claude API integration for the Legal Search MVP.
Uses Claude 3.5 Sonnet for generating legal responses.

Usage:
    from app.services.llm import ClaudeLLM

    llm = ClaudeLLM()
    response = await llm.generate(system_prompt, user_prompt)
"""

import os
from typing import Optional

from anthropic import AsyncAnthropic


# Model configuration
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"  # Claude 3.5 Sonnet
MAX_TOKENS = 4096
TEMPERATURE = 0.0  # Deterministic for legal accuracy


class ClaudeLLM:
    """
    Claude LLM service for legal response generation.

    Uses low temperature for factual accuracy and deterministic responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize Claude LLM service.

        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
            model: Model ID to use.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> str:
        """
        Generate a response from Claude.

        Args:
            system_prompt: System instructions (in Icelandic for legal assistant)
            user_prompt: User's query with context
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            Claude's response text
        """
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        # Extract text from response
        return message.content[0].text

    async def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Callable interface for use as llm_fn in ChatService.

        Args:
            system_prompt: System instructions
            user_prompt: User prompt with context

        Returns:
            Response text
        """
        return await self.generate(system_prompt, user_prompt)


# Convenience function
async def generate_response(
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a response from Claude.

    Creates a new client each call - for repeated use,
    create a ClaudeLLM instance instead.

    Args:
        system_prompt: System instructions
        user_prompt: User prompt
        api_key: Optional Anthropic API key

    Returns:
        Response text
    """
    llm = ClaudeLLM(api_key=api_key)
    return await llm.generate(system_prompt, user_prompt)
