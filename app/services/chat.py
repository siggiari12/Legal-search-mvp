"""
Chat Service for Legal Search MVP

This module handles the chat/query flow:
1. Parse query for references
2. Retrieve evidence via hybrid search
3. Check if sufficient evidence exists
4. Build LLM context from chunks
5. Call LLM with strict JSON schema
6. Validate citations
7. Return answer or refusal

CRITICAL: No answer is returned unless all citations validate.
"""

import json
from typing import List, Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass

from app.services.canonicalize import canonicalize
from app.services.search import (
    Chunk,
    HybridSearcher,
    SearchType,
    extract_law_reference,
    extract_article_reference,
)
from app.services.validation import (
    ResponseValidator,
    ValidationContext,
    validate_and_retry,
    STRICT_QUOTE_INSTRUCTIONS_IS,
)
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    Confidence,
    FailureReason,
)


# System prompt for the LLM (Icelandic)
SYSTEM_PROMPT_IS = """Þú ert lögfræðilegur aðstoðarmaður sem svarar spurningum um íslensk lög.

STRANGAR REGLUR:
1. Svaraðu EINGÖNGU út frá heimildunum sem fylgja (context)
2. Ef þú finnur ekki nægjanlegar upplýsingar, neita að svara
3. HVER fullyrðing verður að hafa tilvitnun
4. Tilvitnanir verða að vera ORÐRÉTTAR - afritaðu nákvæmlega úr textanum
5. Giska ALDREI á lagagreinar, dagsetningar, eða lagaleg áhrif
6. Notaðu locator sem fylgir heimildinni í citation

SVARSNIÐ (JSON):
{
  "answer_markdown": "Svarið hér með [1] tilvísunum...",
  "citations": [
    {
      "document_id": "...",
      "locator": "Lög nr. X/XXXX - Y. gr.",
      "quote": "NÁKVÆM tilvitnun úr textanum"
    }
  ],
  "needs_clarification": false,
  "clarification_question": null
}

Ef þú þarft að neita að svara:
{
  "answer_markdown": "Ég finn ekki upplýsingar um þetta í heimildunum.",
  "citations": [],
  "needs_clarification": false,
  "clarification_question": null
}

Ef spurningin er óljós:
{
  "answer_markdown": null,
  "citations": [],
  "needs_clarification": true,
  "clarification_question": "Vinsamlegast tilgreindu nánar..."
}
"""

# Minimum chunks required to attempt an answer
MIN_CHUNKS_FOR_ANSWER = 1


@dataclass
class ChatContext:
    """Context for a chat request."""
    query: str
    chunks: List[Chunk]
    search_type: SearchType
    law_reference: Optional[str] = None
    article_reference: Optional[str] = None


class ChatService:
    """
    Service for handling legal chat queries.

    Uses hybrid search for retrieval and strict validation for responses.
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        llm_fn: Callable[[str, str], Awaitable[str]],
        embedding_fn: Optional[Callable[[str], Awaitable[List[float]]]] = None,
        debug: bool = False,
    ):
        """
        Initialize chat service.

        Args:
            searcher: HybridSearcher instance for retrieval
            llm_fn: Async function that calls LLM (system_prompt, user_prompt) -> response
            embedding_fn: Async function to generate embeddings for query
            debug: If True, include debug info in responses
        """
        self.searcher = searcher
        self.llm_fn = llm_fn
        self.embedding_fn = embedding_fn
        self.debug = debug
        self.validator = ResponseValidator()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request.

        Args:
            request: ChatRequest with query

        Returns:
            ChatResponse with answer, citations, or refusal
        """
        query = request.query.strip()

        if not query:
            return ChatResponse(
                answer=None,
                citations=[],
                confidence=Confidence.NONE,
                failure_reason=FailureReason.AMBIGUOUS_QUERY,
                clarification_question="Vinsamlegast sláðu inn spurningu.",
            )

        # Step 1: Extract references from query
        law_ref = extract_law_reference(query)
        article_ref = extract_article_reference(query)

        # Step 2: Generate embedding if function provided
        embedding = None
        if self.embedding_fn:
            try:
                embedding = await self.embedding_fn(query)
            except Exception:
                pass  # Fall back to keyword-only search

        # Step 3: Retrieve evidence via hybrid search
        try:
            chunks, search_type = await self.searcher.search(
                query=query,
                query_embedding=embedding,
                top_k=15,
            )
        except Exception as e:
            return ChatResponse(
                answer=None,
                citations=[],
                confidence=Confidence.NONE,
                failure_reason=FailureReason.INTERNAL_ERROR,
                debug={"error": f"Search failed: {str(e)}"} if self.debug else None,
            )

        context = ChatContext(
            query=query,
            chunks=chunks,
            search_type=search_type,
            law_reference=law_ref,
            article_reference=article_ref,
        )

        # Step 4: Check if sufficient evidence
        if not chunks or len(chunks) < MIN_CHUNKS_FOR_ANSWER:
            return self._no_evidence_response(context)

        # Step 5: Check for ambiguous query
        if self._is_ambiguous_query(query, chunks):
            return self._clarification_response(context)

        # Step 6: Generate and validate response
        return await self._generate_validated_response(context)

    async def _generate_validated_response(self, context: ChatContext) -> ChatResponse:
        """
        Generate LLM response with validation and retry.
        """
        # Build LLM prompt with context
        user_prompt = self._build_user_prompt(context)

        async def generate_fn(query: str, chunks: List[Chunk], extra: Optional[str]) -> Dict[str, Any]:
            """Generate LLM response."""
            prompt = user_prompt
            if extra:
                prompt = f"{extra}\n\n{prompt}"

            response_text = await self.llm_fn(SYSTEM_PROMPT_IS, prompt)
            return self._parse_llm_response(response_text)

        # Use validation with retry logic
        response = await validate_and_retry(
            generate_fn=generate_fn,
            query=context.query,
            chunks=context.chunks,
            max_retries=1,
            language="is",
        )

        # Add debug info if enabled
        if self.debug:
            response.debug = response.debug or {}
            response.debug.update({
                "search_type": context.search_type.value,
                "chunk_count": len(context.chunks),
                "law_reference": context.law_reference,
            })

        return response

    def _build_user_prompt(self, context: ChatContext) -> str:
        """
        Build the user prompt with context chunks.
        """
        parts = [
            f"SPURNING: {context.query}",
            "",
            "HEIMILDIR:",
        ]

        for i, chunk in enumerate(context.chunks[:10]):  # Limit context size
            parts.append(f"\n--- Heimild {i+1} ---")
            parts.append(f"Locator: {chunk.locator}")
            parts.append(f"Texti: {canonicalize(chunk.chunk_text)}")

        parts.append("\n\nSvaraðu spurningunni í JSON sniði.")

        return "\n".join(parts)

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response JSON.
        """
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text.strip()

            return json.loads(json_str)
        except json.JSONDecodeError:
            # Return as plain answer without citations
            return {
                "answer_markdown": response_text,
                "citations": [],
            }

    def _is_ambiguous_query(self, query: str, chunks: List[Chunk]) -> bool:
        """
        Check if query is too ambiguous to answer.
        """
        query_lower = query.lower()

        # Very short queries
        if len(query.split()) < 3:
            return True

        # Queries without specific content
        vague_patterns = [
            "hvað segir",
            "segðu mér",
            "hvað er",
            "hvernig",
        ]

        has_specific_reference = (
            extract_law_reference(query) is not None or
            extract_article_reference(query) is not None
        )

        if not has_specific_reference:
            for pattern in vague_patterns:
                if pattern in query_lower and len(query.split()) < 6:
                    return True

        return False

    def _no_evidence_response(self, context: ChatContext) -> ChatResponse:
        """
        Response when no relevant evidence found.
        """
        return ChatResponse(
            answer=None,
            citations=[],
            confidence=Confidence.NONE,
            failure_reason=FailureReason.NO_RELEVANT_DATA,
            debug={
                "query": context.query,
                "search_type": context.search_type.value if context.search_type else None,
            } if self.debug else None,
        )

    def _clarification_response(self, context: ChatContext) -> ChatResponse:
        """
        Response requesting clarification for ambiguous query.
        """
        question = "Vinsamlegast tilgreindu nánar hvað þú leitar að."

        if not context.law_reference:
            question += " Til dæmis: Hvaða lög eða lagagrein áttu við?"

        return ChatResponse(
            answer=None,
            citations=[],
            confidence=Confidence.NONE,
            failure_reason=FailureReason.AMBIGUOUS_QUERY,
            clarification_question=question,
        )


# Failure messages in Icelandic
FAILURE_MESSAGES = {
    FailureReason.AMBIGUOUS_QUERY: {
        "title": "Spurningin er of almenn",
        "message": "Vinsamlegast tilgreindu nánar hvað þú leitar að.",
        "suggestion": "Reyndu að nefna tiltekna löggjöf eða lagalegt hugtak.",
    },
    FailureReason.NO_RELEVANT_DATA: {
        "title": "Engar heimildir fundust",
        "message": "Ég fann engar viðeigandi upplýsingar í lagagagnagrunni um þessa spurningu.",
        "suggestion": "Athugaðu hvort spurningin varðar íslensk lög sem eru í gagnagrunni.",
    },
    FailureReason.VALIDATION_FAILED: {
        "title": "Ekki tókst að staðfesta svar",
        "message": "Kerfið gat ekki staðfest að svarið byggist á nákvæmum heimildum.",
        "suggestion": "Reyndu að orða spurninguna á annan hátt.",
    },
    FailureReason.RATE_LIMITED: {
        "title": "Of margar fyrirspurnir",
        "message": "Vinsamlegast bíddu í smástund.",
        "suggestion": None,
    },
    FailureReason.INTERNAL_ERROR: {
        "title": "Kerfisvilla",
        "message": "Óvænt villa kom upp.",
        "suggestion": "Vinsamlegast reyndu aftur síðar.",
    },
}


def get_failure_message(reason: FailureReason) -> Dict[str, str]:
    """Get user-facing failure message."""
    return FAILURE_MESSAGES.get(reason, FAILURE_MESSAGES[FailureReason.INTERNAL_ERROR])
