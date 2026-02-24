"""
Chat Service for Legal Search MVP

This module handles the chat/query flow:
1. Parse query for references
2. Retrieve evidence via hybrid search (match_documents_hybrid RPC)
3. Check if sufficient evidence exists
4. Build LLM context from chunks (with chunk_id for grounding)
5. Call LLM with strict JSON schema
6. Validate citations (chunk_id anchored, per-chunk quote match)
7. Return answer or refusal

CRITICAL: No answer is returned unless all citations validate.
"""

import json
from typing import List, Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass

from app.services.retrieval import retrieve_hybrid_async
from app.services.citation import build_context, validate_citations
from app.services.search import (
    extract_law_reference,
    extract_article_reference,
)
from app.services.validation import STRICT_QUOTE_INSTRUCTIONS_IS
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
6. chunk_id verður að vera nákvæmlega sami og birtist í SOURCE blokkinni

SVARSNIÐ (JSON):
{
  "answer_markdown": "Svarið hér með [1] tilvísunum...",
  "citations": [
    {
      "chunk_id": "<chunk_id nákvæmlega eins og í SOURCE blokk>",
      "law_reference": "<law_reference nákvæmlega eins og í SOURCE blokk>",
      "article_locator": "<article_locator nákvæmlega eins og í SOURCE blokk>",
      "quote": "NÁKVÆM tilvitnun — orðrétt úr textanum"
    }
  ],
  "confidence": "high|medium|low|none",
  "needs_clarification": false,
  "clarification_question": null
}

Ef þú þarft að neita að svara:
{
  "answer_markdown": "Ég finn ekki upplýsingar um þetta í heimildunum.",
  "citations": [],
  "confidence": "none",
  "needs_clarification": false,
  "clarification_question": null
}

Ef spurningin er óljós:
{
  "answer_markdown": null,
  "citations": [],
  "confidence": "none",
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
    chunks: list  # list[dict] from retrieve_hybrid_async
    law_reference: Optional[str] = None
    article_reference: Optional[str] = None


def _compute_confidence(n_citations: int) -> Confidence:
    if n_citations >= 2:
        return Confidence.HIGH
    if n_citations == 1:
        return Confidence.MEDIUM
    return Confidence.NONE


class ChatService:
    """
    Service for handling legal chat queries.

    Uses hybrid retrieval (match_documents_hybrid RPC) and chunk_id-anchored
    citation validation for responses.
    """

    def __init__(
        self,
        pool,
        llm_fn: Callable[[str, str], Awaitable[str]],
        embedding_fn: Optional[Callable[[str], Awaitable[List[float]]]] = None,
        debug: bool = False,
    ):
        """
        Initialize chat service.

        Args:
            pool: asyncpg.Pool for database access
            llm_fn: Async function that calls LLM (system_prompt, user_prompt) -> response
            embedding_fn: Async function to generate embeddings for query
            debug: If True, include debug info in responses
        """
        self.pool = pool
        self.llm_fn = llm_fn
        self.embedding_fn = embedding_fn
        self.debug = debug

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
                pass  # Fall back to FTS-only search

        # Step 3: Retrieve evidence via hybrid retrieval RPC
        try:
            hits = await retrieve_hybrid_async(
                pool=self.pool,
                embedding=embedding,
                query_text=query,
                top_k=8,
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
            chunks=hits,
            law_reference=law_ref,
            article_reference=article_ref,
        )

        # Step 4: Check if sufficient evidence
        if not hits or len(hits) < MIN_CHUNKS_FOR_ANSWER:
            return self._no_evidence_response(context)

        # Step 5: Check for ambiguous query
        if self._is_ambiguous_query(query, hits):
            return self._clarification_response(context)

        # Step 6: Generate and validate response
        return await self._generate_validated_response(context)

    async def _generate_validated_response(self, context: ChatContext) -> ChatResponse:
        """
        Generate LLM response with chunk_id-anchored validation and retry.
        """
        user_prompt, ctx_chunks = self._build_user_prompt(context)

        for attempt in range(2):  # 0 = first try, 1 = retry with strict instructions
            if attempt > 0:
                prompt = (STRICT_QUOTE_INSTRUCTIONS_IS + "\n\n" + user_prompt).strip()
            else:
                prompt = user_prompt

            raw = await self.llm_fn(SYSTEM_PROMPT_IS, prompt)
            parsed = self._parse_llm_response(raw)

            citations_raw = parsed.get("citations", [])
            errors = validate_citations(citations_raw, ctx_chunks)

            if not errors:
                citations = [
                    Citation(
                        document_id=c.get("chunk_id", ""),
                        locator=c.get("article_locator", ""),
                        quote=c.get("quote", ""),
                        canonical_url=None,
                    )
                    for c in citations_raw
                ]
                response = ChatResponse(
                    answer=parsed.get("answer_markdown") or parsed.get("answer"),
                    citations=citations,
                    confidence=_compute_confidence(len(citations)),
                    debug={"attempt": attempt, "search": "hybrid"} if self.debug else None,
                )
                if self.debug:
                    response.debug = response.debug or {}
                    response.debug.update({
                        "chunk_count": len(context.chunks),
                        "law_reference": context.law_reference,
                    })
                return response

        # Both attempts failed validation
        return ChatResponse(
            answer=None,
            citations=[],
            confidence=Confidence.NONE,
            failure_reason=FailureReason.VALIDATION_FAILED,
            debug={"attempts": 2, "last_errors": errors} if self.debug else None,
        )

    def _build_user_prompt(self, context: ChatContext) -> tuple:
        """
        Build the user prompt with context chunks using build_context().

        Returns:
            (prompt_str, ctx_chunks) — ctx_chunks needed for validation
        """
        ctx_str, ctx_chunks = build_context(context.chunks)
        prompt = (
            f"SPURNING: {context.query}\n\n"
            f"HEIMILDIR:\n{ctx_str}\n\n"
            f"Svaraðu í JSON sniði."
        )
        return prompt, ctx_chunks

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response JSON.
        """
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
            return {
                "answer_markdown": response_text,
                "citations": [],
            }

    def _is_ambiguous_query(self, query: str, chunks: list) -> bool:
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
                "search": "hybrid",
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
