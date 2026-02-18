"""
Chat API Endpoint for Legal Search MVP

POST /chat endpoint that:
1. Receives query
2. Retrieves evidence via hybrid search
3. Generates answer with LLM
4. Validates all citations
5. Returns answer or refusal

No streaming - response returned only after validation completes.
"""

import os
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from app.services.canonicalize import canonicalize
from app.services.chat import ChatService, get_failure_message
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    FailureReason,
    Confidence,
)


@dataclass
class APIResponse:
    """API response format."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None
    request_id: str = ""


@dataclass
class QueryLog:
    """Privacy-preserving query log entry."""
    request_id: str
    timestamp: str
    query_length: int
    query_hash: str  # SHA256 prefix, not full query
    chunk_count: int
    validation_passed: bool
    retry_count: int
    failure_reason: Optional[str] = None


class ChatEndpoint:
    """
    Chat endpoint handler.

    Handles request processing, logging, and response formatting.
    """

    def __init__(
        self,
        chat_service: ChatService,
        log_fn: Optional[callable] = None,
        debug: bool = False,
    ):
        """
        Initialize endpoint.

        Args:
            chat_service: ChatService instance
            log_fn: Optional async function to log queries
            debug: If True, include debug info in responses
        """
        self.chat_service = chat_service
        self.log_fn = log_fn
        self.debug = debug

    async def handle_chat(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> APIResponse:
        """
        Handle a chat request.

        Args:
            query: User query
            filters: Optional search filters
            ip_address: Client IP for rate limiting (not stored directly)

        Returns:
            APIResponse with answer or error
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = datetime.now(timezone.utc)

        # Create request object
        request = ChatRequest(
            query=query,
            filters=filters,
        )

        # Process request
        try:
            response = await self.chat_service.chat(request)
        except Exception as e:
            return APIResponse(
                success=False,
                error={
                    "type": "internal_error",
                    "message": "Kerfisvilla kom upp",
                },
                request_id=request_id,
            )

        # Log query (privacy-preserving)
        if self.log_fn:
            log_entry = QueryLog(
                request_id=request_id,
                timestamp=start_time.isoformat(),
                query_length=len(query),
                query_hash=hashlib.sha256(query.encode()).hexdigest()[:16],
                chunk_count=len(response.debug.get("chunk_count", 0)) if response.debug else 0,
                validation_passed=response.failure_reason != FailureReason.VALIDATION_FAILED,
                retry_count=response.debug.get("attempts", 1) if response.debug else 1,
                failure_reason=response.failure_reason.value if response.failure_reason else None,
            )
            try:
                await self.log_fn(log_entry)
            except Exception:
                pass  # Don't fail request due to logging error

        # Format response
        if response.answer or not response.failure_reason:
            # Success response
            return APIResponse(
                success=True,
                data=self._format_success_response(response),
                request_id=request_id,
            )
        else:
            # Failure response
            failure_msg = get_failure_message(response.failure_reason)
            return APIResponse(
                success=False,
                data=self._format_failure_response(response, failure_msg),
                request_id=request_id,
            )

    def _format_success_response(self, response: ChatResponse) -> Dict[str, Any]:
        """Format successful response."""
        result = {
            "answer": response.answer,
            "citations": [
                {
                    "locator": c.locator,
                    "quote": c.quote,
                    "url": c.canonical_url,
                }
                for c in response.citations
            ],
            "confidence": response.confidence.value,
        }

        if self.debug and response.debug:
            result["debug"] = response.debug

        return result

    def _format_failure_response(
        self,
        response: ChatResponse,
        failure_msg: Dict[str, str],
    ) -> Dict[str, Any]:
        """Format failure response."""
        result = {
            "failure_type": response.failure_reason.value,
            "title": failure_msg.get("title"),
            "message": failure_msg.get("message"),
            "suggestion": failure_msg.get("suggestion"),
        }

        if response.clarification_question:
            result["clarification_question"] = response.clarification_question

        if self.debug and response.debug:
            result["debug"] = response.debug

        return result


# FastAPI application setup
def create_chat_router():
    """
    Create FastAPI router for chat endpoint.

    Note: This requires FastAPI to be installed and imported.
    For testing, we use the ChatEndpoint class directly.
    """
    try:
        from fastapi import APIRouter, Request, HTTPException
        from pydantic import BaseModel
    except ImportError:
        return None

    router = APIRouter()

    class ChatRequestModel(BaseModel):
        query: str
        filters: Optional[Dict[str, Any]] = None

    class ChatResponseModel(BaseModel):
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[Dict[str, str]] = None
        request_id: str = ""

    @router.post("/chat", response_model=ChatResponseModel)
    async def chat_endpoint(
        request: ChatRequestModel,
        http_request: Request,
    ):
        """
        Chat endpoint for legal queries.

        Args:
            request: ChatRequestModel with query

        Returns:
            ChatResponseModel with answer or error
        """
        # Get endpoint handler from app state
        endpoint: ChatEndpoint = http_request.app.state.chat_endpoint

        # Get client IP (for rate limiting, not stored)
        client_ip = http_request.client.host if http_request.client else None

        response = await endpoint.handle_chat(
            query=request.query,
            filters=request.filters,
            ip_address=client_ip,
        )

        return ChatResponseModel(**asdict(response))

    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    return router
