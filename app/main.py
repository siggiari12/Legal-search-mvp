"""
Main FastAPI Application for Legal Search MVP

This is the entry point for the API server.

Usage:
    # Development
    uvicorn app.main:app --reload

    # Production
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import FastAPI (may not be installed in test environment)
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# Global service instance
_production_chat_service = None


@asynccontextmanager
async def lifespan(app):
    """
    Application lifespan handler.

    Initializes services on startup and cleans up on shutdown.
    """
    global _production_chat_service

    # Startup
    print("Starting Legal Search MVP...")

    try:
        from app.services.production_chat import ProductionChatService

        debug = os.environ.get("DEBUG", "").lower() == "true"
        skip_embeddings = os.environ.get("SKIP_EMBEDDINGS", "").lower() == "true"

        _production_chat_service = ProductionChatService(
            debug=debug,
            skip_embeddings=skip_embeddings,
        )
        await _production_chat_service.connect()
        print("Connected to all services")

        # Check health
        health = await _production_chat_service.health_check()
        print(f"Service health: {health}")

    except Exception as e:
        print(f"WARNING: Failed to initialize chat service: {e}")
        print("API will return errors for chat requests")

    yield

    # Shutdown
    print("Shutting down...")
    if _production_chat_service:
        await _production_chat_service.close()
    print("Cleanup complete")


def create_app() -> Optional["FastAPI"]:
    """
    Create and configure FastAPI application.

    Returns:
        FastAPI application or None if FastAPI not available
    """
    if not FASTAPI_AVAILABLE:
        return None

    app = FastAPI(
        title="Legal Search MVP",
        description="Closed-book AI chat system for Icelandic legal information",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request models
    class ChatRequestModel(BaseModel):
        query: str
        filters: Optional[Dict[str, Any]] = None

    class CitationModel(BaseModel):
        locator: str
        quote: str
        url: Optional[str] = None

    class ChatResponseModel(BaseModel):
        success: bool
        answer: Optional[str] = None
        citations: list = []
        confidence: Optional[str] = None
        failure_type: Optional[str] = None
        message: Optional[str] = None
        clarification_question: Optional[str] = None
        request_id: str = ""

    # Get frontend directory path
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

    # Mount static files if frontend directory exists
    if os.path.exists(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    # Root endpoint - serve frontend
    @app.get("/")
    async def root():
        if os.path.exists(frontend_dir):
            return FileResponse(os.path.join(frontend_dir, "index.html"))
        return {
            "name": "Legal Search MVP",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
        }

    # Serve CSS and JS files directly
    @app.get("/styles.css")
    async def styles():
        return FileResponse(
            os.path.join(frontend_dir, "styles.css"),
            media_type="text/css"
        )

    @app.get("/app.js")
    async def app_js():
        return FileResponse(
            os.path.join(frontend_dir, "app.js"),
            media_type="application/javascript"
        )

    # Health check
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        from datetime import datetime, timezone

        status = {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

        if _production_chat_service:
            try:
                health = await _production_chat_service.health_check()
                status["services"] = health
            except Exception:
                status["services"] = {"error": "Failed to check service health"}

        return status

    # Chat endpoint
    @app.post("/api/chat", response_model=ChatResponseModel)
    async def chat_endpoint(request: ChatRequestModel, http_request: Request):
        """
        Chat endpoint for legal queries.

        Accepts a query in Icelandic and returns an answer with citations.
        All citations are validated against the source documents.
        """
        import uuid

        request_id = str(uuid.uuid4())[:8]

        if not _production_chat_service:
            return ChatResponseModel(
                success=False,
                failure_type="service_unavailable",
                message="Chat service not available",
                request_id=request_id,
            )

        try:
            from app.models.schemas import ChatRequest

            response = await _production_chat_service.chat(
                ChatRequest(query=request.query, filters=request.filters)
            )

            if response.answer or not response.failure_reason:
                return ChatResponseModel(
                    success=True,
                    answer=response.answer,
                    citations=[
                        {
                            "locator": c.locator,
                            "quote": c.quote,
                            "url": c.canonical_url,
                        }
                        for c in response.citations
                    ],
                    confidence=response.confidence.value if response.confidence else None,
                    request_id=request_id,
                )
            else:
                from app.services.chat import get_failure_message

                failure_msg = get_failure_message(response.failure_reason)
                return ChatResponseModel(
                    success=False,
                    failure_type=response.failure_reason.value,
                    message=failure_msg.get("message"),
                    clarification_question=response.clarification_question,
                    request_id=request_id,
                )

        except Exception as e:
            return ChatResponseModel(
                success=False,
                failure_type="internal_error",
                message=f"Kerfisvilla: {str(e)}" if os.environ.get("DEBUG") else "Kerfisvilla kom upp",
                request_id=request_id,
            )

    # Stats endpoint
    @app.get("/api/stats")
    async def stats():
        """Get database statistics."""
        if not _production_chat_service:
            raise HTTPException(status_code=503, detail="Service not available")

        try:
            db_stats = await _production_chat_service._searcher.db.get_stats()
            return {
                "documents": db_stats.get("documents", 0),
                "chunks": db_stats.get("chunks", 0),
                "embedded_chunks": db_stats.get("embedded_chunks", 0),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Create the app instance
app = create_app()


# Development server entry point
if __name__ == "__main__":
    import uvicorn

    if app:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )
    else:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
