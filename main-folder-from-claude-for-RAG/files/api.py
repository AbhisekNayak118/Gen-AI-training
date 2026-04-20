# ============================================================
# api.py — FastAPI Backend
# ============================================================
# Exposes the RAG pipeline as a REST API.
# In a real company, your frontend (React, mobile app, etc.)
# calls these endpoints instead of importing Python directly.
#
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   GET  /             → Welcome message
#   GET  /health       → System health + vector store stats
#   POST /ask          → Ask a question (JSON response)
#   POST /ask/stream   → Ask with streaming response (SSE)
#   POST /ingest       → Trigger re-indexing of documents
# ============================================================

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from src.chain import ask, ask_stream
from src.retriever import get_collection_stats
from src.ingest import run_ingestion

# ─── App Setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Customer Support API",
    description="AI-powered customer support chatbot using open-source LLMs",
    version="1.0.0",
)

# CORS allows your frontend (React, etc.) to call this API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # In production: restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ───────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Customer's question",
        example="How do I return a product I bought last week?",
    )
    top_k: Optional[int] = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve",
    )


class SourceDocument(BaseModel):
    file: str
    page: int | str
    preview: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    model_used: str = "mistral (via Ollama)"


class HealthResponse(BaseModel):
    api_status: str
    vectorstore: dict


# ─── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {
        "message": "RAG Customer Support API is running",
        "docs": "/docs",   # FastAPI auto-generates Swagger UI here
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    System health check. Use this in your Docker/k8s readiness probe.
    Returns vector store stats so you know how many documents are indexed.
    """
    stats = get_collection_stats()
    return {
        "api_status": "healthy",
        "vectorstore": stats,
    }


@app.post("/ask", response_model=AnswerResponse, tags=["Chat"])
def ask_question(request: QuestionRequest):
    """
    Main Q&A endpoint. Returns a complete JSON response.

    The client sends a question, we retrieve relevant context,
    generate an answer with Mistral, and return answer + sources.

    Use this for: traditional web apps, mobile apps, backend integrations.
    """
    try:
        logger.info(f"API /ask | question: {request.question[:60]}...")
        result = ask(request.question)

        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate answer")

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "model_used": "mistral (via Ollama)",
        }

    except ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure it's running: `ollama serve`",
        )
    except Exception as e:
        logger.error(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream", tags=["Chat"])
def ask_question_stream(request: QuestionRequest):
    """
    Streaming endpoint using Server-Sent Events (SSE).

    Tokens stream from Ollama → FastAPI → Browser as they're generated.
    This creates the "typing" effect that feels much faster than waiting
    for the full answer.

    Client-side usage (JavaScript):
        const response = await fetch('/ask/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: 'How do I return an item?'})
        });
        const reader = response.body.getReader();
        // Read tokens as they arrive...
    """
    def token_generator():
        try:
            for token in ask_stream(request.question):
                # SSE format: "data: <content>\n\n"
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@app.post("/ingest", tags=["Admin"])
def trigger_ingestion(reset: bool = False, background_tasks: BackgroundTasks = None):
    """
    Triggers a re-indexing of all documents in /documents folder.

    In production, call this endpoint:
    - When you add new support documents
    - When you update existing FAQs
    - Scheduled nightly to catch any changes

    Args:
        reset: If True, clears all existing vectors before re-indexing.

    This runs as a background task so the API doesn't time out.
    """
    background_tasks.add_task(run_ingestion, reset=reset)

    return {
        "message": "Ingestion started in background",
        "reset": reset,
        "status": "check /health to monitor progress",
    }
