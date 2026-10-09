"""
RAG Chatbot — FastAPI Server
==============================
REST API with endpoints for:
  - Document upload and ingestion
  - Chat queries with RAG
  - Session and collection management
  - Health checks and statistics

Also serves the frontend static files.
"""

import os
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from document_processor import process_uploaded_file, SUPPORTED_EXTENSIONS
from vector_store import vector_store_manager
from rag_chain import rag_chain

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    logger.info("━━━ RAG Chatbot Server Starting ━━━")
    logger.info(f"  Model      : {settings.gemini_model}")
    logger.info(f"  Embeddings : {settings.embedding_model}")
    logger.info(f"  Chunk size : {settings.chunk_size}")
    logger.info(f"  Top-K      : {settings.top_k_results}")
    yield
    logger.info("━━━ RAG Chatbot Server Shutting Down ━━━")


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot powered by LangChain & Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list
    search_query: str
    num_chunks_retrieved: int
    session_id: str


# ── API Routes ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": settings.gemini_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the vector store.
    Supports: PDF, TXT, DOCX, CSV
    """
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. "
                   f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{file.filename}"
    file_path = os.path.join(settings.upload_dir, safe_name)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Process and ingest
        chunks = process_uploaded_file(file_path)
        num_added = vector_store_manager.add_documents(chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": num_added,
            "total_chunks_in_store": vector_store_manager.get_document_count(),
        }

    except Exception as e:
        logger.error(f"Upload failed for '{file.filename}': {e}")
        # Clean up on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a question to the RAG chatbot.
    Returns a grounded answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vector_store_manager.get_document_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload documents first.",
        )

    try:
        result = await rag_chain.query(
            question=request.question,
            session_id=request.session_id,
        )
        result["session_id"] = request.session_id
        return result

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """Get vector store statistics."""
    return vector_store_manager.get_collection_stats()


@app.post("/api/clear")
async def clear_store():
    """Clear all documents from the vector store."""
    vector_store_manager.clear_store()
    rag_chain.clear_all_histories()
    return {"status": "cleared", "message": "All documents and histories removed."}


@app.post("/api/clear-history")
async def clear_history(session_id: str = "default"):
    """Clear conversation history for a session."""
    rag_chain.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# ── Static Files & SPA ──────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "RAG Chatbot API is running. Frontend not found."}


# ── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
