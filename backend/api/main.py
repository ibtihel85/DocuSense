"""
backend/api/main.py
────────────────────
FastAPI application entry point.
Configures middleware, routers, startup events, and error handlers.
"""
from __future__ import annotations
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.api.routes import documents, query, health
from backend.core.config import get_settings
from backend.utils.logger import setup_logging, get_logger

settings = get_settings()
setup_logging(settings.log_level, settings.log_format, settings.log_file)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic."""
    logger.info(
        "DocuSense starting",
        model=settings.ollama_model,
        embedding=settings.embedding_model,
        chroma_dir=settings.chroma_persist_dir,
    )
    # Warm up embedding model on startup
    try:
        from backend.core.embeddings import get_embedding_model
        _ = get_embedding_model()
        logger.info("Embedding model warmed up")
    except Exception as e:
        logger.warning("Embedding model warmup failed", error=str(e))

    yield
    logger.info("DocuSense shutting down")


app = FastAPI(
    title="DocuSense API",
    description="Agentic Document Intelligence Platform for Legal & Compliance",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow Streamlit frontend) ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(int((time.time() - start) * 1000)) + "ms"
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "DocuSense",
        "version": "1.0.0",
        "description": "Agentic Document Intelligence Platform",
        "docs": "/docs",
    }
