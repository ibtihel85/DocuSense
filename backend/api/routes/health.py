"""Health check endpoint."""
from fastapi import APIRouter
from backend.api.schemas import HealthResponse
from backend.core.llm import OllamaClient
from backend.core.vectorstore import get_vector_store
from backend.core.bm25_index import get_bm25_index
from backend.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check — verify all components are operational."""
    llm = OllamaClient()
    vs = get_vector_store()
    bm25 = get_bm25_index()
    return HealthResponse(
        status="ok",
        ollama_available=llm.is_available(),
        ollama_model=settings.ollama_model,
        vector_store_chunks=vs.get_chunk_count(),
        bm25_chunks=bm25.chunk_count,
    )
