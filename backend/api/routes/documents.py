"""
backend/api/routes/documents.py
────────────────────────────────
Document upload and management endpoints.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.api.schemas import DocumentInfo, IngestionResponse
from backend.core.config import get_settings
from backend.core.vectorstore import get_vector_store
from backend.pipeline.ingestion import IngestionPipeline
from backend.utils.logger import get_logger

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)

ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


@router.post("/documents/upload", response_model=IngestionResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF, DOCX, TXT, or scanned image).
    Triggers OCR + chunking + embedding + indexing pipeline.
    """
    # Validate file type
    content_type = file.content_type or ""
    suffix = Path(file.filename or "").suffix.lower()
    allowed_suffixes = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}

    if suffix not in allowed_suffixes:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed_suffixes}")

    # Check file size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / (file.filename or "upload")

    try:
        # Save file
        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(413, f"File too large. Max: {settings.max_upload_size_mb}MB")

        with open(save_path, "wb") as f:
            f.write(content)

        logger.info("File saved, starting ingestion", file=file.filename, size_kb=len(content)//1024)

        # Run ingestion pipeline
        pipeline = IngestionPipeline()
        result = pipeline.ingest_file(save_path, doc_name=file.filename)

        if not result.success:
            raise HTTPException(500, f"Ingestion failed: {result.error}")

        return IngestionResponse(
            doc_id=result.doc_id, doc_name=result.doc_name,
            file_type=result.file_type, page_count=result.page_count,
            chunk_count=result.chunk_count, total_tokens=result.total_tokens,
            ocr_used=result.ocr_used, duration_seconds=result.duration_seconds,
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed", error=str(e), exc_info=True)
        raise HTTPException(500, f"Upload processing failed: {e}")


@router.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all ingested documents."""
    vs = get_vector_store()
    docs = vs.list_documents()
    return [DocumentInfo(**d) for d in docs]


@router.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from the system."""
    pipeline = IngestionPipeline()
    success = pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(500, "Failed to delete document")
    return {"message": f"Document {doc_id} deleted successfully"}
