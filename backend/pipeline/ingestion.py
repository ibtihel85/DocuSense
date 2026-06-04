"""
backend/pipeline/ingestion.py
──────────────────────────────
Full document ingestion pipeline:
  Upload → OCR/Parse → Clean → Chunk → Embed → Store (ChromaDB + BM25)
"""
from __future__ import annotations
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from backend.core.config import get_settings
from backend.core.embeddings import get_embedding_model
from backend.core.vectorstore import get_vector_store
from backend.core.bm25_index import get_bm25_index
from backend.ocr.processor import get_extractor
from backend.pipeline.chunker import SemanticChunker
from backend.utils.helpers import compute_file_hash, extract_doc_id_from_filename
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class IngestionResult:
    doc_id: str
    doc_name: str
    file_type: str
    page_count: int
    chunk_count: int
    total_tokens: int
    ocr_used: bool
    duration_seconds: float
    success: bool
    error: str | None = None


class IngestionPipeline:
    """
    Orchestrates the full document ingestion workflow.
    Thread-safe for sequential use; use one instance per request for concurrency.
    """

    def __init__(self) -> None:
        self.extractor = get_extractor()
        self.chunker = SemanticChunker()
        self.embedder = get_embedding_model()
        self.vector_store = get_vector_store()
        self.bm25_index = get_bm25_index()

    def ingest_file(self, file_path: str | Path, doc_name: str | None = None) -> IngestionResult:
        """
        Ingest a single document file.

        Args:
            file_path: Path to the document
            doc_name: Display name (defaults to filename)

        Returns:
            IngestionResult with stats and status
        """
        import time
        start = time.time()
        path = Path(file_path)

        if not path.exists():
            return IngestionResult("", "", "", 0, 0, 0, False, 0.0, False, f"File not found: {path}")

        doc_name = doc_name or path.name
        file_hash = compute_file_hash(path)
        doc_id = f"{extract_doc_id_from_filename(path.name)}_{file_hash[:8]}"
        file_type = path.suffix.lower().lstrip(".")

        logger.info("Starting document ingestion", doc_id=doc_id, file=doc_name)

        try:
            # ── Step 1: Extract text ─────────────────────────────────────────
            extracted_pages = self.extractor.extract(path)
            ocr_used = any(p.method == "ocr" for p in extracted_pages)
            page_count = len(extracted_pages)

            # ── Step 2: Chunk ────────────────────────────────────────────────
            pages_data = [(p.page_number, p.text) for p in extracted_pages if p.text.strip()]
            chunks = self.chunker.chunk_pages(pages_data)

            if not chunks:
                logger.warning("No text extracted from document", doc_id=doc_id)
                return IngestionResult(doc_id, doc_name, file_type, page_count, 0, 0, ocr_used,
                                       time.time() - start, False, "No text content extracted")

            # ── Step 3: Build metadata ───────────────────────────────────────
            ingested_at = datetime.datetime.utcnow().isoformat()
            chunk_ids = [f"{doc_id}_chunk_{c.chunk_index}" for c in chunks]
            texts = [c.text for c in chunks]
            metadatas: list[dict[str, Any]] = [
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "file_type": file_type,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number or 0,
                    "token_count": c.token_count,
                    "ocr_used": str(ocr_used),
                    "ingested_at": ingested_at,
                    "file_hash": file_hash,
                }
                for c in chunks
            ]

            # ── Step 4: Embed ────────────────────────────────────────────────
            logger.info("Computing embeddings", chunk_count=len(chunks))
            embeddings = self.embedder.embed_documents(texts, show_progress=len(chunks) > 20)

            # ── Step 5: Store in ChromaDB ────────────────────────────────────
            self.vector_store.add_chunks(chunk_ids, texts, embeddings, metadatas)

            # ── Step 6: Update BM25 index ────────────────────────────────────
            self.bm25_index.add_chunks(chunk_ids, texts, metadatas)

            total_tokens = sum(c.token_count for c in chunks)
            duration = time.time() - start

            logger.info(
                "Document ingestion complete",
                doc_id=doc_id,
                chunks=len(chunks),
                tokens=total_tokens,
                duration=f"{duration:.2f}s",
            )

            return IngestionResult(
                doc_id=doc_id,
                doc_name=doc_name,
                file_type=file_type,
                page_count=page_count,
                chunk_count=len(chunks),
                total_tokens=total_tokens,
                ocr_used=ocr_used,
                duration_seconds=duration,
                success=True,
            )

        except Exception as e:
            logger.error("Ingestion failed", doc_id=doc_id, error=str(e), exc_info=True)
            return IngestionResult(
                doc_id=doc_id, doc_name=doc_name, file_type=file_type,
                page_count=0, chunk_count=0, total_tokens=0, ocr_used=False,
                duration_seconds=time.time() - start, success=False, error=str(e),
            )

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from all indexes."""
        try:
            self.vector_store.delete_document(doc_id)
            self.bm25_index.remove_document(doc_id)
            logger.info("Document deleted", doc_id=doc_id)
            return True
        except Exception as e:
            logger.error("Delete failed", doc_id=doc_id, error=str(e))
            return False
