"""
backend/core/vectorstore.py
────────────────────────────
ChromaDB vector store wrapper.
Handles document storage, retrieval, and metadata management.
Supports collection-level document filtering.
"""

from __future__ import annotations

from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DocumentChunk:
    """Represents a single retrieved document chunk with metadata."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        doc_id: str,
        doc_name: str,
        chunk_index: int,
        page_number: int | None,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.page_number = page_number
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorStore:
    """
    ChromaDB-backed vector store for document chunks.
    
    Each chunk is stored with:
    - Embedding vector
    - Text content
    - Metadata: doc_id, doc_name, chunk_index, page_number, file_type
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        logger.info("Initialising ChromaDB", persist_dir=self.persist_dir)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            "ChromaDB collection ready",
            collection=self.collection_name,
            count=self._collection.count(),
        )

    # ── Ingestion ────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]] | np.ndarray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add chunks to the vector store.
        Uses upsert to handle re-ingestion gracefully.

        Args:
            chunk_ids: Unique IDs for each chunk
            texts: Raw text of each chunk
            embeddings: Pre-computed embedding vectors
            metadatas: Metadata dicts (must contain doc_id, doc_name, etc.)
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        # ChromaDB requires string values in metadata
        sanitized_meta = [
            {k: str(v) if v is not None else "" for k, v in m.items()}
            for m in metadatas
        ]

        self._collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=sanitized_meta,
        )

        logger.info("Added chunks to vector store", count=len(chunk_ids))

    # ── Retrieval ────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 10,
        doc_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """
        Retrieve top-k similar chunks using dense vector search.

        Args:
            query_embedding: Query vector (should be L2-normalized)
            top_k: Number of results to return
            doc_ids: Optional filter — only search within these document IDs

        Returns:
            List of DocumentChunk objects sorted by similarity score
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        where_filter: dict[str, Any] | None = None
        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"doc_id": doc_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": doc_ids}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, self._collection.count())),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if not results["ids"] or not results["ids"][0]:
            return chunks

        for i, chunk_id in enumerate(results["ids"][0]):
            text = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            # ChromaDB returns distance (lower = more similar for cosine)
            # Convert to similarity score (0–1)
            distance = results["distances"][0][i]
            score = 1.0 - (distance / 2.0)  # Cosine distance → similarity

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=text,
                    doc_id=meta.get("doc_id", ""),
                    doc_name=meta.get("doc_name", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    page_number=int(meta["page_number"]) if meta.get("page_number") else None,
                    score=float(score),
                    metadata=dict(meta),
                )
            )

        return chunks

    # ── Management ───────────────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted document from vector store", doc_id=doc_id)

    def list_documents(self) -> list[dict[str, Any]]:
        """Return list of unique documents in the store."""
        results = self._collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": meta.get("doc_name", ""),
                    "file_type": meta.get("file_type", ""),
                    "ingested_at": meta.get("ingested_at", ""),
                }
        return list(seen.values())

    def get_chunk_count(self) -> int:
        """Return total number of stored chunks."""
        return self._collection.count()

    def get_document_chunks(self, doc_id: str) -> list[DocumentChunk]:
        """Retrieve all chunks for a specific document."""
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=results["documents"][i],
                    doc_id=doc_id,
                    doc_name=meta.get("doc_name", ""),
                    chunk_index=int(meta.get("chunk_index", i)),
                    page_number=int(meta["page_number"]) if meta.get("page_number") else None,
                    score=1.0,
                    metadata=dict(meta),
                )
            )

        # Sort by chunk_index for reading order
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def reset(self) -> None:
        """DANGER: Delete all data in the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Vector store collection reset")


from functools import lru_cache


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return cached singleton VectorStore instance."""
    return VectorStore()
