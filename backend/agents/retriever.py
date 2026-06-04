"""
backend/agents/retriever.py
────────────────────────────
Retriever Agent: Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.
Optionally reranks with MiniLM cross-encoder.
"""
from __future__ import annotations
from typing import Any
from backend.core.embeddings import get_embedding_model
from backend.core.vectorstore import get_vector_store, DocumentChunk
from backend.core.bm25_index import get_bm25_index
from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RetrievedContext:
    """Container for retrieved chunks with fusion metadata."""
    def __init__(self, chunks: list[DocumentChunk], retrieval_method: str) -> None:
        self.chunks = chunks
        self.retrieval_method = retrieval_method

    def to_context_string(self) -> str:
        """Format chunks for LLM context window with citation markers."""
        parts = []
        for i, chunk in enumerate(self.chunks, 1):
            parts.append(
                f"[Doc: {chunk.doc_name}, Chunk: {chunk.chunk_index}, "
                f"Page: {chunk.page_number or 'N/A'}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.chunks]


def reciprocal_rank_fusion(
    dense_results: list[DocumentChunk],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[DocumentChunk]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion.
    RRF score = sum(weight / (k + rank)) for each result list.

    Args:
        k: RRF smoothing constant (60 is standard)
        dense_weight / sparse_weight: Relative importance
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, DocumentChunk] = {}

    # Dense results contribution
    for rank, chunk in enumerate(dense_results, 1):
        cid = chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank)
        chunk_map[cid] = chunk

    # Sparse (BM25) results contribution
    for rank, result in enumerate(sparse_results, 1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + sparse_weight / (k + rank)
        if cid not in chunk_map:
            # Reconstruct DocumentChunk from BM25 result
            meta = result.get("metadata", {})
            chunk_map[cid] = DocumentChunk(
                chunk_id=cid,
                text=result["text"],
                doc_id=meta.get("doc_id", ""),
                doc_name=meta.get("doc_name", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                page_number=int(meta["page_number"]) if meta.get("page_number") else None,
                score=result["score"],
                metadata=meta,
            )

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = []
    for cid in sorted_ids:
        chunk = chunk_map[cid]
        chunk.score = scores[cid]  # Update score with fused value
        fused.append(chunk)
    return fused


class RetrieverAgent:
    """
    Hybrid retriever combining dense (ChromaDB) and sparse (BM25) search.
    Uses Reciprocal Rank Fusion for combining results.
    Optionally applies MiniLM cross-encoder reranking.
    """

    def __init__(self) -> None:
        self.embedder = get_embedding_model()
        self.vector_store = get_vector_store()
        self.bm25_index = get_bm25_index()
        self._reranker = None  # Lazy load

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> RetrievedContext:
        """
        Run hybrid retrieval for a query.

        Args:
            query: Natural language query
            top_k: Number of final chunks to return
            doc_ids: Filter to specific documents

        Returns:
            RetrievedContext with ranked chunks
        """
        top_k = top_k or settings.top_k_final

        # ── Dense retrieval ──────────────────────────────────────────────────
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=settings.top_k_dense,
            doc_ids=doc_ids,
        )

        # ── Sparse (BM25) retrieval ──────────────────────────────────────────
        sparse_results = self.bm25_index.search(
            query=query,
            top_k=settings.top_k_sparse,
            doc_ids=doc_ids,
        )

        logger.debug(
            "Retrieval results",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        # ── Reciprocal Rank Fusion ───────────────────────────────────────────
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        # ── Optional reranking ───────────────────────────────────────────────
        if settings.use_reranker and settings.enable_reranking and len(fused) > top_k:
            fused = self._rerank(query, fused, top_k=top_k * 2)

        final = fused[:top_k]

        # Deduplicate by chunk_id
        seen = set()
        unique = []
        for chunk in final:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique.append(chunk)

        method = "hybrid_bm25_dense_rrf"
        if settings.use_reranker:
            method += "_reranked"

        logger.info("Retrieval complete", final_chunks=len(unique), method=method)
        return RetrievedContext(chunks=unique[:top_k], retrieval_method=method)

    def _rerank(self, query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
        """Rerank using MiniLM cross-encoder."""
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder reranker", model=settings.reranker_model)
                self._reranker = CrossEncoder(settings.reranker_model, device="cpu")

            pairs = [[query, chunk.text] for chunk in chunks]
            scores = self._reranker.predict(pairs)

            for chunk, score in zip(chunks, scores):
                chunk.score = float(score)

            chunks.sort(key=lambda c: c.score, reverse=True)
            logger.debug("Reranking complete", chunks_in=len(chunks))
            return chunks[:top_k]

        except Exception as e:
            logger.warning("Reranker failed, using fusion scores", error=str(e))
            return chunks[:top_k]
