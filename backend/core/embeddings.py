"""
backend/core/embeddings.py
──────────────────────────
Sentence-transformer embeddings wrapper.
Uses BAAI/bge-small-en-v1.5 (CPU-friendly, strong performance).
Provides batch embedding with progress tracking.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for document/query embedding.

    The BGE model prepends a special instruction for queries vs passages:
    - Passage: embedded as-is
    - Query: prefixed with "Represent this sentence: " for asymmetric search
    """

    # BGE instruction prefix (improves retrieval quality)
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device

        logger.info("Loading embedding model", model=self.model_name, device=self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded", dimension=self.dimension)

    def embed_documents(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of document passages (no prefix added).

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once (tune for RAM)
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.debug("Embedding documents", count=len(texts), batch_size=batch_size)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query (adds BGE instruction prefix for bge models).

        Returns:
            numpy array of shape (dimension,)
        """
        # Add BGE query instruction prefix only for BGE models
        if "bge" in self.model_name.lower():
            text = self.QUERY_INSTRUCTION + query
        else:
            text = query

        embedding = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].astype(np.float32)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Embed multiple queries."""
        texts = [
            (self.QUERY_INSTRUCTION + q if "bge" in self.model_name.lower() else q)
            for q in queries
        ]
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Return cached singleton EmbeddingModel instance."""
    return EmbeddingModel()
