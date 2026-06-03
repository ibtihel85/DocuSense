"""
backend/core/bm25_index.py
──────────────────────────
BM25 sparse retrieval index using rank-bm25.
Persisted to disk as pickle. Rebuilt on new document ingestion.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi
from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    import re
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class BM25Index:
    """
    BM25 sparse retrieval index.
    Stores a mapping from token positions back to chunk metadata.
    """

    def __init__(self, index_path: str | None = None) -> None:
        self.index_path = Path(index_path or settings.bm25_index_path)
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadata: list[dict[str, Any]] = []

        if self.index_path.exists():
            self._load()

    def build(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Build the BM25 index from scratch."""
        logger.info("Building BM25 index", chunk_count=len(texts))
        self._chunk_ids = chunk_ids
        self._chunk_texts = texts
        self._chunk_metadata = metadatas
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._save()
        logger.info("BM25 index built and saved")

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Append new chunks and rebuild index."""
        self._chunk_ids.extend(chunk_ids)
        self._chunk_texts.extend(texts)
        self._chunk_metadata.extend(metadatas)
        tokenized = [_tokenize(t) for t in self._chunk_texts]
        self._bm25 = BM25Okapi(tokenized)
        self._save()

    def search(self, query: str, top_k: int = 10, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Search for top-k chunks matching the query.
        Returns list of dicts with chunk_id, text, score, metadata.
        """
        if self._bm25 is None or not self._chunk_ids:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair scores with indices, apply doc_id filter
        indexed = [(i, float(s)) for i, s in enumerate(scores)]
        if doc_ids:
            doc_id_set = set(doc_ids)
            indexed = [
                (i, s) for i, s in indexed
                if self._chunk_metadata[i].get("doc_id", "") in doc_id_set
            ]

        # Sort by score descending
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:top_k]

        results = []
        max_score = top[0][1] if top else 1.0
        for idx, score in top:
            # Normalize score to 0-1
            normalized = score / max_score if max_score > 0 else 0.0
            results.append({
                "chunk_id": self._chunk_ids[idx],
                "text": self._chunk_texts[idx],
                "score": normalized,
                "metadata": self._chunk_metadata[idx],
            })
        return results

    def remove_document(self, doc_id: str) -> None:
        """Remove all chunks for a document and rebuild."""
        keep = [i for i, m in enumerate(self._chunk_metadata) if m.get("doc_id") != doc_id]
        self._chunk_ids = [self._chunk_ids[i] for i in keep]
        self._chunk_texts = [self._chunk_texts[i] for i in keep]
        self._chunk_metadata = [self._chunk_metadata[i] for i in keep]
        if self._chunk_ids:
            tokenized = [_tokenize(t) for t in self._chunk_texts]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None
        self._save()

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunk_ids": self._chunk_ids,
                "chunk_texts": self._chunk_texts,
                "chunk_metadata": self._chunk_metadata,
            }, f)

    def _load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunk_ids = data["chunk_ids"]
            self._chunk_texts = data["chunk_texts"]
            self._chunk_metadata = data["chunk_metadata"]
            logger.info("BM25 index loaded", chunk_count=len(self._chunk_ids))
        except Exception as e:
            logger.warning("Failed to load BM25 index", error=str(e))

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)


from functools import lru_cache

@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    return BM25Index()
