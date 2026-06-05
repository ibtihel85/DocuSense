"""
evaluation/metrics.py
──────────────────────
Local evaluation metrics for RAG quality assessment.
No external APIs required — uses cosine similarity + heuristics.
"""
from __future__ import annotations
import re
from typing import Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
    """
    Heuristic faithfulness: checks that answer sentences have
    high lexical overlap with the retrieved context.
    Range: 0.0 (hallucinated) to 1.0 (fully grounded).
    """
    if not answer or not context_chunks:
        return 0.0

    context_text = " ".join(context_chunks).lower()
    context_words = set(re.findall(r'\b\w{4,}\b', context_text))

    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return 0.5

    scores = []
    for sent in sentences:
        sent_words = set(re.findall(r'\b\w{4,}\b', sent.lower()))
        if not sent_words:
            continue
        overlap = len(sent_words & context_words) / len(sent_words)
        scores.append(overlap)

    return float(np.mean(scores)) if scores else 0.0


def context_precision(retrieved_chunks: list[str], relevant_chunks: list[str]) -> float:
    """
    What fraction of retrieved chunks are actually relevant?
    Uses exact match on chunk IDs or text similarity.
    """
    if not retrieved_chunks:
        return 0.0
    relevant_set = set(relevant_chunks)
    hits = sum(1 for c in retrieved_chunks if c in relevant_set)
    return hits / len(retrieved_chunks)


def context_recall(retrieved_chunks: list[str], relevant_chunks: list[str]) -> float:
    """What fraction of relevant chunks were retrieved?"""
    if not relevant_chunks:
        return 1.0
    relevant_set = set(relevant_chunks)
    hits = sum(1 for c in relevant_set if c in retrieved_chunks)
    return hits / len(relevant_chunks)


def citation_correctness_score(citations: list[dict], context_chunks: list[dict]) -> float:
    """
    Check that cited chunk_index values actually exist in retrieved chunks.
    """
    if not citations:
        return 0.0

    context_keys = {(c.get("doc_name", ""), c.get("chunk_index", -1)) for c in context_chunks}
    valid = sum(
        1 for cit in citations
        if (cit.get("doc_name", ""), cit.get("chunk_index", -999)) in context_keys
    )
    return valid / len(citations)


def semantic_similarity(text1: str, text2: str, embedder) -> float:
    """Cosine similarity between two texts using the embedding model."""
    emb1 = embedder.embed_documents([text1])
    emb2 = embedder.embed_documents([text2])
    return float(cosine_similarity(emb1, emb2)[0][0])


def answer_relevance(query: str, answer: str, embedder) -> float:
    """How relevant is the answer to the query? (semantic similarity)"""
    return semantic_similarity(query, answer, embedder)
