"""
backend/agents/qa_agent.py
───────────────────────────
QA / Synthesis Agent: generates answers with chunk-level citations
and confidence scores. Anti-hallucination enforced via prompting.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any
from backend.core.llm import OllamaClient, QA_SYSTEM_PROMPT
from backend.agents.retriever import RetrievedContext
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    doc_name: str
    chunk_index: int
    page_number: int | None
    excerpt: str  # Short excerpt from cited chunk


@dataclass
class QAResponse:
    answer: str
    citations: list[Citation]
    confidence: float
    context_used: int  # Number of chunks used
    retrieval_method: str
    raw_context: list[dict[str, Any]] = field(default_factory=list)


class QAAgent:
    """
    Synthesizes final answers from retrieved context.
    Always includes citations and a confidence score.
    Refuses to answer if context is insufficient.
    """

    def __init__(self, llm: OllamaClient | None = None) -> None:
        self.llm = llm or OllamaClient()

    def answer(self, query: str, context: RetrievedContext) -> QAResponse:
        """
        Generate a cited answer from retrieved context.

        Args:
            query: User question
            context: Retrieved chunks from RetrieverAgent

        Returns:
            QAResponse with answer, citations, and confidence
        """
        if not context.chunks:
            return QAResponse(
                answer="I could not find relevant information in the document collection to answer this question.",
                citations=[],
                confidence=0.0,
                context_used=0,
                retrieval_method=context.retrieval_method,
            )

        # Build context string for LLM
        context_str = context.to_context_string()

        prompt = f"""Answer the following question using ONLY the document excerpts provided below.

QUESTION: {query}

DOCUMENT EXCERPTS:
{context_str}

INSTRUCTIONS:
1. Answer concisely and accurately based ONLY on the above excerpts.
2. For each claim in your answer, add a citation in the format [Doc: <doc_name>, Chunk: <chunk_index>].
3. If the excerpts do not contain enough information, say: "The provided documents do not contain sufficient information to answer this question."
4. At the end, add a line: CONFIDENCE: <score between 0.0 and 1.0>
5. Be precise with legal/compliance language. Do not paraphrase obligations loosely.

ANSWER:"""

        raw_answer = self.llm.generate(prompt, system=QA_SYSTEM_PROMPT)

        # Parse confidence from answer
        confidence = self._extract_confidence(raw_answer)
        clean_answer = self._remove_confidence_line(raw_answer)

        # Parse citations from answer text
        citations = self._extract_citations(clean_answer, context.chunks)

        logger.info(
            "QA answer generated",
            query_len=len(query),
            answer_len=len(clean_answer),
            citations=len(citations),
            confidence=confidence,
        )

        return QAResponse(
            answer=clean_answer,
            citations=citations,
            confidence=confidence,
            context_used=len(context.chunks),
            retrieval_method=context.retrieval_method,
            raw_context=context.to_list(),
        )

    def _extract_confidence(self, text: str) -> float:
        """Extract CONFIDENCE: X.X from LLM output."""
        match = re.search(r'CONFIDENCE:\s*([0-9.]+)', text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        # Heuristic: if answer says "not found" → low confidence
        if any(phrase in text.lower() for phrase in ["not contain", "not find", "insufficient"]):
            return 0.1
        return 0.7

    def _remove_confidence_line(self, text: str) -> str:
        """Remove CONFIDENCE line from answer text."""
        return re.sub(r'\n?CONFIDENCE:\s*[0-9.]+\n?', '', text, flags=re.IGNORECASE).strip()

    def _extract_citations(self, answer: str, chunks) -> list[Citation]:
        """
        Parse [Doc: X, Chunk: Y] citation markers from the answer
        and match them to actual chunks.
        """
        pattern = r'\[Doc:\s*([^,\]]+),\s*Chunk:\s*(\d+)\]'
        matches = re.findall(pattern, answer)

        citations = []
        seen = set()
        chunk_map = {(c.doc_name, c.chunk_index): c for c in chunks}

        for doc_name, chunk_idx_str in matches:
            try:
                chunk_idx = int(chunk_idx_str)
                key = (doc_name.strip(), chunk_idx)
                if key in seen:
                    continue
                seen.add(key)

                # Find matching chunk
                chunk = chunk_map.get(key)
                if chunk:
                    excerpt = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                    citations.append(Citation(
                        doc_name=chunk.doc_name,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        excerpt=excerpt,
                    ))
            except ValueError:
                continue

        return citations
