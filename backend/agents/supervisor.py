"""
backend/agents/supervisor.py
─────────────────────────────
Supervisor Agent: classifies query intent and routes to the
appropriate downstream agent(s).

Intent categories:
  - multi_document_qa
  - single_document_qa
  - compliance_check
  - structured_extraction
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from backend.core.llm import OllamaClient, CLASSIFICATION_SYSTEM_PROMPT
from backend.utils.logger import get_logger

logger = get_logger(__name__)

QueryIntent = Literal[
    "multi_document_qa",
    "single_document_qa",
    "compliance_check",
    "structured_extraction",
]


class ClassificationResult(BaseModel):
    intent: QueryIntent
    confidence: float
    reasoning: str
    target_doc_id: str | None = None  # For single-doc queries


# Keyword-based fast path (saves LLM call for clear cases)
_KEYWORD_RULES: list[tuple[list[str], QueryIntent]] = [
    (["gdpr", "dsgvo", "eu ai act", "compliance", "compliant", "regulation", "article", "recital",
      "data protection", "datenschutz", "konform"], "compliance_check"),
    (["extract", "list all", "find all", "parties", "dates", "obligations", "payment terms",
      "monetary", "amount", "deadline", "extrahiere", "alle parteien"], "structured_extraction"),
    (["compare", "across documents", "all contracts", "multiple", "which documents",
      "any contract", "all agreements"], "multi_document_qa"),
]


class SupervisorAgent:
    """
    Routes incoming queries to the correct agent.
    Uses keyword heuristics first, falls back to LLM classification.
    """

    def __init__(self, llm: OllamaClient | None = None) -> None:
        self.llm = llm or OllamaClient()

    def classify(self, query: str, doc_id: str | None = None) -> ClassificationResult:
        """
        Classify the query intent.

        Args:
            query: User's natural language query
            doc_id: If a specific document is selected in the UI

        Returns:
            ClassificationResult with intent and metadata
        """
        # Fast path: if doc_id is provided, assume single-doc QA unless overridden
        if doc_id:
            # But still check for compliance/extraction keywords
            q_lower = query.lower()
            for keywords, intent in _KEYWORD_RULES:
                if any(k in q_lower for k in keywords):
                    return ClassificationResult(
                        intent=intent,
                        confidence=0.9,
                        reasoning=f"Keyword match for '{intent}'",
                        target_doc_id=doc_id,
                    )
            return ClassificationResult(
                intent="single_document_qa",
                confidence=0.95,
                reasoning="Specific document selected by user",
                target_doc_id=doc_id,
            )

        # Keyword fast path
        q_lower = query.lower()
        for keywords, intent in _KEYWORD_RULES:
            if any(k in q_lower for k in keywords):
                logger.debug("Keyword classification", intent=intent, query=query[:50])
                return ClassificationResult(
                    intent=intent,
                    confidence=0.88,
                    reasoning=f"Keyword match: {[k for k in keywords if k in q_lower][:2]}",
                )

        # LLM classification fallback
        return self._llm_classify(query)

    def _llm_classify(self, query: str) -> ClassificationResult:
        """Use LLM to classify ambiguous queries."""
        prompt = f"""Classify this query for a legal document intelligence system.

Query: "{query}"

Categories:
- multi_document_qa: spans multiple documents, comparison, "any contract", "all agreements"
- single_document_qa: general question about document content
- compliance_check: GDPR, EU AI Act, regulations, compliance assessment
- structured_extraction: extract specific data like parties, dates, payment terms, amounts

Return JSON: {{"intent": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief reason>"}}"""

        try:
            result = self.llm.generate_json(prompt, system=CLASSIFICATION_SYSTEM_PROMPT)
            intent = result.get("intent", "multi_document_qa")
            # Validate intent
            valid_intents = ["multi_document_qa", "single_document_qa", "compliance_check", "structured_extraction"]
            if intent not in valid_intents:
                intent = "multi_document_qa"

            return ClassificationResult(
                intent=intent,
                confidence=float(result.get("confidence", 0.7)),
                reasoning=result.get("reasoning", "LLM classification"),
            )
        except Exception as e:
            logger.warning("LLM classification failed, defaulting", error=str(e))
            return ClassificationResult(
                intent="multi_document_qa",
                confidence=0.5,
                reasoning="Classification failed, using default",
            )
