"""
backend/agents/extraction.py
─────────────────────────────
Extraction Agent: pulls structured JSON from contract/legal documents.
Extracts: parties, dates, obligations, monetary values, termination clauses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from backend.core.llm import OllamaClient, EXTRACTION_SYSTEM_PROMPT
from backend.agents.retriever import RetrievedContext
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedData:
    parties: list[dict] = field(default_factory=list)
    dates: list[dict] = field(default_factory=list)
    obligations: list[dict] = field(default_factory=list)
    monetary_values: list[dict] = field(default_factory=list)
    termination_clauses: list[str] = field(default_factory=list)
    governing_law: str | None = None
    contract_type: str | None = None
    confidence: float = 0.0
    raw_json: dict = field(default_factory=dict)


EXTRACTION_SCHEMA = """
{
  "parties": [{"name": "string", "role": "string (e.g. Buyer, Seller, Processor)", "address": "string or null"}],
  "dates": [{"type": "string (e.g. Effective Date, Expiry)", "date": "string (ISO format or as written)", "description": "string"}],
  "obligations": [{"party": "string", "obligation": "string", "deadline": "string or null", "condition": "string or null"}],
  "monetary_values": [{"description": "string", "amount": "number or null", "currency": "string", "conditions": "string or null"}],
  "termination_clauses": ["string"],
  "governing_law": "string or null",
  "contract_type": "string (e.g. Service Agreement, DPA, NDA, Procurement)"
}
"""


class ExtractionAgent:
    """
    Extracts structured contract data from document chunks.
    Uses LLM with JSON mode for reliable structured output.
    """

    def __init__(self, llm: OllamaClient | None = None) -> None:
        self.llm = llm or OllamaClient()

    def extract(self, context: RetrievedContext, doc_name: str = "") -> ExtractedData:
        """
        Extract structured data from retrieved document context.

        Args:
            context: Retrieved chunks (ideally full document)
            doc_name: Document name for logging

        Returns:
            ExtractedData with structured fields
        """
        if not context.chunks:
            return ExtractedData(confidence=0.0)

        # Use first N chunks for extraction (prioritize beginning of document)
        top_chunks = context.chunks[:8]
        context_text = "\n\n".join(
            f"[Page {c.page_number or '?'}]\n{c.text}" for c in top_chunks
        )

        prompt = f"""Extract structured information from this legal/contract document.

DOCUMENT: {doc_name}

TEXT:
{context_text}

Return JSON matching this exact schema:
{EXTRACTION_SCHEMA}

Rules:
- Extract only what is explicitly stated. Do not infer.
- For amounts, use numbers (e.g., 50000 not "fifty thousand").
- For dates, use the exact text if not in ISO format.
- If a field cannot be found, use null or empty array [].
- Include a "confidence" field (0.0-1.0) at the top level."""

        raw = self.llm.generate_json(prompt, system=EXTRACTION_SYSTEM_PROMPT)

        if not raw:
            logger.warning("Extraction returned empty result", doc=doc_name)
            return ExtractedData(confidence=0.0, raw_json={})

        confidence = float(raw.pop("confidence", 0.6))
        confidence = max(0.0, min(1.0, confidence))

        logger.info(
            "Extraction complete",
            doc=doc_name,
            parties=len(raw.get("parties", [])),
            dates=len(raw.get("dates", [])),
            obligations=len(raw.get("obligations", [])),
            confidence=confidence,
        )

        return ExtractedData(
            parties=raw.get("parties", []),
            dates=raw.get("dates", []),
            obligations=raw.get("obligations", []),
            monetary_values=raw.get("monetary_values", []),
            termination_clauses=raw.get("termination_clauses", []),
            governing_law=raw.get("governing_law"),
            contract_type=raw.get("contract_type"),
            confidence=confidence,
            raw_json=raw,
        )
