"""
backend/core/llm.py
────────────────────
Ollama LLM client wrapper.
Provides synchronous and async interfaces with retry logic,
timeout handling, and structured prompt templates.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class OllamaClient:
    """
    Thin wrapper around Ollama's HTTP API.
    
    Supports:
    - Simple text completion
    - JSON-mode responses (structured extraction)
    - Async streaming
    - Automatic retry on transient failures
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.temperature = temperature if temperature is not None else settings.ollama_temperature
        self.timeout = timeout or settings.ollama_timeout

    # ── Health Check ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check whether Ollama service is running."""
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return list of locally available Ollama models."""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Synchronous Generation ───────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a text response from Ollama.

        Args:
            prompt: The user prompt
            system: Optional system message
            json_mode: If True, instruct model to return valid JSON

        Returns:
            Generated text string
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 2048,
            },
        }

        if json_mode:
            payload["format"] = "json"
            # Reinforce JSON instruction in prompt
            if messages:
                messages[-1]["content"] += "\n\nRespond ONLY with valid JSON. No markdown, no explanation."

        logger.debug(
            "Calling Ollama",
            model=self.model,
            prompt_len=len(prompt),
            json_mode=json_mode,
        )

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["message"]["content"]
        logger.debug("Ollama response received", response_len=len(content))
        return content

    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """
        Generate a structured JSON response.

        Returns:
            Parsed dict, or empty dict on parse failure.
        """
        from backend.utils.helpers import safe_json_loads

        text = self.generate(prompt, system=system, json_mode=True)
        result = safe_json_loads(text)
        if result is None:
            logger.warning("Failed to parse JSON from LLM response", raw_response=text[:200])
            return {}
        return result

    # ── Async Generation ─────────────────────────────────────────────────────

    async def agenerate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Async version of generate()."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 2048},
        }

        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data["message"]["content"]

    async def agenerate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Async JSON generation."""
        from backend.utils.helpers import safe_json_loads

        text = await self.agenerate(prompt, system=system, json_mode=True)
        result = safe_json_loads(text)
        return result if result is not None else {}


# ── Prompt Templates ─────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are DocuSense, an expert legal and compliance document analyst.
Your role is to answer questions based strictly on the provided document excerpts.

Rules:
1. Answer ONLY based on the provided context. Do not use external knowledge.
2. Always cite the exact document and chunk you used (use [Doc: X, Chunk: Y] format).
3. If the context does not contain enough information, say so explicitly.
4. For legal/compliance questions, be precise and conservative.
5. Respond in the same language the user asked in (German or English).
6. Include a confidence score (0.0–1.0) based on context completeness."""

EXTRACTION_SYSTEM_PROMPT = """You are a legal document data extraction specialist.
Extract structured information from the provided document text.
Always return valid JSON matching the requested schema.
If a field cannot be found, use null. Never invent information."""

COMPLIANCE_SYSTEM_PROMPT = """You are a GDPR and EU AI Act compliance expert.
Analyze the provided document text against the specified compliance rules.
Be precise, cite specific articles and clauses, and clearly distinguish between:
- Compliant aspects
- Non-compliant aspects  
- Unclear/requires human review
Return your analysis in JSON format."""

CLASSIFICATION_SYSTEM_PROMPT = """You are a query routing system for a document intelligence platform.
Classify the user's query into exactly one of these categories:
- multi_document_qa: Questions spanning multiple documents
- single_document_qa: Questions about a specific document
- compliance_check: GDPR, EU AI Act, or regulatory compliance questions
- structured_extraction: Extracting specific data points (parties, dates, amounts)
Return JSON with keys: intent, confidence, reasoning"""
