"""
backend/api/schemas.py
───────────────────────
Pydantic models for all API request/response types.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestionResponse(BaseModel):
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


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    file_type: str
    ingested_at: str
    chunk_count: int | None = None


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    doc_id: str | None = Field(None, description="Filter to specific document")
    doc_name: str = Field("", description="Document name for reporting")
    top_k: int = Field(5, ge=1, le=20)


class CitationModel(BaseModel):
    doc_name: str
    chunk_index: int
    page_number: int | None
    excerpt: str


class QAResponseModel(BaseModel):
    type: Literal["qa"] = "qa"
    answer: str
    citations: list[CitationModel]
    confidence: float
    retrieval_method: str
    chunks_used: int


class ExtractionResponseModel(BaseModel):
    type: Literal["extraction"] = "extraction"
    data: dict[str, Any]
    retrieval_method: str


class ComplianceFindingModel(BaseModel):
    rule_id: str
    rule_name: str
    status: str
    explanation: str
    relevant_text: str | None = None
    recommendation: str | None = None


class ComplianceReportModel(BaseModel):
    overall_status: str
    framework: str
    summary: str
    compliant_count: int
    non_compliant_count: int
    unclear_count: int
    confidence: float
    findings: list[ComplianceFindingModel]


class ComplianceResponseModel(BaseModel):
    type: Literal["compliance"] = "compliance"
    report: ComplianceReportModel
    retrieval_method: str


class QueryResponse(BaseModel):
    """Union response — one of qa/extraction/compliance."""
    type: str
    data: dict[str, Any]
    query: str
    intent: str | None = None
    processing_time_ms: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    ollama_model: str
    vector_store_chunks: int
    bm25_chunks: int
    version: str = "1.0.0"
