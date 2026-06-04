"""
backend/api/routes/query.py
────────────────────────────
Query processing endpoint — runs the full LangGraph agent pipeline.
"""
from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException
from backend.api.schemas import QueryRequest, QueryResponse
from backend.agents.graph import run_query
from backend.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Process a natural language query through the multi-agent pipeline.
    
    Routes to the appropriate agent based on query intent:
    - QA: returns answer with citations and confidence score
    - Extraction: returns structured JSON (parties, dates, obligations)
    - Compliance: returns GDPR/EU AI Act compliance report
    """
    start = time.time()
    logger.info("Query received", query=request.query[:100], doc_id=request.doc_id)

    try:
        result = run_query(
            query=request.query,
            doc_id=request.doc_id,
            doc_name=request.doc_name,
        )
    except Exception as e:
        logger.error("Query processing failed", error=str(e), exc_info=True)
        raise HTTPException(500, f"Query processing failed: {e}")

    elapsed_ms = int((time.time() - start) * 1000)
    response_type = result.get("type", "unknown")

    logger.info("Query complete", type=response_type, elapsed_ms=elapsed_ms)

    return QueryResponse(
        type=response_type,
        data=result,
        query=request.query,
        intent=response_type,
        processing_time_ms=elapsed_ms,
    )
