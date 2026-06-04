"""
backend/agents/graph.py
────────────────────────
LangGraph orchestration: wires all agents into a directed graph.
The Supervisor routes each query to the correct agent(s).
"""
from __future__ import annotations
from typing import Any, TypedDict
from langgraph.graph import StateGraph, END
from backend.agents.supervisor import SupervisorAgent
from backend.agents.retriever import RetrieverAgent
from backend.agents.qa_agent import QAAgent
from backend.agents.extraction import ExtractionAgent
from backend.agents.compliance import ComplianceAgent
from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    query: str
    doc_id: str | None          # Optional: filter to specific document
    doc_name: str               # For reporting
    intent: str                 # Classified query intent
    intent_confidence: float
    # Retrieval
    retrieved_chunks: list[dict]
    retrieval_method: str
    # Responses (only one is populated per run)
    qa_answer: str
    qa_citations: list[dict]
    qa_confidence: float
    extraction_result: dict
    compliance_report: dict
    # Final
    final_response: dict
    error: str | None


def _make_initial_state(query: str, doc_id: str | None = None, doc_name: str = "") -> AgentState:
    return AgentState(
        query=query, doc_id=doc_id, doc_name=doc_name,
        intent="", intent_confidence=0.0,
        retrieved_chunks=[], retrieval_method="",
        qa_answer="", qa_citations=[], qa_confidence=0.0,
        extraction_result={}, compliance_report={},
        final_response={}, error=None,
    )


# ── Node functions ────────────────────────────────────────────────────────────

def node_supervise(state: AgentState) -> AgentState:
    """Classify query intent."""
    agent = SupervisorAgent()
    result = agent.classify(state["query"], doc_id=state.get("doc_id"))
    state["intent"] = result.intent
    state["intent_confidence"] = result.confidence
    logger.info("Supervisor classified query", intent=result.intent, confidence=result.confidence)
    return state


def node_retrieve(state: AgentState) -> AgentState:
    """Hybrid retrieval."""
    agent = RetrieverAgent()
    doc_ids = [state["doc_id"]] if state.get("doc_id") else None
    context = agent.retrieve(state["query"], doc_ids=doc_ids)
    state["retrieved_chunks"] = context.to_list()
    state["retrieval_method"] = context.retrieval_method
    return state


def node_qa(state: AgentState) -> AgentState:
    """QA / synthesis."""
    from backend.agents.retriever import RetrievedContext
    from backend.core.vectorstore import DocumentChunk
    agent = QAAgent()
    # Reconstruct RetrievedContext from state
    chunks = [
        DocumentChunk(
            chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"],
            doc_name=c["doc_name"], chunk_index=c["chunk_index"],
            page_number=c.get("page_number"), score=c.get("score", 0.5),
        )
        for c in state["retrieved_chunks"]
    ]
    context = RetrievedContext(chunks=chunks, retrieval_method=state["retrieval_method"])
    response = agent.answer(state["query"], context)
    state["qa_answer"] = response.answer
    state["qa_citations"] = [
        {"doc_name": c.doc_name, "chunk_index": c.chunk_index,
         "page_number": c.page_number, "excerpt": c.excerpt}
        for c in response.citations
    ]
    state["qa_confidence"] = response.confidence
    return state


def node_extract(state: AgentState) -> AgentState:
    """Structured extraction."""
    from backend.agents.retriever import RetrievedContext
    from backend.core.vectorstore import DocumentChunk
    agent = ExtractionAgent()
    chunks = [
        DocumentChunk(
            chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"],
            doc_name=c["doc_name"], chunk_index=c["chunk_index"],
            page_number=c.get("page_number"), score=c.get("score", 0.5),
        )
        for c in state["retrieved_chunks"]
    ]
    context = RetrievedContext(chunks=chunks, retrieval_method=state["retrieval_method"])
    result = agent.extract(context, doc_name=state.get("doc_name", ""))
    import dataclasses
    state["extraction_result"] = dataclasses.asdict(result)
    return state


def node_compliance(state: AgentState) -> AgentState:
    """Compliance check."""
    from backend.agents.retriever import RetrievedContext
    from backend.core.vectorstore import DocumentChunk
    import dataclasses, re

    agent = ComplianceAgent()
    chunks = [
        DocumentChunk(
            chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"],
            doc_name=c["doc_name"], chunk_index=c["chunk_index"],
            page_number=c.get("page_number"), score=c.get("score", 0.5),
        )
        for c in state["retrieved_chunks"]
    ]
    context = RetrievedContext(chunks=chunks, retrieval_method=state["retrieval_method"])

    # Determine framework from query
    query_lower = state["query"].lower()
    framework = "eu_ai_act" if any(k in query_lower for k in ["ai act", "ki-verordnung", "euaia"]) else "gdpr"

    report = agent.check(context, framework=framework, doc_name=state.get("doc_name", ""))
    # Convert dataclass findings to dicts
    report_dict = {
        "overall_status": report.overall_status,
        "framework": report.framework,
        "summary": report.summary,
        "compliant_count": report.compliant_count,
        "non_compliant_count": report.non_compliant_count,
        "unclear_count": report.unclear_count,
        "confidence": report.confidence,
        "findings": [dataclasses.asdict(f) for f in report.findings],
    }
    state["compliance_report"] = report_dict
    return state


def node_finalize(state: AgentState) -> AgentState:
    """Package final response based on intent."""
    intent = state["intent"]
    if intent in ("multi_document_qa", "single_document_qa"):
        state["final_response"] = {
            "type": "qa",
            "answer": state["qa_answer"],
            "citations": state["qa_citations"],
            "confidence": state["qa_confidence"],
            "retrieval_method": state["retrieval_method"],
            "chunks_used": len(state["retrieved_chunks"]),
        }
    elif intent == "structured_extraction":
        state["final_response"] = {
            "type": "extraction",
            "data": state["extraction_result"],
            "retrieval_method": state["retrieval_method"],
        }
    elif intent == "compliance_check":
        state["final_response"] = {
            "type": "compliance",
            "report": state["compliance_report"],
            "retrieval_method": state["retrieval_method"],
        }
    else:
        state["final_response"] = {"type": "error", "message": f"Unknown intent: {intent}"}
    return state


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_retrieve(state: AgentState) -> str:
    intent = state["intent"]
    if intent in ("multi_document_qa", "single_document_qa"):
        return "qa"
    elif intent == "structured_extraction":
        return "extract"
    elif intent == "compliance_check":
        return "compliance"
    return "qa"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Construct and compile the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("supervise", node_supervise)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("qa", node_qa)
    graph.add_node("extract", node_extract)
    graph.add_node("compliance", node_compliance)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("supervise")
    graph.add_edge("supervise", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"qa": "qa", "extract": "extract", "compliance": "compliance"},
    )

    graph.add_edge("qa", "finalize")
    graph.add_edge("extract", "finalize")
    graph.add_edge("compliance", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Singleton compiled graph
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, doc_id: str | None = None, doc_name: str = "") -> dict[str, Any]:
    """
    Main entry point: run the full agent pipeline for a query.
    Returns the final_response dict.
    """
    graph = get_graph()
    initial = _make_initial_state(query, doc_id=doc_id, doc_name=doc_name)
    try:
        final_state = graph.invoke(initial)
        return final_state["final_response"]
    except Exception as e:
        logger.error("Graph execution failed", error=str(e), exc_info=True)
        return {"type": "error", "message": str(e)}
