"""
frontend/app.py
───────────────
DocuSense Streamlit UI — professional document intelligence interface.
Connects to the FastAPI backend via HTTP.
"""
from __future__ import annotations
import json
import time
import httpx
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuSense",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api/v1"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .metric-card {
        background: #1e2130; border-radius: 8px;
        padding: 16px; margin: 8px 0;
        border-left: 4px solid #4f8ef7;
    }
    .citation-card {
        background: #1a1f2e; border-radius: 6px;
        padding: 12px; margin: 6px 0;
        border-left: 3px solid #38d9a9;
        font-size: 0.85em;
    }
    .compliance-compliant { border-left: 4px solid #38d9a9 !important; }
    .compliance-non_compliant { border-left: 4px solid #ff6b6b !important; }
    .compliance-unclear { border-left: 4px solid #ffd43b !important; }
    .status-badge-compliant { background: #2d6a4f; color: #52b788; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
    .status-badge-non_compliant { background: #6b2737; color: #ff8fa3; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
    .status-badge-unclear { background: #6b5c00; color: #ffd43b; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 10) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, json_data: dict = None, files=None, timeout: int = 180) -> dict | None:
    try:
        if files:
            r = httpx.post(f"{API_BASE}{path}", files=files, timeout=timeout)
        else:
            r = httpx.post(f"{API_BASE}{path}", json=json_data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("# ⚖️ DocuSense")
        st.markdown("*Agentic Document Intelligence*")
        st.divider()

        # System health
        health = api_get("/health", timeout=5)
        if health:
            ollama_ok = health.get("ollama_available", False)
            st.markdown(f"**Ollama:** {'🟢' if ollama_ok else '🔴'} `{health.get('ollama_model', 'N/A')}`")
            st.markdown(f"**Chunks indexed:** `{health.get('vector_store_chunks', 0)}`")
        else:
            st.warning("⚠️ Backend not reachable")

        st.divider()

        # Document selector
        st.markdown("### 📂 Documents")
        docs_data = api_get("/documents", timeout=5) or []
        doc_options = {"All documents": None}
        for d in docs_data:
            label = d.get("doc_name", d.get("doc_id", "Unknown"))[:40]
            doc_options[label] = d.get("doc_id")

        selected_label = st.selectbox("Filter to document:", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_label]
        selected_doc_name = selected_label if selected_label != "All documents" else ""

        st.divider()
        st.markdown("### 📤 Upload Document")
        uploaded = st.file_uploader(
            "PDF, DOCX, TXT, or scanned image",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        )
        if uploaded and st.button("🚀 Ingest Document", type="primary"):
            with st.spinner("Processing document..."):
                result = api_post(
                    "/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
                if result and result.get("success"):
                    st.success(f"✅ Ingested: {result['chunk_count']} chunks")
                    if result.get("ocr_used"):
                        st.info("🔍 OCR was used for scanned content")
                    st.rerun()

        if docs_data:
            st.divider()
            st.markdown("### 🗑️ Delete Document")
            del_options = {d.get("doc_name", d.get("doc_id")): d.get("doc_id") for d in docs_data}
            del_label = st.selectbox("Select document:", list(del_options.keys()), key="del_select")
            if st.button("Delete", type="secondary"):
                doc_id = del_options[del_label]
                result = httpx.delete(f"{API_BASE}/documents/{doc_id}", timeout=10)
                if result.status_code == 200:
                    st.success("Deleted")
                    st.rerun()

    return selected_doc_id, selected_doc_name


# ── Response renderers ────────────────────────────────────────────────────────

def render_qa_response(data: dict):
    answer = data.get("answer", "")
    citations = data.get("citations", [])
    confidence = data.get("confidence", 0.0)
    chunks_used = data.get("chunks_used", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{confidence:.0%}")
    col2.metric("Chunks Used", chunks_used)
    col3.metric("Citations", len(citations))

    st.markdown("### 💬 Answer")
    st.markdown(answer)

    if citations:
        st.markdown("### 📎 Citations")
        for i, cit in enumerate(citations, 1):
            page_info = f"Page {cit['page_number']}" if cit.get("page_number") else "Page N/A"
            st.markdown(f"""<div class="citation-card">
<b>[{i}] {cit['doc_name']}</b> · Chunk {cit['chunk_index']} · {page_info}<br>
<i>{cit.get('excerpt', '')[:200]}</i>
</div>""", unsafe_allow_html=True)


def render_extraction_response(data: dict):
    extracted = data.get("data", {})

    st.markdown("### 📋 Extracted Data")

    tabs = st.tabs(["👥 Parties", "📅 Dates", "📌 Obligations", "💰 Monetary", "🔚 Termination", "🔧 Raw JSON"])

    with tabs[0]:
        parties = extracted.get("parties", [])
        if parties:
            for p in parties:
                st.markdown(f"**{p.get('name', 'Unknown')}** — *{p.get('role', '')}*")
                if p.get("address"):
                    st.caption(p["address"])
        else:
            st.info("No parties extracted")

    with tabs[1]:
        dates = extracted.get("dates", [])
        if dates:
            for d in dates:
                st.markdown(f"**{d.get('type', 'Date')}:** `{d.get('date', '')}`")
                if d.get("description"):
                    st.caption(d["description"])
        else:
            st.info("No dates extracted")

    with tabs[2]:
        obligations = extracted.get("obligations", [])
        if obligations:
            for o in obligations:
                st.markdown(f"**{o.get('party', '')}:** {o.get('obligation', '')}")
                if o.get("deadline"):
                    st.caption(f"Deadline: {o['deadline']}")
        else:
            st.info("No obligations extracted")

    with tabs[3]:
        amounts = extracted.get("monetary_values", [])
        if amounts:
            for a in amounts:
                amt = a.get("amount")
                curr = a.get("currency", "")
                desc = a.get("description", "")
                st.markdown(f"**{desc}:** {f'{amt:,.2f}' if amt else 'N/A'} {curr}")
        else:
            st.info("No monetary values extracted")

    with tabs[4]:
        terms = extracted.get("termination_clauses", [])
        if terms:
            for t in terms:
                st.markdown(f"• {t}")
        else:
            st.info("No termination clauses found")
        if extracted.get("governing_law"):
            st.markdown(f"**Governing Law:** {extracted['governing_law']}")

    with tabs[5]:
        st.json(extracted)


def render_compliance_response(data: dict):
    report = data.get("report", {})
    overall = report.get("overall_status", "unclear")
    framework = report.get("framework", "GDPR")

    status_icons = {"compliant": "✅", "non_compliant": "❌", "unclear": "⚠️"}
    icon = status_icons.get(overall, "⚠️")

    st.markdown(f"### {icon} {framework} Compliance Report")
    st.markdown(f"**Overall Status:** `{overall.upper()}`")
    st.markdown(f"**Summary:** {report.get('summary', '')}")

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Compliant", report.get("compliant_count", 0))
    col2.metric("❌ Non-compliant", report.get("non_compliant_count", 0))
    col3.metric("⚠️ Unclear", report.get("unclear_count", 0))

    st.markdown("### 📋 Findings")
    for finding in report.get("findings", []):
        status = finding.get("status", "unclear")
        css_class = f"compliance-{status}"
        status_badge = f'<span class="status-badge-{status}">{status.upper()}</span>'
        st.markdown(f"""<div class="citation-card {css_class}">
<b>{finding['rule_id']}: {finding['rule_name']}</b> {status_badge}<br>
{finding['explanation']}
{f"<br><i>Recommendation: {finding['recommendation']}</i>" if finding.get('recommendation') else ''}
</div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    doc_id, doc_name = render_sidebar()

    st.title("⚖️ DocuSense")
    st.caption("Agentic Document Intelligence for Legal & Compliance Teams")

    # Example queries
    st.markdown("**Example queries:**")
    example_cols = st.columns(3)
    examples = [
        "What are the termination conditions?",
        "Extract all parties and payment terms",
        "Check GDPR Art. 28 compliance",
    ]
    for i, (col, ex) in enumerate(zip(example_cols, examples)):
        if col.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query_input"] = ex

    # Query input
    query = st.text_area(
        "Ask a question about your documents",
        value=st.session_state.get("query_input", ""),
        height=80,
        placeholder="e.g. What are the liability caps in this contract?",
        key="query_box",
    )

    col_btn, col_info = st.columns([1, 3])
    submit = col_btn.button("🔍 Analyze", type="primary", use_container_width=True)
    if doc_id:
        col_info.info(f"🎯 Searching in: **{doc_name}**")
    else:
        col_info.info("🌐 Searching across **all documents**")

    if submit and query.strip():
        st.session_state["query_input"] = query
        with st.spinner("🤔 Analyzing documents..."):
            start = time.time()
            result = api_post("/query", json_data={
                "query": query,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "top_k": 5,
            }, timeout=180)
            elapsed = time.time() - start

        if result:
            st.caption(f"⏱️ Processed in {elapsed:.1f}s · Method: `{result.get('data', {}).get('retrieval_method', 'N/A')}`")
            response_type = result.get("type", "")
            response_data = result.get("data", {})

            if response_type == "qa":
                render_qa_response(response_data)
            elif response_type == "extraction":
                render_extraction_response(response_data)
            elif response_type == "compliance":
                render_compliance_response(response_data)
            elif response_type == "error":
                st.error(f"Error: {response_data.get('message', 'Unknown error')}")
            else:
                st.json(result)

    elif not query.strip() and submit:
        st.warning("Please enter a query")

    # Footer
    st.divider()
    st.caption("DocuSense v1.0 · Fully local · No external APIs · CPU-only")


if __name__ == "__main__":
    main()
