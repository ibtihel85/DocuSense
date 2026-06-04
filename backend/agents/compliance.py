"""
backend/agents/compliance.py
─────────────────────────────
Compliance Agent: checks documents against GDPR and EU AI Act rules.
Combines rule-based checks (regex/keyword) with LLM analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import re
from backend.core.llm import OllamaClient, COMPLIANCE_SYSTEM_PROMPT
from backend.agents.retriever import RetrievedContext
from backend.utils.logger import get_logger

logger = get_logger(__name__)

ComplianceStatus = Literal["compliant", "non_compliant", "unclear", "not_applicable"]


@dataclass
class ComplianceFinding:
    rule_id: str
    rule_name: str
    status: ComplianceStatus
    explanation: str
    relevant_text: str | None = None
    recommendation: str | None = None


@dataclass
class ComplianceReport:
    overall_status: ComplianceStatus
    framework: str
    findings: list[ComplianceFinding] = field(default_factory=list)
    summary: str = ""
    compliant_count: int = 0
    non_compliant_count: int = 0
    unclear_count: int = 0
    confidence: float = 0.0


# ── Rule definitions ─────────────────────────────────────────────────────────

GDPR_RULES = [
    {
        "id": "GDPR-6",
        "name": "Lawful basis for processing (Art. 6)",
        "keywords": ["lawful basis", "rechtsgrundlage", "legitimate interest", "consent", "einwilligung",
                     "contract performance", "legal obligation"],
        "required": True,
    },
    {
        "id": "GDPR-13",
        "name": "Transparency / Privacy notice (Art. 13)",
        "keywords": ["privacy notice", "datenschutzerklärung", "privacy policy", "data subject rights",
                     "purpose of processing", "verarbeitungszweck"],
        "required": True,
    },
    {
        "id": "GDPR-28",
        "name": "Data Processing Agreement (Art. 28)",
        "keywords": ["data processor", "auftragsverarbeiter", "processing agreement", "auftragsverarbeitungsvertrag",
                     "subprocessor", "unterauftragnehmer", "instructions of the controller"],
        "required": True,
    },
    {
        "id": "GDPR-32",
        "name": "Security of processing (Art. 32)",
        "keywords": ["technical measures", "technische maßnahmen", "encryption", "verschlüsselung",
                     "pseudonymization", "pseudonymisierung", "access control", "zugangskontrolle"],
        "required": True,
    },
    {
        "id": "GDPR-44",
        "name": "Data transfers to third countries (Art. 44-49)",
        "keywords": ["third country", "drittland", "standard contractual clauses", "scc",
                     "adequacy decision", "angemessenheitsbeschluss", "binding corporate rules"],
        "required": False,
    },
]

EU_AI_ACT_RULES = [
    {
        "id": "EUAIA-9",
        "name": "Risk management system (Art. 9)",
        "keywords": ["risk management", "risikomanagement", "risk assessment", "residual risk",
                     "risk mitigation", "risikoanalyse"],
        "required": True,
    },
    {
        "id": "EUAIA-13",
        "name": "Transparency obligations (Art. 13)",
        "keywords": ["transparency", "transparenz", "explainability", "ai system", "ki-system",
                     "decision-making", "automated decision"],
        "required": True,
    },
    {
        "id": "EUAIA-14",
        "name": "Human oversight (Art. 14)",
        "keywords": ["human oversight", "menschliche aufsicht", "human review", "human-in-the-loop",
                     "override", "intervention"],
        "required": True,
    },
    {
        "id": "EUAIA-17",
        "name": "Quality management system (Art. 17)",
        "keywords": ["quality management", "qualitätsmanagementsystem", "qms", "iso 9001",
                     "conformity assessment", "konformitätsbewertung"],
        "required": False,
    },
]

FRAMEWORKS = {
    "gdpr": GDPR_RULES,
    "eu_ai_act": EU_AI_ACT_RULES,
}


class ComplianceAgent:
    """
    Checks documents against GDPR / EU AI Act compliance rules.
    Uses keyword-based rule matching + LLM analysis for nuanced findings.
    """

    def __init__(self, llm: OllamaClient | None = None) -> None:
        self.llm = llm or OllamaClient()

    def check(
        self,
        context: RetrievedContext,
        framework: str = "gdpr",
        doc_name: str = "",
    ) -> ComplianceReport:
        """
        Run compliance check on retrieved document context.

        Args:
            context: Retrieved document chunks
            framework: "gdpr" or "eu_ai_act"
            doc_name: Document name for reporting

        Returns:
            ComplianceReport with per-rule findings
        """
        rules = FRAMEWORKS.get(framework.lower(), GDPR_RULES)
        full_text = " ".join(c.text for c in context.chunks).lower()

        findings: list[ComplianceFinding] = []

        # ── Rule-based keyword check ─────────────────────────────────────────
        for rule in rules:
            matched_keywords = [k for k in rule["keywords"] if k in full_text]

            if matched_keywords:
                # Found keywords → use LLM to assess quality
                finding = self._llm_assess_rule(rule, context, matched_keywords)
            elif rule["required"]:
                finding = ComplianceFinding(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    status="non_compliant",
                    explanation=f"No evidence found for {rule['name']}. Required keywords absent.",
                    recommendation=f"Ensure the document explicitly addresses {rule['name']}.",
                )
            else:
                finding = ComplianceFinding(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    status="not_applicable",
                    explanation="Not applicable to this document based on content analysis.",
                )

            findings.append(finding)

        # ── Aggregate results ────────────────────────────────────────────────
        compliant = sum(1 for f in findings if f.status == "compliant")
        non_compliant = sum(1 for f in findings if f.status == "non_compliant")
        unclear = sum(1 for f in findings if f.status == "unclear")
        required_rules = [r for r in rules if r["required"]]
        required_findings = [f for f in findings if any(r["id"] == f.rule_id and r["required"] for r in rules)]
        all_required_compliant = all(f.status == "compliant" for f in required_findings)

        if non_compliant > 0:
            overall = "non_compliant"
        elif unclear > 0 and not all_required_compliant:
            overall = "unclear"
        elif compliant >= len(required_rules):
            overall = "compliant"
        else:
            overall = "unclear"

        summary = self._generate_summary(framework, findings, overall)

        report = ComplianceReport(
            overall_status=overall,
            framework=framework.upper().replace("_", " "),
            findings=findings,
            summary=summary,
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            unclear_count=unclear,
            confidence=0.8 if len(context.chunks) >= 3 else 0.5,
        )

        logger.info(
            "Compliance check complete",
            framework=framework,
            overall=overall,
            findings=len(findings),
        )
        return report

    def _llm_assess_rule(
        self,
        rule: dict,
        context: RetrievedContext,
        matched_keywords: list[str],
    ) -> ComplianceFinding:
        """Use LLM to assess whether the found keywords represent genuine compliance."""
        # Use only top 3 chunks for efficiency
        relevant_text = "\n\n".join(c.text[:300] for c in context.chunks[:3])

        prompt = f"""Assess compliance with {rule['id']}: {rule['name']}

Found keywords: {matched_keywords[:3]}

Document excerpt:
{relevant_text}

Determine if the document genuinely complies with {rule['name']}.
Return JSON: {{
  "status": "compliant" | "non_compliant" | "unclear",
  "explanation": "brief explanation",
  "relevant_text": "most relevant sentence or null",
  "recommendation": "what to fix if non-compliant, or null"
}}"""

        try:
            result = self.llm.generate_json(prompt, system=COMPLIANCE_SYSTEM_PROMPT)
            status = result.get("status", "unclear")
            if status not in ("compliant", "non_compliant", "unclear", "not_applicable"):
                status = "unclear"

            return ComplianceFinding(
                rule_id=rule["id"],
                rule_name=rule["name"],
                status=status,
                explanation=result.get("explanation", "LLM assessment"),
                relevant_text=result.get("relevant_text"),
                recommendation=result.get("recommendation"),
            )
        except Exception as e:
            logger.warning("LLM compliance assessment failed", rule=rule["id"], error=str(e))
            return ComplianceFinding(
                rule_id=rule["id"],
                rule_name=rule["name"],
                status="unclear",
                explanation=f"Keywords found but could not assess quality: {e}",
            )

    def _generate_summary(self, framework: str, findings: list[ComplianceFinding], overall: str) -> str:
        """Generate a human-readable compliance summary."""
        non_compliant = [f for f in findings if f.status == "non_compliant"]
        if overall == "compliant":
            return f"The document appears to comply with all required {framework.upper()} provisions."
        elif non_compliant:
            issues = "; ".join(f.rule_name for f in non_compliant[:3])
            return f"Compliance issues identified: {issues}. Review and remediation required."
        else:
            return f"Compliance status is unclear for some {framework.upper()} provisions. Manual review recommended."
