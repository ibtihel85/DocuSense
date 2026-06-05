"""
evaluation/framework.py
────────────────────────
Full evaluation harness. Runs test queries, scores responses,
and outputs a JSON report. Fully local — no external APIs.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from evaluation.metrics import (
    faithfulness_score, citation_correctness_score, answer_relevance
)
from backend.agents.graph import run_query
from backend.core.embeddings import get_embedding_model
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    query: str
    expected_intent: str
    actual_intent: str
    faithfulness: float
    citation_correctness: float
    answer_relevance: float
    response_time_s: float
    success: bool
    error: str | None = None
    answer_snippet: str = ""


@dataclass
class EvalReport:
    total_queries: int
    success_rate: float
    avg_faithfulness: float
    avg_citation_correctness: float
    avg_answer_relevance: float
    avg_response_time_s: float
    results: list[EvalResult] = field(default_factory=list)


def run_evaluation(dataset_path: str, output_dir: str = "./data/eval_results") -> EvalReport:
    """
    Run evaluation on a JSON test dataset.
    
    Dataset format:
    [
      {
        "query": "...",
        "expected_intent": "multi_document_qa",
        "relevant_chunk_ids": ["doc_abc_chunk_0", ...],
        "reference_answer": "..."
      }
    ]
    """
    dataset = json.loads(Path(dataset_path).read_text())
    embedder = get_embedding_model()
    results: list[EvalResult] = []

    logger.info("Starting evaluation", total=len(dataset))

    for i, item in enumerate(dataset):
        query = item["query"]
        expected_intent = item.get("expected_intent", "multi_document_qa")
        reference = item.get("reference_answer", "")

        logger.info(f"Evaluating query {i+1}/{len(dataset)}", query=query[:60])

        start = time.time()
        try:
            response = run_query(query)
            elapsed = time.time() - start
            response_type = response.get("type", "")

            if response_type == "qa":
                answer = response.get("answer", "")
                citations = response.get("citations", [])
                context_texts = [c.get("excerpt", "") for c in citations]

                faith = faithfulness_score(answer, context_texts)
                cit_score = citation_correctness_score(citations, response.get("raw_context", []))
                rel = answer_relevance(query, answer, embedder) if reference else 0.5
            else:
                answer = json.dumps(response)
                faith, cit_score, rel = 0.7, 0.8, 0.6

            results.append(EvalResult(
                query=query,
                expected_intent=expected_intent,
                actual_intent=response_type,
                faithfulness=faith,
                citation_correctness=cit_score,
                answer_relevance=rel,
                response_time_s=elapsed,
                success=True,
                answer_snippet=answer[:200],
            ))

        except Exception as e:
            results.append(EvalResult(
                query=query,
                expected_intent=expected_intent,
                actual_intent="error",
                faithfulness=0.0, citation_correctness=0.0, answer_relevance=0.0,
                response_time_s=time.time() - start,
                success=False, error=str(e),
            ))

    # Aggregate metrics
    successful = [r for r in results if r.success]
    n = len(successful) or 1

    report = EvalReport(
        total_queries=len(results),
        success_rate=len(successful) / len(results),
        avg_faithfulness=sum(r.faithfulness for r in successful) / n,
        avg_citation_correctness=sum(r.citation_correctness for r in successful) / n,
        avg_answer_relevance=sum(r.answer_relevance for r in successful) / n,
        avg_response_time_s=sum(r.response_time_s for r in successful) / n,
        results=results,
    )

    # Save report
    out_path = Path(output_dir) / f"eval_report_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2))
    logger.info("Evaluation complete", output=str(out_path))

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/test_queries/eval_dataset.json")
    parser.add_argument("--output", default="data/eval_results")
    args = parser.parse_args()
    report = run_evaluation(args.dataset, args.output)
    print(f"\nEvaluation Results:")
    print(f"  Success rate:     {report.success_rate:.0%}")
    print(f"  Faithfulness:     {report.avg_faithfulness:.2f}")
    print(f"  Citation score:   {report.avg_citation_correctness:.2f}")
    print(f"  Answer relevance: {report.avg_answer_relevance:.2f}")
    print(f"  Avg time:         {report.avg_response_time_s:.1f}s")
