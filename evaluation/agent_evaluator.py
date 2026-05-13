from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "evaluation" / "agent_eval_samples.jsonl"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "evaluation" / "results"
REFERENCE_MARKERS = ("参考", "资料", "知识库", "根据", "检索", "文档", "来源", "【参考")


@dataclass
class EvalSample:
    id: str
    category: str
    query: str
    expected_tools: list[str]
    forbidden_tools: list[str]
    requires_knowledge_reference: bool
    completion_keywords: list[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalSample":
        return cls(
            id=raw["id"],
            category=raw["category"],
            query=raw["query"],
            expected_tools=list(raw.get("expected_tools", [])),
            forbidden_tools=list(raw.get("forbidden_tools", [])),
            requires_knowledge_reference=bool(raw.get("requires_knowledge_reference", False)),
            completion_keywords=list(raw.get("completion_keywords", [])),
        )


def load_samples(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(EvalSample.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid sample at {path}:{line_number}: {exc}") from exc
    _validate_samples(samples)
    return samples


def _validate_samples(samples: list[EvalSample]) -> None:
    if not 30 <= len(samples) <= 50:
        raise ValueError(f"Expected 30-50 samples, got {len(samples)}")

    seen_ids: set[str] = set()
    for sample in samples:
        if sample.id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample.id}")
        seen_ids.add(sample.id)
        if not sample.query:
            raise ValueError(f"Empty query in sample: {sample.id}")


def run_agent(samples: list[EvalSample], limit: int | None = None) -> list[dict[str, Any]]:
    from agent.react_agent import ReactAgent

    agent = ReactAgent()
    predictions: list[dict[str, Any]] = []
    selected_samples = samples[:limit] if limit else samples

    for sample in selected_samples:
        try:
            prediction = agent.evaluate_once(sample.query)
            predictions.append(
                {
                    "id": sample.id,
                    "query": sample.query,
                    "answer": prediction["answer"],
                    "tools_used": prediction["tools_used"],
                    "route": prediction["route"],
                    "error": None,
                }
            )
        except Exception as exc:
            predictions.append(
                {
                    "id": sample.id,
                    "query": sample.query,
                    "answer": "",
                    "tools_used": [],
                    "route": None,
                    "error": repr(exc),
                }
            )
    return predictions


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, dict) and "predictions" in raw:
        return list(raw["predictions"])
    if isinstance(raw, list):
        return list(raw)
    raise ValueError("Prediction file must be a list or contain a 'predictions' list")


def evaluate_predictions(
    samples: list[EvalSample],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_by_id = {sample.id: sample for sample in samples}
    per_sample = []

    for prediction in predictions:
        sample = sample_by_id[prediction["id"]]
        used_tools = set(prediction.get("tools_used") or [])
        expected_tools = set(sample.expected_tools)
        forbidden_tools = set(sample.forbidden_tools)
        answer = prediction.get("answer") or ""
        error = prediction.get("error")

        expected_tools_hit = expected_tools.issubset(used_tools)
        forbidden_tools_clean = forbidden_tools.isdisjoint(used_tools)
        exact_tool_match = used_tools == expected_tools
        tool_call_correct = expected_tools_hit and forbidden_tools_clean
        task_completed = _is_task_completed(answer, sample.completion_keywords, error)
        knowledge_reference_correct = _is_knowledge_reference_correct(
            answer=answer,
            used_tools=used_tools,
            requires_reference=sample.requires_knowledge_reference,
            error=error,
        )

        per_sample.append(
            {
                "id": sample.id,
                "category": sample.category,
                "query": sample.query,
                "expected_tools": sample.expected_tools,
                "forbidden_tools": sample.forbidden_tools,
                "tools_used": sorted(used_tools),
                "tool_call_correct": tool_call_correct,
                "expected_tools_hit": expected_tools_hit,
                "forbidden_tools_clean": forbidden_tools_clean,
                "exact_tool_match": exact_tool_match,
                "task_completed": task_completed,
                "requires_knowledge_reference": sample.requires_knowledge_reference,
                "knowledge_reference_correct": knowledge_reference_correct,
                "answer": answer,
                "error": error,
            }
        )

    return {
        "summary": _summarize(per_sample),
        "per_sample": per_sample,
        "predictions": predictions,
    }


def _is_task_completed(answer: str, keywords: list[str], error: str | None) -> bool:
    if error or not answer.strip():
        return False
    if not keywords:
        return True
    return any(keyword in answer for keyword in keywords)


def _is_knowledge_reference_correct(
    *,
    answer: str,
    used_tools: set[str],
    requires_reference: bool,
    error: str | None,
) -> bool | None:
    if not requires_reference:
        return None
    if error or "rag_summarize" not in used_tools:
        return False
    return any(marker in answer for marker in REFERENCE_MARKERS)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _summarize(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(per_sample)
    reference_samples = [
        row for row in per_sample if row["requires_knowledge_reference"]
    ]
    categories = sorted({row["category"] for row in per_sample})
    by_category = {}

    for category in categories:
        rows = [row for row in per_sample if row["category"] == category]
        by_category[category] = {
            "sample_count": len(rows),
            "tool_call_accuracy": _ratio(sum(row["tool_call_correct"] for row in rows), len(rows)),
            "task_completion_rate": _ratio(sum(row["task_completed"] for row in rows), len(rows)),
        }

    return {
        "sample_count": total,
        "tool_call_accuracy": _ratio(sum(row["tool_call_correct"] for row in per_sample), total),
        "expected_tool_recall": _ratio(sum(row["expected_tools_hit"] for row in per_sample), total),
        "exact_tool_match_rate": _ratio(sum(row["exact_tool_match"] for row in per_sample), total),
        "task_completion_rate": _ratio(sum(row["task_completed"] for row in per_sample), total),
        "knowledge_reference_rate": _ratio(
            sum(row["knowledge_reference_correct"] is True for row in reference_samples),
            len(reference_samples),
        ),
        "knowledge_reference_sample_count": len(reference_samples),
        "by_category": by_category,
    }


def save_report(report: dict[str, Any], result_dir: Path = DEFAULT_RESULT_DIR) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = result_dir / f"agent_eval_{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LangChain ReAct Agent behavior.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--predictions", type=Path, help="Evaluate an existing prediction JSON file.")
    parser.add_argument("--limit", type=int, help="Only run the first N samples.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate and summarize samples.")
    parser.add_argument("--output", type=Path, help="Output JSON path. Defaults to evaluation/results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples)

    if args.dry_run:
        categories = {}
        for sample in samples:
            categories[sample.category] = categories.get(sample.category, 0) + 1
        print_summary(
            {
                "sample_count": len(samples),
                "categories": categories,
                "live_agent_run": False,
            }
        )
        return

    predictions = load_predictions(args.predictions) if args.predictions else run_agent(samples, args.limit)
    report = evaluate_predictions(samples, predictions)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        output_path = args.output
    else:
        output_path = save_report(report)

    print_summary(report["summary"])
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
