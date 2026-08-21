"""Sanitized report writers for provider-free PostFlow product validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

HUMAN_REVIEW_COLUMNS = (
    "case_id",
    "family",
    "topic",
    "direction",
    "research_summary",
    "final_post_or_candidate",
    "expected_outcome",
    "historical_outcome",
    "reviewer_label",
    "reviewer_notes",
    "backlog_action",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def write_cases_csv(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    columns = (
        "case_id",
        "family",
        "topic",
        "risk_type",
        "expected_outcome",
        "failure_category",
        "historical_outcome",
        "fixture_valid",
        "human_review_required",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in columns})


def write_human_review_csv(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record.get("case_id", ""),
                    "family": record.get("family", ""),
                    "topic": record.get("topic", ""),
                    "direction": record.get("risk_type", ""),
                    "research_summary": "",
                    "final_post_or_candidate": "",
                    "expected_outcome": record.get("expected_outcome", ""),
                    "historical_outcome": record.get("historical_outcome", ""),
                    "reviewer_label": "",
                    "reviewer_notes": "",
                    "backlog_action": "",
                }
            )


def build_product_validation_report_markdown(
    *,
    experiment_id: str,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    baseline_comparison: dict[str, Any] | None,
) -> str:
    lines = [
        "# PostFlow Product Validation Benchmark",
        "",
        f"experiment_id: `{experiment_id}`",
        f"corpus_id: `{manifest['corpus_id']}`",
        f"case_count: `{metrics['case_count']}`",
        "provider_call_count: `0`",
        "",
        "## Outcome Taxonomy",
    ]
    for outcome, count in metrics["outcome_counts"].items():
        lines.append(f"- `{outcome}`: {count}")
    lines.extend(["", "## Infrastructure vs Product Failure"])
    for category, count in metrics["failure_category_counts"].items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Human Review", f"- cases: {metrics['human_review_case_count']}"])
    lines.extend(["", "## Baseline Comparison"])
    lines.append(
        json.dumps(baseline_comparison, ensure_ascii=False, sort_keys=True)
        if baseline_comparison is not None
        else "No baseline comparison supplied."
    )
    lines.extend(
        [
            "",
            "## Decision Framework",
            "Use this report to decide whether product failures belong in the backlog, "
            "manual review, or a future isolated architecture scope. Do not treat this "
            "benchmark as production acceptance logic.",
            "",
        ]
    )
    return "\n".join(lines)


def write_human_review_markdown(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    lines = ["# PostFlow Human Review Worksheet", ""]
    for record in records:
        lines.extend(
            [
                f"## {record.get('case_id', '')}",
                f"- family: `{record.get('family', '')}`",
                f"- topic: `{record.get('topic', '')}`",
                f"- expected_outcome: `{record.get('expected_outcome', '')}`",
                "- reviewer_label:",
                "- reviewer_notes:",
                "- backlog_action:",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
