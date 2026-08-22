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
    "historical_expected_outcome",
    "historical_classification",
    "historical_notes",
    "live_outcome",
    "live_failure_category",
    "live_failure_stage",
    "live_failure_code",
    "original_candidate",
    "repaired_candidate",
    "final_candidate",
    "repair_target",
    "grounding_summary",
    "qe_scores",
    "failed_criteria",
    "preservation_rate",
    "changed_sentence_count",
    "generic_marker_count",
    "length_before",
    "length_after",
    "infrastructure_failure",
    "human_review_priority",
    "human_final_disposition",
    "human_primary_reason",
    "human_repair_target",
    "human_genericization",
    "human_grounding_fidelity",
    "human_distinctive_voice",
    "human_voice",
    "human_cta",
    "human_author_point_of_view",
    "human_repair_locality",
    "review_confidence",
    "review_status",
    "final_text_quality_1_5",
    "voice_preservation_1_5",
    "genericization_none_mild_material",
    "evidence_fidelity_pass_concern_fail",
    "repair_improvement_worse_neutral_better",
    "repair_was_necessary_yes_no",
    "would_publish_yes_no",
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
        "historical_expected_outcome",
        "live_execution_status",
        "live_outcome",
        "live_failure_category",
        "live_failure_code",
        "live_failure_stage",
        "failure_category",
        "historical_outcome",
        "fixture_valid",
        "human_review_required",
        "provider_call_count",
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
                    "historical_expected_outcome": record.get("historical_expected_outcome", ""),
                    "historical_classification": record.get("historical_classification", ""),
                    "historical_notes": record.get("historical_notes", ""),
                    "live_outcome": record.get("live_outcome", ""),
                    "live_failure_category": record.get("live_failure_category", ""),
                    "live_failure_stage": record.get("live_failure_stage", ""),
                    "live_failure_code": record.get("live_failure_code", ""),
                    "original_candidate": _candidate_text(record.get("final_candidate")),
                    "repaired_candidate": _candidate_text(record.get("repaired_candidate")),
                    "final_candidate": _candidate_text(record.get("accepted_payload") or record.get("final_candidate")),
                    "repair_target": (record.get("product_case_distribution") or {}).get("repair_target", ""),
                    "grounding_summary": _json_cell((record.get("live_stage_outcomes") or {}).get("semantic_grounding")),
                    "qe_scores": _json_cell((record.get("quality_review_summary") or {}).get("scores")),
                    "failed_criteria": _json_cell((record.get("quality_review_summary") or {}).get("failed_criteria")),
                    "preservation_rate": record.get("preservation_rate", ""),
                    "changed_sentence_count": record.get("changed_sentence_count", ""),
                    "generic_marker_count": record.get("generic_marker_count", ""),
                    "length_before": (record.get("product_case_distribution") or {}).get("length", ""),
                    "length_after": len(_candidate_text(record.get("accepted_payload") or record.get("final_candidate"))),
                    "infrastructure_failure": (
                        record.get("live_failure_category") == "infrastructure_failure"
                    ),
                    "human_review_priority": _human_review_priority(record),
                    "human_final_disposition": _truth(record).get("final_disposition", ""),
                    "human_primary_reason": _truth(record).get("primary_reason", ""),
                    "human_repair_target": _truth(record).get("repair_target", ""),
                    "human_genericization": _truth(record).get("genericization", ""),
                    "human_grounding_fidelity": _truth(record).get("grounding_fidelity", ""),
                    "human_distinctive_voice": _truth(record).get("distinctive_voice", ""),
                    "human_voice": _truth(record).get("human_voice", ""),
                    "human_cta": _truth(record).get("cta", ""),
                    "human_author_point_of_view": _truth(record).get("author_point_of_view", ""),
                    "human_repair_locality": _truth(record).get("repair_locality", ""),
                    "review_confidence": _truth(record).get("confidence", ""),
                    "review_status": _truth(record).get("review_status", ""),
                    "final_text_quality_1_5": "",
                    "voice_preservation_1_5": "",
                    "genericization_none_mild_material": "",
                    "evidence_fidelity_pass_concern_fail": "",
                    "repair_improvement_worse_neutral_better": "",
                    "repair_was_necessary_yes_no": "",
                    "would_publish_yes_no": "",
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
        f"provider_call_count: `{metrics['provider_call_count']}`",
        "",
        "THIS RUN IS MEASUREMENT ONLY.",
        "Do not recommend a code fix solely because one case fails.",
        "",
        "## Live Outcome Taxonomy",
    ]
    for outcome, count in metrics.get("live_outcome_counts", {}).items():
        lines.append(f"- `{outcome}`: {count}")
    lines.extend(["", "## Live Failure Categories"])
    for category, count in metrics.get("live_failure_category_counts", {}).items():
        lines.append(f"- `{category}`: {count}")
    product_metrics = metrics.get("product_metrics", {})
    lines.extend(
        [
            "",
            "## Product Corpus Adequacy",
            f"- status: `{product_metrics.get('product_corpus_adequacy_status', '')}`",
            f"- product_case_count: `{product_metrics.get('product_case_count', 0)}`",
            f"- minimum_product_case_count: `{product_metrics.get('minimum_product_case_count', 0)}`",
            f"- preferred_product_case_count: `{product_metrics.get('preferred_product_case_count', 0)}`",
            f"- gap_to_minimum_product_cases: `{product_metrics.get('gap_to_minimum_product_cases', 0)}`",
            f"- product_rates_are_exploratory: `{product_metrics.get('product_rates_are_exploratory', False)}`",
            "",
            "## Historical Outcome Taxonomy",
        ]
    )
    for outcome, count in metrics["outcome_counts"].items():
        lines.append(f"- `{outcome}`: {count}")
    lines.extend(["", "## Infrastructure vs Product Failure"])
    for category, count in metrics["failure_category_counts"].items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Human Ground Truth Coverage", f"- cases: {metrics.get('human_ground_truth_metrics', {}).get('human_ground_truth_case_count', 0)}"])
    lines.extend(["", "## Product Corpus Distribution"])
    lines.append(json.dumps(metrics.get("product_case_distribution", {}), ensure_ascii=False, sort_keys=True))
    lines.append("Product rates must be interpreted alongside `independence_group_count`; repeated failures count independent groups, not raw case variants.")
    lines.extend(["", "## Human vs Runtime Confusion Matrix"])
    lines.append(json.dumps(metrics.get("human_ground_truth_metrics", {}), ensure_ascii=False, sort_keys=True))
    lines.extend(["", "## Unsafe Accepts"])
    lines.append(f"- unsafe_accept_count: `{metrics.get('human_ground_truth_metrics', {}).get('unsafe_accept_count', 0)}`")
    lines.append("Definition: runtime accepted a case that human ground truth marks `NOT_READY`; this is a severe product signal for that experiment.")
    lines.extend(["", "## Unnecessary Repairs"])
    lines.append(f"- unnecessary_repair_count: `{metrics.get('human_ground_truth_metrics', {}).get('unnecessary_repair_count', 0)}`")
    lines.extend(["", "## Missed Repairs"])
    lines.append(f"- missed_repair_count: `{metrics.get('human_ground_truth_metrics', {}).get('missed_repair_count', 0)}`")
    lines.extend(["", "## Genericization Blocks"])
    lines.append(f"- human_material_genericization_count: `{metrics.get('human_ground_truth_metrics', {}).get('human_material_genericization_count', 0)}`")
    lines.append(f"- runtime_accepted_material_genericization_count: `{metrics.get('human_ground_truth_metrics', {}).get('runtime_accepted_material_genericization_count', 0)}`")
    lines.extend(["", "## Repair Product Outcomes"])
    lines.append(json.dumps(metrics.get("product_metrics", {}), ensure_ascii=False, sort_keys=True))
    lines.extend(["", "## Known Non-Systemic Provider Anomalies"])
    lines.append("- `QE_TOTAL_SCORE_MISMATCH`: observed_count=1; systemic=false; no production normalization change.")
    lines.extend(["", "## Human Review", f"- cases: {metrics['human_review_case_count']}"])
    lines.extend(["", "## Live Product Metrics"])
    lines.append(json.dumps(metrics.get("product_metrics", {}), ensure_ascii=False, sort_keys=True))
    lines.extend(["", "## Live Boundary Metrics"])
    lines.append(json.dumps(metrics.get("boundary_metrics", {}), ensure_ascii=False, sort_keys=True))
    lines.extend(["", "## Source Metrics"])
    lines.append(json.dumps(metrics.get("source_metrics", {}), ensure_ascii=False, sort_keys=True))
    lines.extend(["", "## Critical Invariants"])
    lines.append(json.dumps(metrics.get("critical_invariants", {}), ensure_ascii=False, sort_keys=True))
    lines.extend(["", "## Repeated Failure Signatures"])
    lines.append(json.dumps(metrics.get("repeated_failure_signatures", {}), ensure_ascii=False, sort_keys=True))
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
                f"- historical_expected_outcome: `{record.get('historical_expected_outcome', '')}`",
                f"- live_outcome: `{record.get('live_outcome', '')}`",
                f"- human_review_priority: `{_human_review_priority(record)}`",
                f"- human_final_disposition: `{_truth(record).get('final_disposition', '')}`",
                f"- human_primary_reason: `{_truth(record).get('primary_reason', '')}`",
                f"- human_repair_target: `{_truth(record).get('repair_target', '')}`",
                f"- human_genericization: `{_truth(record).get('genericization', '')}`",
                f"- review_confidence: `{_truth(record).get('confidence', '')}`",
                f"- review_status: `{_truth(record).get('review_status', '')}`",
                "- reviewer_label:",
                "- reviewer_notes:",
                "- backlog_action:",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _candidate_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("post_text") or "")
    return ""


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _human_review_priority(record: dict[str, Any]) -> str:
    outcome = record.get("live_outcome")
    if record.get("family") != "candidate_quality_case":
        return "LOW"
    if outcome in {
        "LIVE_REPAIR_ACCEPTED",
        "LIVE_REPAIR_REQUIRED_NOT_EXECUTED",
        "LIVE_REPAIR_TARGET_NOT_FIXED",
        "LIVE_REPAIR_REGRESSION",
        "LIVE_REPAIR_EXCESSIVE_REWRITE",
        "LIVE_GROUNDING_BLOCK",
    }:
        return "HIGH"
    if record.get("human_review_required") or record.get("live_failure_category"):
        return "MEDIUM"
    return "LOW"


def _truth(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("human_ground_truth")
    return value if isinstance(value, dict) else {}
