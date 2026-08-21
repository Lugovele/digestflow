"""Metrics for provider-free PostFlow product validation benchmarks."""
from __future__ import annotations

from collections import Counter
from typing import Any

from services.packaging.postflow_product_validation_corpus import (
    FAILURE_CATEGORY_INFRASTRUCTURE,
    FAILURE_CATEGORY_PRODUCT,
    FAILURE_CATEGORY_PRODUCT_BEHAVIOR,
)


def classify_product_validation_record(record: dict[str, Any]) -> str:
    return str(record.get("expected_outcome") or "historical_inconclusive")


def compute_product_validation_metrics(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    family_counts = Counter(str(record.get("family", "")) for record in records)
    outcome_counts = Counter(classify_product_validation_record(record) for record in records)
    failure_category_counts = Counter(
        str(record.get("failure_category", "")) for record in records
    )
    topic_counts = Counter(str(record.get("topic", "")) for record in records)
    risk_type_counts = Counter(str(record.get("risk_type", "")) for record in records)
    human_review_count = sum(1 for record in records if record.get("human_review_required"))
    return {
        "case_count": len(records),
        "provider_call_count": 0,
        "family_counts": dict(sorted(family_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "failure_category_counts": dict(sorted(failure_category_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "risk_type_counts": dict(sorted(risk_type_counts.items())),
        "human_review_case_count": human_review_count,
        "product_failure_case_count": failure_category_counts.get(FAILURE_CATEGORY_PRODUCT, 0),
        "infrastructure_failure_case_count": failure_category_counts.get(
            FAILURE_CATEGORY_INFRASTRUCTURE, 0
        ),
        "product_behavior_case_count": failure_category_counts.get(
            FAILURE_CATEGORY_PRODUCT_BEHAVIOR, 0
        ),
    }


def compare_product_validation_metrics(
    current_metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None
) -> dict[str, Any] | None:
    if baseline_metrics is None:
        return None
    comparable_keys = (
        "case_count",
        "human_review_case_count",
        "product_failure_case_count",
        "infrastructure_failure_case_count",
        "product_behavior_case_count",
    )
    return {
        key: int(current_metrics.get(key, 0)) - int(baseline_metrics.get(key, 0))
        for key in comparable_keys
    }
