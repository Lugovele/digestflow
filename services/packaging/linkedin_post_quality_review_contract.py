"""Canonical quality-review normalization for final LinkedIn post flow."""
from __future__ import annotations

import copy
from typing import Any

from services.packaging.linkedin_post_flow_decision import HUMAN_REVIEW_FLAGS
from services.packaging.linkedin_post_flow_handoffs import (
    QualityReviewResult as HandoffQualityReviewResult,
)
from services.packaging.linkedin_post_pipeline import (
    QualityReviewResult as PipelineQualityReviewResult,
)


CANONICAL_QUALITY_SCORE_KEYS = (
    "hook",
    "controlling_angle",
    "reader_problem",
    "pattern_interrupt",
    "evidence",
    "author_point_of_view",
    "human_voice",
    "practical_value",
    "cta",
)

REQUIRED_QUALITY_REVIEW_FIELDS = (
    "scores",
    "total_score",
    "pass",
    "failed_criteria",
    "automatic_fail_reason",
)

OPTIONAL_QUALITY_REVIEW_FIELDS = (
    "notes",
    "requires_human_review",
    "human_review_reason",
    *HUMAN_REVIEW_FLAGS,
)

PASS_FIELD = "pass"
LEGACY_PASS_FIELD = "pass_result"
UNSUPPORTED_PASS_FIELD = "passed"
MIN_CRITERION_SCORE = 1
MAX_CRITERION_SCORE = 5
MIN_TOTAL_SCORE = len(CANONICAL_QUALITY_SCORE_KEYS) * MIN_CRITERION_SCORE
MAX_TOTAL_SCORE = len(CANONICAL_QUALITY_SCORE_KEYS) * MAX_CRITERION_SCORE


def normalize_quality_review_result(review: object | dict) -> dict[str, Any]:
    """Return a decision-controller-compatible quality review dictionary."""

    raw_review = _review_to_dict(review)
    return _normalize_quality_review_dict(raw_review)


def _review_to_dict(review: object | dict) -> dict[str, Any]:
    if isinstance(review, HandoffQualityReviewResult):
        return review.to_dict()
    if isinstance(review, PipelineQualityReviewResult):
        return {
            "scores": copy.deepcopy(review.scores),
            "total_score": review.total_score,
            LEGACY_PASS_FIELD: review.pass_result,
            "failed_criteria": copy.deepcopy(review.failed_criteria),
            "automatic_fail_reason": review.automatic_fail_reason,
        }
    if isinstance(review, dict):
        return copy.deepcopy(review)
    raise ValueError("quality review must be a supported QualityReviewResult or dictionary.")


def _normalize_quality_review_dict(review: dict[str, Any]) -> dict[str, Any]:
    pass_value = _extract_pass_value(review)
    scores = _normalize_scores(review.get("scores"))
    total_score = _normalize_total_score(_require_present(review, "total_score"))
    failed_criteria = _normalize_failed_criteria(review.get("failed_criteria"))
    automatic_fail_reason = review.get("automatic_fail_reason")
    if not isinstance(automatic_fail_reason, str):
        raise ValueError("quality review automatic_fail_reason must be a string.")

    normalized = {
        "scores": scores,
        "total_score": total_score,
        "pass": pass_value,
        "failed_criteria": failed_criteria,
        "automatic_fail_reason": automatic_fail_reason,
    }
    _copy_optional_fields(review, normalized)
    return normalized


def _extract_pass_value(review: dict[str, Any]) -> bool:
    present_pass_fields = [
        field
        for field in (PASS_FIELD, LEGACY_PASS_FIELD, UNSUPPORTED_PASS_FIELD)
        if field in review
    ]
    if not present_pass_fields:
        raise ValueError("quality review is missing pass result field.")
    if len(present_pass_fields) > 1:
        raise ValueError(
            "quality review has conflicting pass fields: "
            + ", ".join(present_pass_fields)
        )

    pass_field = present_pass_fields[0]
    if pass_field == UNSUPPORTED_PASS_FIELD:
        raise ValueError("quality review field 'passed' is not a supported input shape.")
    pass_value = review[pass_field]
    if not isinstance(pass_value, bool):
        raise ValueError("quality review pass result must be a boolean.")
    return pass_value


def _normalize_scores(scores: Any) -> dict[str, int]:
    if not isinstance(scores, dict):
        raise ValueError("quality review scores must be a dictionary.")

    actual_keys = set(scores)
    expected_keys = set(CANONICAL_QUALITY_SCORE_KEYS)
    missing = sorted(expected_keys - actual_keys)
    if missing:
        raise ValueError(f"quality review scores missing criteria: {missing}.")
    unexpected = sorted(actual_keys - expected_keys)
    if unexpected:
        raise ValueError(f"quality review scores contain unexpected criteria: {unexpected}.")

    return {
        key: _normalize_score_value(key, scores[key])
        for key in CANONICAL_QUALITY_SCORE_KEYS
    }


def _normalize_score_value(criterion: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"quality review score '{criterion}' must be an integer.")
    if value < MIN_CRITERION_SCORE or value > MAX_CRITERION_SCORE:
        raise ValueError(
            f"quality review score '{criterion}' must be between "
            f"{MIN_CRITERION_SCORE} and {MAX_CRITERION_SCORE}."
        )
    return value


def _normalize_total_score(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("quality review total_score must be an integer.")
    if value < MIN_TOTAL_SCORE or value > MAX_TOTAL_SCORE:
        raise ValueError(
            f"quality review total_score must be between "
            f"{MIN_TOTAL_SCORE} and {MAX_TOTAL_SCORE}."
        )
    return value


def _normalize_failed_criteria(failed_criteria: Any) -> list[str]:
    if not isinstance(failed_criteria, (list, tuple)):
        raise ValueError("quality review failed_criteria must be a list or tuple.")
    if not all(isinstance(item, str) for item in failed_criteria):
        raise ValueError("quality review failed_criteria must contain strings only.")
    return list(failed_criteria)


def _copy_optional_fields(source: dict[str, Any], target: dict[str, Any]) -> None:
    if "notes" in source:
        target["notes"] = _normalize_notes(source["notes"])
    if "requires_human_review" in source:
        target["requires_human_review"] = _normalize_bool_optional(
            source["requires_human_review"],
            "requires_human_review",
        )
    if "human_review_reason" in source:
        target["human_review_reason"] = _normalize_string_optional(
            source["human_review_reason"],
            "human_review_reason",
        )
    for flag in HUMAN_REVIEW_FLAGS:
        if flag in source:
            target[flag] = _normalize_bool_optional(source[flag], flag)


def _normalize_notes(notes: Any) -> list[str]:
    if not isinstance(notes, (list, tuple)):
        raise ValueError("quality review notes must be a list or tuple.")
    if not all(isinstance(item, str) for item in notes):
        raise ValueError("quality review notes must contain strings only.")
    return list(notes)


def _normalize_bool_optional(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"quality review {field_name} must be a boolean.")
    return value


def _normalize_string_optional(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"quality review {field_name} must be a string.")
    return value


def _require_present(review: dict[str, Any], field_name: str) -> Any:
    if field_name not in review:
        raise ValueError(f"quality review is missing {field_name}.")
    return copy.deepcopy(review[field_name])
