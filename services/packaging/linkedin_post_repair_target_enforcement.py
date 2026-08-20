"""Repair-target enforcement for repaired final-post attempts.

This module evaluates whether a repaired attempt fixed the criterion that caused
repair. It is deterministic and provider-free; it does not adjudicate a full
attempt, execute prompts, run evaluators, mutate payloads, or persist data.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_post_quality_rubric_contract import (
    QualityEvaluatorRubricPayload,
    get_quality_evaluator_rubric_payload,
    normalize_quality_evaluator_rubric_payload,
)


AUTHOR_POV_CRITERION = "author_point_of_view"
EXPLICIT_AUTHOR_OWNED_STATEMENT_REQUIRED = "explicit_author_owned_statement_required"


@dataclass(frozen=True)
class RepairTargetEnforcementDiagnostics:
    initiating_failed_criterion: str | None
    target_quality_score: int | None
    target_required_minimum: int | None
    repair_target_fixed: bool | None
    repair_target_failure_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "initiating_failed_criterion": self.initiating_failed_criterion,
            "target_quality_score": self.target_quality_score,
            "target_required_minimum": self.target_required_minimum,
            "repair_target_fixed": self.repair_target_fixed,
            "repair_target_failure_reason": self.repair_target_failure_reason,
        }


def evaluate_repair_target_enforcement(
    *,
    quality_review: dict[str, Any] | None,
    initiating_failed_criterion: str | None,
    angle_decision: object | dict | None,
    rubric: QualityEvaluatorRubricPayload | dict[str, Any] | None = None,
) -> RepairTargetEnforcementDiagnostics:
    """Return target-fix diagnostics for a repaired attempt."""

    criterion = _normalize_criterion(initiating_failed_criterion)
    if criterion is None:
        return RepairTargetEnforcementDiagnostics(
            initiating_failed_criterion=None,
            target_quality_score=None,
            target_required_minimum=None,
            repair_target_fixed=None,
            repair_target_failure_reason="initiating repair criterion unavailable",
        )
    normalized_rubric = (
        get_quality_evaluator_rubric_payload()
        if rubric is None
        else normalize_quality_evaluator_rubric_payload(rubric)
    )
    required_minimum = _target_required_minimum(
        criterion,
        angle_decision=angle_decision,
        rubric=normalized_rubric,
    )
    if not isinstance(quality_review, dict):
        return RepairTargetEnforcementDiagnostics(
            initiating_failed_criterion=criterion,
            target_quality_score=None,
            target_required_minimum=required_minimum,
            repair_target_fixed=None,
            repair_target_failure_reason="quality review unavailable for repair target enforcement",
        )

    score = _quality_score(quality_review, criterion)
    if score is None:
        return RepairTargetEnforcementDiagnostics(
            initiating_failed_criterion=criterion,
            target_quality_score=None,
            target_required_minimum=required_minimum,
            repair_target_fixed=None,
            repair_target_failure_reason=f"quality score for {criterion} unavailable",
        )

    failed_criteria = _failed_criteria(quality_review)
    target_fixed = score >= required_minimum and criterion not in failed_criteria
    if target_fixed:
        reason = f"{criterion} score {score} >= required {required_minimum} and criterion no longer failed"
    elif criterion in failed_criteria:
        reason = f"repair target not fixed: {criterion} remains in failed_criteria"
    else:
        reason = f"repair target not fixed: {criterion} score {score} < required {required_minimum}"

    return RepairTargetEnforcementDiagnostics(
        initiating_failed_criterion=criterion,
        target_quality_score=score,
        target_required_minimum=required_minimum,
        repair_target_fixed=target_fixed,
        repair_target_failure_reason=reason,
    )


def _normalize_criterion(criterion: str | None) -> str | None:
    if not isinstance(criterion, str):
        return None
    normalized = criterion.strip()
    return normalized or None


def _target_required_minimum(
    criterion: str,
    *,
    angle_decision: object | dict | None,
    rubric: QualityEvaluatorRubricPayload,
) -> int:
    required_minimum = rubric.required_minimums.get(criterion, rubric.score_max - 1)
    if (
        criterion == AUTHOR_POV_CRITERION
        and _explicit_author_owned_statement_required(angle_decision)
    ):
        required_minimum = max(required_minimum, rubric.score_max)
    return required_minimum


def _explicit_author_owned_statement_required(angle_decision: object | dict | None) -> bool:
    directive = _field_value(angle_decision, "authorial_voice_directive")
    return (
        _field_value(directive, "personal_presence_requirement")
        == EXPLICIT_AUTHOR_OWNED_STATEMENT_REQUIRED
    )


def _quality_score(quality_review: dict[str, Any], criterion: str) -> int | None:
    scores = quality_review.get("scores")
    if not isinstance(scores, dict):
        return None
    score = scores.get(criterion)
    if isinstance(score, bool) or not isinstance(score, int):
        return None
    return score


def _failed_criteria(quality_review: dict[str, Any]) -> list[str]:
    failed_criteria = quality_review.get("failed_criteria")
    if not isinstance(failed_criteria, list):
        return []
    return [copy.deepcopy(item) for item in failed_criteria if isinstance(item, str)]


def _field_value(source: object | dict | None, field_name: str) -> Any:
    if isinstance(source, dict):
        return source.get(field_name)
    return getattr(source, field_name, None)
