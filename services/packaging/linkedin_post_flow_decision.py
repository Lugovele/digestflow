"""Deterministic routing for future final LinkedIn post flow attempts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_final_post_diagnostics import FinalPostDiagnostics
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_ACCEPT,
    ACTION_NEEDS_HUMAN_REVIEW,
    ACTION_NOT_READY,
    ACTION_REPAIR_EDITORIAL,
    ACTION_REPAIR_MECHANICAL,
    ACTION_TRY_ALTERNATIVE_MODEL,
    FinalPostAttempt,
    FinalPostAttemptHistory,
    FinalPostDecision,
)


# Quality thresholds come from the 9-criterion / 45-point LinkedIn quality target.
QUALITY_PASS_THRESHOLD = 36
REQUIRED_QUALITY_MINIMUMS = {
    "hook": 4,
    "controlling_angle": 4,
    "author_point_of_view": 4,
    "human_voice": 4,
    "evidence": 3,
}
HUMAN_REVIEW_FLAGS = (
    "blocking_factuality_ambiguity",
    "unsupported_claims_cannot_be_safely_repaired",
    "sensitive_topic_risk",
)


@dataclass(frozen=True)
class FinalPostDecisionPolicy:
    max_total_attempts: int = 3
    max_mechanical_repairs: int = 1
    max_editorial_repairs: int = 1
    allow_alternative_model: bool = False


class FinalPostDecisionController:
    """Route a final-post attempt from structured results only."""

    def decide(
        self,
        *,
        validation_passed: bool,
        validation_error: str = "",
        diagnostics: FinalPostDiagnostics | None,
        quality_review: dict | None,
        attempt_history: FinalPostAttemptHistory,
        policy: FinalPostDecisionPolicy | None = None,
        alternative_model_available: bool = False,
        target_model_provider: str | None = None,
        target_model_name: str | None = None,
    ) -> FinalPostDecision:
        active_policy = policy or FinalPostDecisionPolicy()

        if _has_mechanical_failure(validation_passed, diagnostics):
            if _repair_attempts_remain(
                attempt_history,
                action=ACTION_REPAIR_MECHANICAL,
                max_repairs=active_policy.max_mechanical_repairs,
                max_total_attempts=active_policy.max_total_attempts,
            ):
                return FinalPostDecision(
                    action=ACTION_REPAIR_MECHANICAL,
                    reason=_mechanical_failure_reason(
                        validation_passed,
                        validation_error,
                        diagnostics,
                    ),
                    repair_type="mechanical",
                    target_model_provider=None,
                    target_model_name=None,
                    needs_human_review=False,
                )
            return FinalPostDecision(
                action=ACTION_NOT_READY,
                reason="mechanical failures remain after allowed repair attempts",
                repair_type=None,
                target_model_provider=None,
                target_model_name=None,
                needs_human_review=False,
            )

        if quality_review is None:
            return FinalPostDecision(
                action=ACTION_NOT_READY,
                reason="missing quality review",
                repair_type=None,
                target_model_provider=None,
                target_model_name=None,
                needs_human_review=False,
            )

        if _needs_human_review(quality_review):
            return FinalPostDecision(
                action=ACTION_NEEDS_HUMAN_REVIEW,
                reason="quality review flagged a human-review risk",
                repair_type=None,
                target_model_provider=None,
                target_model_name=None,
                needs_human_review=True,
            )

        if _quality_review_passed(quality_review):
            return FinalPostDecision(
                action=ACTION_ACCEPT,
                reason="deterministic and quality gates passed",
                repair_type=None,
                target_model_provider=None,
                target_model_name=None,
                needs_human_review=False,
            )

        if _repair_attempts_remain(
            attempt_history,
            action=ACTION_REPAIR_EDITORIAL,
            max_repairs=active_policy.max_editorial_repairs,
            max_total_attempts=active_policy.max_total_attempts,
        ):
            return FinalPostDecision(
                action=ACTION_REPAIR_EDITORIAL,
                reason=_quality_failure_reason(quality_review),
                repair_type="editorial",
                target_model_provider=None,
                target_model_name=None,
                needs_human_review=False,
            )

        if active_policy.allow_alternative_model and alternative_model_available:
            return FinalPostDecision(
                action=ACTION_TRY_ALTERNATIVE_MODEL,
                reason="editorial repair attempts exhausted and alternative model is available",
                repair_type=None,
                target_model_provider=target_model_provider,
                target_model_name=target_model_name,
                needs_human_review=False,
            )

        return FinalPostDecision(
            action=ACTION_NOT_READY,
            reason="quality gate failed and no repair or model path remains",
            repair_type=None,
            target_model_provider=None,
            target_model_name=None,
            needs_human_review=False,
        )


def _has_mechanical_failure(
    validation_passed: bool,
    diagnostics: FinalPostDiagnostics | None,
) -> bool:
    return (
        not validation_passed
        or diagnostics is None
        or not diagnostics.system_linkedin_ready
        or not diagnostics.deterministic_checks_passed
    )


def _quality_review_passed(quality_review: dict | None) -> bool:
    if not isinstance(quality_review, dict):
        return False
    if quality_review.get("pass") is not True:
        return False
    if int(quality_review.get("total_score") or 0) < QUALITY_PASS_THRESHOLD:
        return False
    if quality_review.get("automatic_fail_reason"):
        return False

    scores = quality_review.get("scores")
    if not isinstance(scores, dict):
        return False
    return all(
        int(scores.get(criterion) or 0) >= minimum
        for criterion, minimum in REQUIRED_QUALITY_MINIMUMS.items()
    )


def _needs_human_review(quality_review: dict | None) -> bool:
    if not isinstance(quality_review, dict):
        return False
    return any(quality_review.get(flag) is True for flag in HUMAN_REVIEW_FLAGS)


def _repair_attempts_remain(
    attempt_history: FinalPostAttemptHistory,
    *,
    action: str,
    max_repairs: int,
    max_total_attempts: int,
) -> bool:
    if len(attempt_history.attempts) >= max_total_attempts:
        return False
    return _count_repair_attempts(attempt_history, action) < max_repairs


def _count_repair_attempts(attempt_history: FinalPostAttemptHistory, action: str) -> int:
    count = 0
    expected_repair_type = "mechanical" if action == ACTION_REPAIR_MECHANICAL else "editorial"
    for attempt in attempt_history.attempts:
        if _attempt_decision_action(attempt) == action:
            count += 1
            continue
        repair_plan = attempt.repair_plan
        if isinstance(repair_plan, dict) and repair_plan.get("repair_type") == expected_repair_type:
            count += 1
    return count


def _attempt_decision_action(attempt: FinalPostAttempt) -> str | None:
    if attempt.decision is None:
        return None
    return attempt.decision.action


def _mechanical_failure_reason(
    validation_passed: bool,
    validation_error: str,
    diagnostics: FinalPostDiagnostics | None,
) -> str:
    if not validation_passed:
        return validation_error or "validation failed"
    if diagnostics is None:
        return "deterministic diagnostics missing"
    if diagnostics.repair_reasons:
        return ", ".join(diagnostics.repair_reasons)
    return "deterministic gate failed"


def _quality_failure_reason(quality_review: dict | None) -> str:
    if not isinstance(quality_review, dict):
        return "quality review missing"
    if quality_review.get("automatic_fail_reason"):
        return f"automatic fail: {quality_review['automatic_fail_reason']}"

    failed_criteria = quality_review.get("failed_criteria")
    if isinstance(failed_criteria, list) and failed_criteria:
        return "quality criteria failed: " + ", ".join(str(item) for item in failed_criteria)

    scores = quality_review.get("scores")
    if isinstance(scores, dict):
        failed_minimums = [
            criterion
            for criterion, minimum in REQUIRED_QUALITY_MINIMUMS.items()
            if int(scores.get(criterion) or 0) < minimum
        ]
        if failed_minimums:
            return "quality minimums failed: " + ", ".join(failed_minimums)

    total_score = int(quality_review.get("total_score") or 0)
    if total_score < QUALITY_PASS_THRESHOLD:
        return f"quality score below threshold: {total_score}/{QUALITY_PASS_THRESHOLD}"

    return "quality gate failed"
