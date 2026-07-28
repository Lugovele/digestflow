"""Pure final-post attempt adjudication after quality evaluation.

This module closes one staged final-post attempt from existing deterministic
facts. It does not execute prompts, call providers, parse raw evaluator output,
normalize quality reviews, generate repair plans, persist data, or connect to
runtime packaging.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_post_attempt_outcome import (
    FinalPostAttemptOutcome,
    build_final_post_attempt_outcome,
)
from services.packaging.linkedin_post_deterministic_orchestration import (
    FinalPostDecisionReadyResult,
)
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_NOT_READY,
    FinalPostAttemptHistory,
    FinalPostDecision,
)
from services.packaging.linkedin_post_flow_decision import (
    FinalPostDecisionController,
    FinalPostDecisionPolicy,
)
from services.packaging.linkedin_post_flow_handoffs import (
    CandidateWriterOutput,
    DeterministicGateOutput,
)


QUALITY_EVALUATION_READY = "ready"
QUALITY_EVALUATION_NOT_RUN = "not_run"
QUALITY_EVALUATION_EXECUTION_FAILED = "execution_failed"
QUALITY_EVALUATION_PARSE_FAILED = "parse_failed"
QUALITY_EVALUATION_NORMALIZATION_FAILED = "normalization_failed"

QUALITY_EVALUATION_FAILURE_STATUSES = (
    QUALITY_EVALUATION_EXECUTION_FAILED,
    QUALITY_EVALUATION_PARSE_FAILED,
    QUALITY_EVALUATION_NORMALIZATION_FAILED,
)
QUALITY_EVALUATION_STATUSES = (
    QUALITY_EVALUATION_READY,
    QUALITY_EVALUATION_NOT_RUN,
    *QUALITY_EVALUATION_FAILURE_STATUSES,
)


@dataclass(frozen=True)
class FinalPostQualityEvaluationState:
    status: str
    quality_review: dict | None
    error_code: str | None = None
    error_message: str = ""
    metadata: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "quality_review": copy.deepcopy(self.quality_review),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "metadata": copy.deepcopy(self.metadata),
        }


def build_final_post_attempt_outcome_from_gate_and_quality(
    *,
    post_brief: object | dict,
    candidate_output: CandidateWriterOutput,
    gate_output: DeterministicGateOutput,
    quality_evaluation: FinalPostQualityEvaluationState,
    attempt_index: int,
    attempt_history: FinalPostAttemptHistory | None = None,
    policy: FinalPostDecisionPolicy | None = None,
    alternative_model_available: bool = False,
    target_model_provider: str | None = None,
    target_model_name: str | None = None,
    repair_plan: dict | None = None,
    created_at: str | None = None,
    parent_attempt_index: int | None = None,
    decision_controller: FinalPostDecisionController | None = None,
) -> FinalPostAttemptOutcome:
    """Adjudicate one candidate attempt from existing gate and quality facts."""

    _validate_quality_evaluation_state(quality_evaluation)
    _validate_candidate_gate_payload_continuity(candidate_output, gate_output)

    history = attempt_history or FinalPostAttemptHistory(attempts=[])
    controller = decision_controller or FinalPostDecisionController()
    quality_review = _quality_review_for_decision(gate_output, quality_evaluation)
    if quality_review is None and _gate_passed(gate_output):
        decision = _not_ready_decision_from_quality_evaluation(quality_evaluation)
    else:
        decision = controller.decide(
            validation_passed=gate_output.validation_passed,
            validation_error=gate_output.validation_error,
            diagnostics=gate_output.diagnostics,
            quality_review=quality_review,
            attempt_history=history,
            policy=policy,
            alternative_model_available=alternative_model_available,
            target_model_provider=target_model_provider,
            target_model_name=target_model_name,
        )

    decision_ready_result = FinalPostDecisionReadyResult(
        post_brief=post_brief,
        candidate_output=candidate_output,
        gate_output=gate_output,
        decision=decision,
        quality_review=quality_review,
        attempt_history=history,
    )
    return build_final_post_attempt_outcome(
        decision_ready_result=decision_ready_result,
        attempt_index=attempt_index,
        repair_plan=repair_plan,
        created_at=created_at,
        parent_attempt_index=parent_attempt_index,
    )


def _validate_quality_evaluation_state(
    quality_evaluation: FinalPostQualityEvaluationState,
) -> None:
    if quality_evaluation.status not in QUALITY_EVALUATION_STATUSES:
        raise ValueError(
            f"Unsupported FinalPostQualityEvaluationState.status: {quality_evaluation.status}"
        )


def _validate_candidate_gate_payload_continuity(
    candidate_output: CandidateWriterOutput,
    gate_output: DeterministicGateOutput,
) -> None:
    if gate_output.payload is not candidate_output.payload:
        raise ValueError(
            "CandidateWriterOutput.payload must match DeterministicGateOutput.payload."
        )


def _quality_review_for_decision(
    gate_output: DeterministicGateOutput,
    quality_evaluation: FinalPostQualityEvaluationState,
) -> dict | None:
    if not _gate_passed(gate_output):
        return None
    if quality_evaluation.status == QUALITY_EVALUATION_READY:
        return quality_evaluation.quality_review
    return None


def _gate_passed(gate_output: DeterministicGateOutput) -> bool:
    return (
        gate_output.validation_passed
        and gate_output.diagnostics.system_linkedin_ready
        and gate_output.diagnostics.deterministic_checks_passed
    )


def _not_ready_decision_from_quality_evaluation(
    quality_evaluation: FinalPostQualityEvaluationState,
) -> FinalPostDecision:
    return FinalPostDecision(
        action=ACTION_NOT_READY,
        reason=_quality_evaluation_not_ready_reason(quality_evaluation),
        repair_type=None,
        target_model_provider=None,
        target_model_name=None,
        needs_human_review=False,
    )


def _quality_evaluation_not_ready_reason(
    quality_evaluation: FinalPostQualityEvaluationState,
) -> str:
    if quality_evaluation.status == QUALITY_EVALUATION_READY:
        return "quality evaluation ready state missing normalized review"
    if quality_evaluation.status == QUALITY_EVALUATION_NOT_RUN:
        return "quality evaluation not run"

    reason = f"quality evaluation {quality_evaluation.status}"
    if quality_evaluation.error_code:
        reason = f"{reason}: {quality_evaluation.error_code}"
    if quality_evaluation.error_message:
        reason = f"{reason} ({quality_evaluation.error_message})"
    return reason
