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
    ACTION_ACCEPT,
    ACTION_NOT_READY,
    ACTION_NEEDS_HUMAN_REVIEW,
    ACTION_REPAIR_EDITORIAL,
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
from services.packaging.linkedin_post_genericization_guard import (
    evaluate_genericization_selection_blocker,
)
from services.packaging.linkedin_post_repair_target_enforcement import (
    evaluate_repair_target_enforcement,
)
from services.packaging.linkedin_post_semantic_grounding_contract import (
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_NEEDS_HUMAN_REVIEW,
    GROUNDING_STATUS_NOT_READY,
    GROUNDING_STATUS_PASS,
    SEMANTIC_GROUNDING_STATUSES,
    FinalPostSemanticGroundingState,
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

SEMANTIC_GROUNDING_READY = GROUNDING_STATUS_PASS
SEMANTIC_GROUNDING_FAILED = GROUNDING_STATUS_FAIL
SEMANTIC_GROUNDING_NOT_READY = GROUNDING_STATUS_NOT_READY
SEMANTIC_GROUNDING_NEEDS_HUMAN_REVIEW = GROUNDING_STATUS_NEEDS_HUMAN_REVIEW


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
    initiating_failed_criterion: str | None = None,
    angle_decision: object | dict | None = None,
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

    decision = _decision_after_repair_target_enforcement(
        decision=decision,
        quality_review=quality_review,
        initiating_failed_criterion=initiating_failed_criterion,
        angle_decision=angle_decision,
    )
    decision = _decision_after_genericization_guard(
        decision=decision,
        candidate_output=candidate_output,
        attempt_history=history,
        policy=policy,
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


def build_final_post_attempt_outcome_from_gate_grounding_and_quality(
    *,
    post_brief: object | dict,
    candidate_output: CandidateWriterOutput,
    gate_output: DeterministicGateOutput,
    semantic_grounding: FinalPostSemanticGroundingState,
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
    initiating_failed_criterion: str | None = None,
    angle_decision: object | dict | None = None,
    decision_controller: FinalPostDecisionController | None = None,
) -> FinalPostAttemptOutcome:
    """Adjudicate one attempt after deterministic, grounding, and quality gates."""

    _validate_semantic_grounding_state(semantic_grounding)
    if not _grounding_passed(semantic_grounding):
        _validate_candidate_gate_payload_continuity(candidate_output, gate_output)
        history = attempt_history or FinalPostAttemptHistory(attempts=[])
        decision = _decision_from_semantic_grounding(semantic_grounding)
        decision_ready_result = FinalPostDecisionReadyResult(
            post_brief=post_brief,
            candidate_output=candidate_output,
            gate_output=gate_output,
            decision=decision,
            quality_review=None,
            attempt_history=history,
        )
        return build_final_post_attempt_outcome(
            decision_ready_result=decision_ready_result,
            attempt_index=attempt_index,
            repair_plan=repair_plan or _grounding_repair_plan(semantic_grounding),
            created_at=created_at,
            parent_attempt_index=parent_attempt_index,
        )

    return build_final_post_attempt_outcome_from_gate_and_quality(
        post_brief=post_brief,
        candidate_output=candidate_output,
        gate_output=gate_output,
        quality_evaluation=quality_evaluation,
        attempt_index=attempt_index,
        attempt_history=attempt_history,
        policy=policy,
        alternative_model_available=alternative_model_available,
        target_model_provider=target_model_provider,
        target_model_name=target_model_name,
        repair_plan=repair_plan,
        created_at=created_at,
        parent_attempt_index=parent_attempt_index,
        initiating_failed_criterion=initiating_failed_criterion,
        angle_decision=angle_decision,
        decision_controller=decision_controller,
    )


def _decision_after_repair_target_enforcement(
    *,
    decision: FinalPostDecision,
    quality_review: dict | None,
    initiating_failed_criterion: str | None,
    angle_decision: object | dict | None,
) -> FinalPostDecision:
    if decision.action != "accept" or initiating_failed_criterion is None:
        return decision
    diagnostics = evaluate_repair_target_enforcement(
        quality_review=quality_review,
        initiating_failed_criterion=initiating_failed_criterion,
        angle_decision=angle_decision,
    )
    if diagnostics.repair_target_fixed is not False:
        return decision
    return FinalPostDecision(
        action=ACTION_NOT_READY,
        reason=diagnostics.repair_target_failure_reason,
        repair_type=None,
        target_model_provider=None,
        target_model_name=None,
        needs_human_review=False,
    )


def _decision_after_genericization_guard(
    *,
    decision: FinalPostDecision,
    candidate_output: CandidateWriterOutput,
    attempt_history: FinalPostAttemptHistory,
    policy: FinalPostDecisionPolicy | None,
) -> FinalPostDecision:
    if decision.action != ACTION_ACCEPT:
        return decision
    post_text = candidate_output.payload.get("post_text")
    diagnostics = evaluate_genericization_selection_blocker(post_text)
    if not diagnostics.blocked:
        return decision

    active_policy = policy or FinalPostDecisionPolicy()
    if _editorial_repair_attempts_remain(attempt_history, active_policy):
        return FinalPostDecision(
            action=ACTION_REPAIR_EDITORIAL,
            reason=diagnostics.reason,
            repair_type="editorial",
            target_model_provider=None,
            target_model_name=None,
            needs_human_review=False,
        )
    return FinalPostDecision(
        action=ACTION_NOT_READY,
        reason=diagnostics.reason,
        repair_type=None,
        target_model_provider=None,
        target_model_name=None,
        needs_human_review=False,
    )


def _editorial_repair_attempts_remain(
    attempt_history: FinalPostAttemptHistory,
    policy: FinalPostDecisionPolicy,
) -> bool:
    if len(attempt_history.attempts) >= policy.max_total_attempts:
        return False
    return _editorial_repair_attempt_count(attempt_history) < policy.max_editorial_repairs


def _editorial_repair_attempt_count(attempt_history: FinalPostAttemptHistory) -> int:
    count = 0
    for attempt in attempt_history.attempts:
        if attempt.decision is not None and attempt.decision.action == ACTION_REPAIR_EDITORIAL:
            count += 1
            continue
        repair_plan = attempt.repair_plan
        if isinstance(repair_plan, dict) and repair_plan.get("repair_type") == "editorial":
            count += 1
    return count

def _validate_quality_evaluation_state(
    quality_evaluation: FinalPostQualityEvaluationState,
) -> None:
    if quality_evaluation.status not in QUALITY_EVALUATION_STATUSES:
        raise ValueError(
            f"Unsupported FinalPostQualityEvaluationState.status: {quality_evaluation.status}"
        )


def _validate_semantic_grounding_state(
    semantic_grounding: FinalPostSemanticGroundingState,
) -> None:
    if semantic_grounding.status not in SEMANTIC_GROUNDING_STATUSES:
        raise ValueError(
            f"Unsupported FinalPostSemanticGroundingState.status: {semantic_grounding.status}"
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


def _grounding_passed(semantic_grounding: FinalPostSemanticGroundingState) -> bool:
    review = semantic_grounding.grounding_review
    return (
        semantic_grounding.status == GROUNDING_STATUS_PASS
        and review is not None
        and review.passed
    )


def _decision_from_semantic_grounding(
    semantic_grounding: FinalPostSemanticGroundingState,
) -> FinalPostDecision:
    if semantic_grounding.status == GROUNDING_STATUS_NEEDS_HUMAN_REVIEW:
        return FinalPostDecision(
            action=ACTION_NEEDS_HUMAN_REVIEW,
            reason=_semantic_grounding_reason(semantic_grounding),
            repair_type=None,
            target_model_provider=None,
            target_model_name=None,
            needs_human_review=True,
        )
    if semantic_grounding.status == GROUNDING_STATUS_FAIL:
        review = semantic_grounding.grounding_review
        if review is not None and review.repairable and not review.requires_human_review:
            # The flow contract currently has a generic repair action. The
            # semantic-specific routing key is the repair_type below.
            return FinalPostDecision(
                action=ACTION_REPAIR_EDITORIAL,
                reason=_semantic_grounding_reason(semantic_grounding),
                repair_type="semantic_grounding",
                target_model_provider=None,
                target_model_name=None,
                needs_human_review=False,
            )
        if review is not None and review.requires_human_review:
            return FinalPostDecision(
                action=ACTION_NEEDS_HUMAN_REVIEW,
                reason=_semantic_grounding_reason(semantic_grounding),
                repair_type=None,
                target_model_provider=None,
                target_model_name=None,
                needs_human_review=True,
            )
    return FinalPostDecision(
        action=ACTION_NOT_READY,
        reason=_semantic_grounding_reason(semantic_grounding),
        repair_type=None,
        target_model_provider=None,
        target_model_name=None,
        needs_human_review=False,
    )


def _semantic_grounding_reason(
    semantic_grounding: FinalPostSemanticGroundingState,
) -> str:
    if semantic_grounding.error_code:
        reason = f"semantic grounding {semantic_grounding.status}: {semantic_grounding.error_code}"
        if semantic_grounding.error_message:
            reason = f"{reason} ({semantic_grounding.error_message})"
        return reason
    review = semantic_grounding.grounding_review
    if review is None:
        return f"semantic grounding {semantic_grounding.status}"
    if review.automatic_fail_reason:
        return f"semantic grounding failed: {review.automatic_fail_reason}"
    if review.blocking_claim_ids:
        return "semantic grounding failed claims: " + ", ".join(review.blocking_claim_ids)
    if review.requires_human_review:
        return review.human_review_reason or "semantic grounding needs human review"
    return f"semantic grounding {semantic_grounding.status}"


def _grounding_repair_plan(
    semantic_grounding: FinalPostSemanticGroundingState,
) -> dict | None:
    review = semantic_grounding.grounding_review
    if review is None or not review.repairable:
        return None
    return {
        "repair_type": "semantic_grounding",
        "failed_claim_ids": list(review.blocking_claim_ids),
        "repair_instruction": "; ".join(
            instruction.instruction for instruction in review.repair_instructions
        ),
        "preserve": [
            "selected evidence",
            "AngleDecision.controlling_angle",
            "valid FinalPostPayload JSON",
        ],
        "avoid": [
            "new facts",
            "stronger certainty than the evidence",
            "unsupported causal language",
            "recovery/stability/optimism drift",
        ],
    }


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
