"""Post-decision outcome construction for one final-post attempt.

This layer records one completed decision-ready attempt and maps the existing
decision action to an explicit outcome. It does not run gates, controllers,
quality evaluation, repair, providers, or persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_post_deterministic_orchestration import (
    FinalPostDecisionReadyResult,
)
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_ACCEPT,
    ACTION_NEEDS_HUMAN_REVIEW,
    ACTION_NOT_READY,
    ACTION_REPAIR_EDITORIAL,
    ACTION_REPAIR_MECHANICAL,
    ACTION_TRY_ALTERNATIVE_MODEL,
    FINAL_POST_DECISION_ACTIONS,
    STATUS_ACCEPTED,
    STATUS_NEEDS_HUMAN_REVIEW,
    STATUS_NOT_READY,
    STATUS_TRY_ALTERNATIVE_MODEL,
    FinalPostAttempt,
    FinalPostAttemptHistory,
    FinalPostDecision,
    FinalPostFlowResult,
)


OUTCOME_ACCEPTED = "accepted"
OUTCOME_REPAIR_REQUIRED = "repair_required"
OUTCOME_NOT_READY = "not_ready"
OUTCOME_NEEDS_HUMAN_REVIEW = "needs_human_review"
OUTCOME_TRY_ALTERNATIVE_MODEL = "try_alternative_model"

FINAL_POST_ATTEMPT_OUTCOMES = (
    OUTCOME_ACCEPTED,
    OUTCOME_REPAIR_REQUIRED,
    OUTCOME_NOT_READY,
    OUTCOME_NEEDS_HUMAN_REVIEW,
    OUTCOME_TRY_ALTERNATIVE_MODEL,
)


@dataclass(frozen=True)
class FinalPostAttemptOutcome:
    outcome: str
    attempt: FinalPostAttempt
    attempt_history: FinalPostAttemptHistory
    decision: FinalPostDecision
    accepted_result: FinalPostFlowResult | None
    repair_required: bool
    repair_plan: dict | None
    terminal_result: FinalPostFlowResult | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "attempt": self.attempt.to_dict(),
            "attempt_history": self.attempt_history.to_dict(),
            "decision": self.decision.to_dict(),
            "accepted_result": (
                self.accepted_result.to_dict() if self.accepted_result else None
            ),
            "repair_required": self.repair_required,
            "repair_plan": _serialize_value(self.repair_plan),
            "terminal_result": (
                self.terminal_result.to_dict() if self.terminal_result else None
            ),
            "reason": self.reason,
        }


def build_final_post_attempt_outcome(
    *,
    decision_ready_result: FinalPostDecisionReadyResult,
    attempt_index: int,
    repair_plan: dict | None = None,
    created_at: str | None = None,
    parent_attempt_index: int | None = None,
) -> FinalPostAttemptOutcome:
    """Record a completed attempt and map its existing decision to an outcome."""

    decision = decision_ready_result.decision
    _validate_decision_action(decision.action)

    attempt = _build_attempt(
        decision_ready_result=decision_ready_result,
        attempt_index=attempt_index,
        repair_plan=repair_plan,
        created_at=created_at,
        parent_attempt_index=parent_attempt_index,
    )
    updated_history = decision_ready_result.attempt_history.add_attempt(attempt)

    return _build_outcome(
        decision=decision,
        attempt=attempt,
        attempt_history=updated_history,
        repair_plan=repair_plan,
    )


def _build_attempt(
    *,
    decision_ready_result: FinalPostDecisionReadyResult,
    attempt_index: int,
    repair_plan: dict | None,
    created_at: str | None,
    parent_attempt_index: int | None,
) -> FinalPostAttempt:
    candidate_output = decision_ready_result.candidate_output
    gate_output = decision_ready_result.gate_output
    return FinalPostAttempt(
        attempt_index=attempt_index,
        payload=gate_output.payload,
        validation_passed=gate_output.validation_passed,
        validation_error=gate_output.validation_error,
        diagnostics=gate_output.diagnostics,
        quality_review=decision_ready_result.quality_review,
        repair_plan=repair_plan,
        decision=decision_ready_result.decision,
        provider=candidate_output.provider,
        model=candidate_output.model,
        prompt_name=candidate_output.prompt_name,
        prompt_version=candidate_output.prompt_version,
        token_usage=candidate_output.token_usage,
        cost_metadata=candidate_output.cost_metadata,
        created_at=created_at,
        parent_attempt_index=parent_attempt_index,
    )


def _build_outcome(
    *,
    decision: FinalPostDecision,
    attempt: FinalPostAttempt,
    attempt_history: FinalPostAttemptHistory,
    repair_plan: dict | None,
) -> FinalPostAttemptOutcome:
    if decision.action == ACTION_ACCEPT:
        accepted_result = _flow_result(
            status=STATUS_ACCEPTED,
            accepted_payload=attempt.payload,
            attempt_history=attempt_history,
            decision=decision,
        )
        return FinalPostAttemptOutcome(
            outcome=OUTCOME_ACCEPTED,
            attempt=attempt,
            attempt_history=attempt_history,
            decision=decision,
            accepted_result=accepted_result,
            repair_required=False,
            repair_plan=None,
            terminal_result=accepted_result,
            reason=decision.reason,
        )

    if decision.action in {ACTION_REPAIR_MECHANICAL, ACTION_REPAIR_EDITORIAL}:
        return FinalPostAttemptOutcome(
            outcome=OUTCOME_REPAIR_REQUIRED,
            attempt=attempt,
            attempt_history=attempt_history,
            decision=decision,
            accepted_result=None,
            repair_required=True,
            repair_plan=repair_plan,
            terminal_result=None,
            reason=decision.reason,
        )

    if decision.action == ACTION_NOT_READY:
        terminal_result = _flow_result(
            status=STATUS_NOT_READY,
            accepted_payload=None,
            attempt_history=attempt_history,
            decision=decision,
        )
        return FinalPostAttemptOutcome(
            outcome=OUTCOME_NOT_READY,
            attempt=attempt,
            attempt_history=attempt_history,
            decision=decision,
            accepted_result=None,
            repair_required=False,
            repair_plan=None,
            terminal_result=terminal_result,
            reason=decision.reason,
        )

    if decision.action == ACTION_NEEDS_HUMAN_REVIEW:
        terminal_result = _flow_result(
            status=STATUS_NEEDS_HUMAN_REVIEW,
            accepted_payload=None,
            attempt_history=attempt_history,
            decision=decision,
        )
        return FinalPostAttemptOutcome(
            outcome=OUTCOME_NEEDS_HUMAN_REVIEW,
            attempt=attempt,
            attempt_history=attempt_history,
            decision=decision,
            accepted_result=None,
            repair_required=False,
            repair_plan=None,
            terminal_result=terminal_result,
            reason=decision.reason,
        )

    if decision.action == ACTION_TRY_ALTERNATIVE_MODEL:
        terminal_result = _flow_result(
            status=STATUS_TRY_ALTERNATIVE_MODEL,
            accepted_payload=None,
            attempt_history=attempt_history,
            decision=decision,
        )
        return FinalPostAttemptOutcome(
            outcome=OUTCOME_TRY_ALTERNATIVE_MODEL,
            attempt=attempt,
            attempt_history=attempt_history,
            decision=decision,
            accepted_result=None,
            repair_required=False,
            repair_plan=None,
            terminal_result=terminal_result,
            reason=decision.reason,
        )

    raise ValueError(f"Unsupported FinalPostDecision.action: {decision.action}")


def _flow_result(
    *,
    status: str,
    accepted_payload: dict | None,
    attempt_history: FinalPostAttemptHistory,
    decision: FinalPostDecision,
) -> FinalPostFlowResult:
    return FinalPostFlowResult(
        status=status,
        accepted_payload=accepted_payload,
        attempt_history=attempt_history,
        final_decision=decision,
        reason=decision.reason,
    )


def _validate_decision_action(action: str) -> None:
    if action not in FINAL_POST_DECISION_ACTIONS:
        raise ValueError(f"Unsupported FinalPostDecision.action: {action}")


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    return value
