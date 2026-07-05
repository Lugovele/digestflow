"""Spec-only contracts for future final LinkedIn post flow attempts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_final_post_diagnostics import FinalPostDiagnostics


ACTION_ACCEPT = "accept"
ACTION_REPAIR_MECHANICAL = "repair_mechanical"
ACTION_REPAIR_EDITORIAL = "repair_editorial"
ACTION_TRY_ALTERNATIVE_MODEL = "try_alternative_model"
ACTION_NOT_READY = "not_ready"
ACTION_NEEDS_HUMAN_REVIEW = "needs_human_review"

FINAL_POST_DECISION_ACTIONS = (
    ACTION_ACCEPT,
    ACTION_REPAIR_MECHANICAL,
    ACTION_REPAIR_EDITORIAL,
    ACTION_TRY_ALTERNATIVE_MODEL,
    ACTION_NOT_READY,
    ACTION_NEEDS_HUMAN_REVIEW,
)

STATUS_ACCEPTED = "accepted"
STATUS_NOT_READY = "not_ready"
STATUS_NEEDS_HUMAN_REVIEW = "needs_human_review"
STATUS_TRY_ALTERNATIVE_MODEL = "try_alternative_model"

FINAL_POST_FLOW_STATUSES = (
    STATUS_ACCEPTED,
    STATUS_NOT_READY,
    STATUS_NEEDS_HUMAN_REVIEW,
    STATUS_TRY_ALTERNATIVE_MODEL,
)


@dataclass(frozen=True)
class FinalPostDecision:
    action: str
    reason: str
    repair_type: str | None
    target_model_provider: str | None
    target_model_name: str | None
    needs_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "repair_type": self.repair_type,
            "target_model_provider": self.target_model_provider,
            "target_model_name": self.target_model_name,
            "needs_human_review": self.needs_human_review,
        }


@dataclass(frozen=True)
class FinalPostAttempt:
    attempt_index: int
    payload: dict
    validation_passed: bool
    validation_error: str
    diagnostics: FinalPostDiagnostics | None
    quality_review: dict | None
    repair_plan: dict | None
    decision: FinalPostDecision | None
    provider: str | None
    model: str | None
    prompt_name: str | None
    prompt_version: str | None
    token_usage: dict | None
    cost_metadata: dict | None
    created_at: str | None
    parent_attempt_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "payload": _serialize_value(self.payload),
            "validation_passed": self.validation_passed,
            "validation_error": self.validation_error,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "quality_review": _serialize_value(self.quality_review),
            "repair_plan": _serialize_value(self.repair_plan),
            "decision": self.decision.to_dict() if self.decision else None,
            "provider": self.provider,
            "model": self.model,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "token_usage": _serialize_value(self.token_usage),
            "cost_metadata": _serialize_value(self.cost_metadata),
            "created_at": self.created_at,
            "parent_attempt_index": self.parent_attempt_index,
        }


@dataclass(frozen=True)
class FinalPostAttemptHistory:
    attempts: list[FinalPostAttempt]

    def add_attempt(self, attempt: FinalPostAttempt) -> "FinalPostAttemptHistory":
        return FinalPostAttemptHistory(attempts=[*self.attempts, attempt])

    def latest_attempt(self) -> FinalPostAttempt | None:
        if not self.attempts:
            return None
        return self.attempts[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class FinalPostFlowResult:
    status: str
    accepted_payload: dict | None
    attempt_history: FinalPostAttemptHistory
    final_decision: FinalPostDecision
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "accepted_payload": _serialize_value(self.accepted_payload),
            "attempt_history": self.attempt_history.to_dict(),
            "final_decision": self.final_decision.to_dict(),
            "reason": self.reason,
        }


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
