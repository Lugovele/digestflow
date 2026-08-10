"""Spec-only handoff contracts for future final LinkedIn post flow agents.

PostEditorialInput is the current perspective boundary. It carries AngleDecision,
including author position and controlling angle. If author_take.core_opinion is
reintroduced as a separate source of truth, it should be added to
PostEditorialInput in a separate architecture step.

Payload snapshots are represented as dictionaries at this spec stage. First-attempt Candidate Writer snapshots contain the core CandidatePost shape {"post_text": ...}. Stricter
snapshot immutability can be added when orchestration is implemented.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from typing import Any

from services.packaging.linkedin_final_post_diagnostics import FinalPostDiagnostics
from services.packaging.linkedin_post_editorial_boundary import PostEditorialInput
from services.packaging.linkedin_post_flow_contracts import (
    FinalPostAttemptHistory,
    FinalPostDecision,
)


@dataclass(frozen=True)
class QualityReviewResult:
    """Quality review result for the 9-criterion / 45-point LinkedIn rubric."""

    scores: dict
    total_score: int
    passed: bool
    failed_criteria: tuple[str, ...]
    automatic_fail_reason: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": _serialize_handoff_value(self.scores),
            "total_score": self.total_score,
            "pass": self.passed,
            "failed_criteria": list(self.failed_criteria),
            "automatic_fail_reason": self.automatic_fail_reason,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TargetedRepairPlan:
    """Instruction-only repair plan for FinalPostRepairAgent."""

    failed_criterion: str
    repair_scope: str
    repair_instruction: str
    preserve: tuple[str, ...]
    avoid: tuple[str, ...]
    repair_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "failed_criterion": self.failed_criterion,
            "repair_scope": self.repair_scope,
            "repair_instruction": self.repair_instruction,
            "preserve": list(self.preserve),
            "avoid": list(self.avoid),
            "repair_type": self.repair_type,
        }


@dataclass(frozen=True)
class CandidateWriterOutput:
    """New core CandidatePost payload created by FinalPostCandidateWriter.

    This is not a mutation of an existing payload. The next required handoff is
    FinalPostDeterministicGate.
    """

    payload: dict
    raw_output: str | None
    provider: str | None
    model: str | None
    prompt_name: str | None
    prompt_version: str | None
    token_usage: dict | None
    cost_metadata: dict | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": _serialize_handoff_value(self.payload),
            "raw_output": self.raw_output,
            "provider": self.provider,
            "model": self.model,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "token_usage": _serialize_handoff_value(self.token_usage),
            "cost_metadata": _serialize_handoff_value(self.cost_metadata),
        }


@dataclass(frozen=True)
class DeterministicGateOutput:
    """Validation and diagnostics output from FinalPostDeterministicGate.

    The gate does not modify the payload. This output must be recorded in
    attempt history and may be used to build PostEditorialInput only after the
    gate has run.
    """

    payload: dict
    validation_passed: bool
    validation_error: str
    diagnostics: FinalPostDiagnostics
    selected_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": _serialize_handoff_value(self.payload),
            "validation_passed": self.validation_passed,
            "validation_error": self.validation_error,
            "diagnostics": self.diagnostics.to_dict(),
            "selected_evidence_ids": list(self.selected_evidence_ids),
        }


@dataclass(frozen=True)
class RepairPlannerInput:
    """Input to FinalPostRepairPlanner for producing a TargetedRepairPlan.

    This contract carries context for planning only. It must not rewrite the
    candidate payload or modify evidence, PostBrief, or AngleDecision.
    """

    post_editorial_input: PostEditorialInput
    decision: FinalPostDecision
    quality_review: QualityReviewResult | None
    attempt_history: FinalPostAttemptHistory

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_editorial_input": self.post_editorial_input.to_dict(),
            "decision": self.decision.to_dict(),
            "quality_review": _serialize_handoff_value(self.quality_review),
            "attempt_history": self.attempt_history.to_dict(),
        }


@dataclass(frozen=True)
class RepairAgentInput:
    """Input to FinalPostRepairAgent with an explicit repair plan.

    The repair agent must not guess repair reasons. It may create only a
    revised FinalPostPayload.
    """

    post_editorial_input: PostEditorialInput
    repair_plan: TargetedRepairPlan
    attempt_index: int
    max_attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_editorial_input": self.post_editorial_input.to_dict(),
            "repair_plan": _serialize_handoff_value(self.repair_plan),
            "attempt_index": self.attempt_index,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True)
class RepairAgentOutput:
    """New revised payload created by FinalPostRepairAgent.

    This is not an in-place mutation and does not include an acceptance
    decision. The next required handoff is FinalPostDeterministicGate.
    """

    payload: dict
    raw_output: str | None
    provider: str | None
    model: str | None
    prompt_name: str | None
    prompt_version: str | None
    token_usage: dict | None
    cost_metadata: dict | None
    parent_attempt_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": _serialize_handoff_value(self.payload),
            "raw_output": self.raw_output,
            "provider": self.provider,
            "model": self.model,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "token_usage": _serialize_handoff_value(self.token_usage),
            "cost_metadata": _serialize_handoff_value(self.cost_metadata),
            "parent_attempt_index": self.parent_attempt_index,
        }


def _serialize_handoff_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return _serialize_handoff_value(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize_handoff_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_handoff_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_handoff_value(item) for item in value]
    return value
