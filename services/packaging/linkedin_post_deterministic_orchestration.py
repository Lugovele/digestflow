"""Small deterministic orchestration slice for final LinkedIn post candidates.

This module connects an existing PostBrief and CandidateWriterOutput to the
deterministic gate and decision controller. It does not execute prompts, call
providers, repair payloads, persist data, or touch production packaging.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from typing import Any

from services.packaging.linkedin_post_deterministic_gate import (
    run_final_post_deterministic_gate,
)
from services.packaging.linkedin_post_flow_contracts import (
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


@dataclass(frozen=True)
class FinalPostDecisionReadyResult:
    """Decision-controller output plus the deterministic facts it used."""

    post_brief: Any
    candidate_output: CandidateWriterOutput
    gate_output: DeterministicGateOutput
    decision: FinalPostDecision
    quality_review: dict | None
    attempt_history: FinalPostAttemptHistory

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_brief": _serialize_value(self.post_brief),
            "candidate_output": self.candidate_output.to_dict(),
            "gate_output": self.gate_output.to_dict(),
            "decision": self.decision.to_dict(),
            "quality_review": _serialize_value(self.quality_review),
            "attempt_history": self.attempt_history.to_dict(),
        }


def prepare_final_post_decision_ready_result(
    *,
    post_brief: Any,
    candidate_output: CandidateWriterOutput,
    quality_review: dict | None = None,
    attempt_history: FinalPostAttemptHistory | None = None,
    decision_controller: FinalPostDecisionController | None = None,
    policy: FinalPostDecisionPolicy | None = None,
    alternative_model_available: bool = False,
    target_model_provider: str | None = None,
    target_model_name: str | None = None,
) -> FinalPostDecisionReadyResult:
    """Run gate and decision routing without crossing into runtime execution."""

    selected_evidence_ids = _selected_evidence_ids_from_post_brief(post_brief)
    gate_output = run_final_post_deterministic_gate(
        candidate_output,
        selected_evidence_ids=selected_evidence_ids,
    )
    history = attempt_history or FinalPostAttemptHistory(attempts=[])
    controller = decision_controller or FinalPostDecisionController()
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

    return FinalPostDecisionReadyResult(
        post_brief=post_brief,
        candidate_output=candidate_output,
        gate_output=gate_output,
        decision=decision,
        quality_review=quality_review,
        attempt_history=history,
    )


def _selected_evidence_ids_from_post_brief(post_brief: Any) -> tuple[str, ...]:
    evidence_to_use = _get_field(post_brief, "evidence_to_use")
    if not isinstance(evidence_to_use, (list, tuple)) or not evidence_to_use:
        raise ValueError("PostBrief.evidence_to_use must be a non-empty list or tuple.")

    selected_ids: list[str] = []
    for item_index, item in enumerate(evidence_to_use):
        evidence_id = _get_field(item, "evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError(
                f"PostBrief.evidence_to_use[{item_index}].evidence_id must be a non-empty string."
            )
        selected_ids.append(evidence_id)
    return tuple(selected_ids)


def _get_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return _serialize_value(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    return value
