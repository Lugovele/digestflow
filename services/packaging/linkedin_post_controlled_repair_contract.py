"""Contracts for one controlled final-post repair attempt.

This module is an audit envelope for a two-attempt standalone flow. It does not
execute prompts, call providers, parse responses, repair text, persist data, or
connect to runtime packaging.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
import math
from typing import Any

from services.packaging.linkedin_post_final_post_attempt_contract import (
    FinalPostAttemptRequest,
)


REPAIR_ELIGIBLE = "eligible"
REPAIR_INELIGIBLE = "ineligible"

FAILURE_REPAIR_INELIGIBLE = "repair_ineligible"
FAILURE_REPAIR_WRITER_REQUEST = "repair_writer_request_failure"
FAILURE_REPAIR_WRITER_PROVIDER = "repair_writer_provider_failure"
FAILURE_REPAIR_WRITER_EMPTY_RESPONSE = "repair_writer_empty_response"
FAILURE_REPAIR_WRITER_PARSE = "repair_writer_parse_failure"
FAILURE_REPAIR_WRITER_ADAPTATION = "repair_writer_adaptation_failure"
FAILURE_REPAIR_TARGET_NOT_FIXED = "repair_target_not_fixed"
FAILURE_REPAIRED_DETERMINISTIC_GATE = "repaired_deterministic_gate_failure"
FAILURE_REPAIRED_SEMANTIC_GROUNDING_REQUEST = (
    "repaired_semantic_grounding_request_failure"
)
FAILURE_REPAIRED_SEMANTIC_GROUNDING_PROVIDER = (
    "repaired_semantic_grounding_provider_failure"
)
FAILURE_REPAIRED_SEMANTIC_GROUNDING_EMPTY_RESPONSE = (
    "repaired_semantic_grounding_empty_response"
)
FAILURE_REPAIRED_SEMANTIC_GROUNDING_PARSE = (
    "repaired_semantic_grounding_parse_failure"
)
FAILURE_REPAIRED_SEMANTIC_GROUNDING_NORMALIZATION = (
    "repaired_semantic_grounding_normalization_failure"
)
FAILURE_REPAIRED_SEMANTIC_GROUNDING = "repaired_semantic_grounding_failure"
FAILURE_REPAIRED_QUALITY_EVALUATOR_REQUEST = (
    "repaired_quality_evaluator_request_failure"
)
FAILURE_REPAIRED_QUALITY_EVALUATOR_PROVIDER = (
    "repaired_quality_evaluator_provider_failure"
)
FAILURE_REPAIRED_QUALITY_EVALUATOR_EMPTY_RESPONSE = (
    "repaired_quality_evaluator_empty_response"
)
FAILURE_REPAIRED_QUALITY_EVALUATOR_PARSE = "repaired_quality_evaluator_parse_failure"
FAILURE_REPAIRED_QUALITY_REVIEW_NORMALIZATION = (
    "repaired_quality_review_normalization_failure"
)


@dataclass(frozen=True)
class FinalPostControlledRepairRequest:
    initial_attempt_request: FinalPostAttemptRequest
    repair_prompt_text: str
    repair_provider: str | None
    repair_model: str | None
    repair_max_output_tokens: int | None = None
    repair_execution_path: str | None = None
    repair_thinking_budget: int | None = None
    repair_enabled: bool = True
    max_controlled_attempts: int = 2
    execution_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_attempt_request": _serialize_repair_value(
                self.initial_attempt_request
            ),
            "repair_prompt_text": _serialize_repair_value(self.repair_prompt_text),
            "repair_provider": _serialize_repair_value(self.repair_provider),
            "repair_model": _serialize_repair_value(self.repair_model),
            "repair_max_output_tokens": _serialize_repair_value(
                self.repair_max_output_tokens
            ),
            "repair_execution_path": _serialize_repair_value(
                self.repair_execution_path
            ),
            "repair_thinking_budget": _serialize_repair_value(
                self.repair_thinking_budget
            ),
            "repair_enabled": _serialize_repair_value(self.repair_enabled),
            "max_controlled_attempts": _serialize_repair_value(
                self.max_controlled_attempts
            ),
            "execution_metadata": _serialize_repair_value(self.execution_metadata),
        }


@dataclass(frozen=True)
class FinalPostRepairEligibility:
    status: str
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FinalPostControlledRepairResult:
    request: FinalPostControlledRepairRequest
    initial_attempt_result: object
    repair_eligibility: FinalPostRepairEligibility
    repair_executed: bool
    repair_prompt_render: object | dict | None = None
    repair_writer_raw_response: object | dict | None = None
    parsed_repair_candidate: dict[str, Any] | None = None
    repair_writer_structural_diagnostics: object | dict | None = None
    repaired_candidate_output: object | dict | None = None
    repaired_deterministic_gate_output: object | dict | None = None
    repaired_semantic_grounding_prompt_render: object | dict | None = None
    repaired_semantic_grounding_raw_response: object | dict | None = None
    repaired_semantic_grounding_state: object | dict | None = None
    repaired_post_editorial_input: object | dict | None = None
    repaired_quality_evaluator_prompt_render: object | dict | None = None
    repaired_quality_evaluator_raw_response: object | dict | None = None
    repaired_quality_evaluation_state: object | dict | None = None
    repaired_attempt_outcome: object | dict | None = None
    repair_target_enforcement_diagnostics: object | dict | None = None
    accepted_payload: dict[str, Any] | None = None
    terminal_outcome: str | None = None
    terminal_reason: str = ""
    failure_stage: str | None = None
    failure_code: str | None = None
    failure_message: str = ""
    candidate_writer_invocation_count: int = 0
    semantic_grounding_invocation_count: int = 0
    repair_invocation_count: int = 0
    quality_evaluator_invocation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "initial_attempt_result": _serialize_repair_value(
                self.initial_attempt_result
            ),
            "repair_eligibility": self.repair_eligibility.to_dict(),
            "repair_executed": self.repair_executed,
            "repair_prompt_render": _serialize_repair_value(self.repair_prompt_render),
            "repair_writer_raw_response": _serialize_repair_value(
                self.repair_writer_raw_response
            ),
            "parsed_repair_candidate": _serialize_repair_value(
                self.parsed_repair_candidate
            ),
            "repair_writer_structural_diagnostics": _serialize_repair_value(
                self.repair_writer_structural_diagnostics
            ),
            "repaired_candidate_output": _serialize_repair_value(
                self.repaired_candidate_output
            ),
            "repaired_deterministic_gate_output": _serialize_repair_value(
                self.repaired_deterministic_gate_output
            ),
            "repaired_semantic_grounding_prompt_render": _serialize_repair_value(
                self.repaired_semantic_grounding_prompt_render
            ),
            "repaired_semantic_grounding_raw_response": _serialize_repair_value(
                self.repaired_semantic_grounding_raw_response
            ),
            "repaired_semantic_grounding_state": _serialize_repair_value(
                self.repaired_semantic_grounding_state
            ),
            "repaired_post_editorial_input": _serialize_repair_value(
                self.repaired_post_editorial_input
            ),
            "repaired_quality_evaluator_prompt_render": _serialize_repair_value(
                self.repaired_quality_evaluator_prompt_render
            ),
            "repaired_quality_evaluator_raw_response": _serialize_repair_value(
                self.repaired_quality_evaluator_raw_response
            ),
            "repaired_quality_evaluation_state": _serialize_repair_value(
                self.repaired_quality_evaluation_state
            ),
            "repaired_attempt_outcome": _serialize_repair_value(
                self.repaired_attempt_outcome
            ),
            "repair_target_enforcement_diagnostics": _serialize_repair_value(
                self.repair_target_enforcement_diagnostics
            ),
            "accepted_payload": _serialize_repair_value(self.accepted_payload),
            "terminal_outcome": self.terminal_outcome,
            "terminal_reason": self.terminal_reason,
            "failure_stage": self.failure_stage,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "candidate_writer_invocation_count": self.candidate_writer_invocation_count,
            "semantic_grounding_invocation_count": (
                self.semantic_grounding_invocation_count
            ),
            "repair_invocation_count": self.repair_invocation_count,
            "quality_evaluator_invocation_count": self.quality_evaluator_invocation_count,
        }


def _serialize_repair_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Unsupported non-finite controlled-repair float.")
        return value
    if isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize_repair_value(value.to_dict())
    if is_dataclass(value):
        return _serialize_repair_value(asdict(value))
    if isinstance(value, dict):
        return {
            _serialize_repair_mapping_key(key): _serialize_repair_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_serialize_repair_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_repair_value(item) for item in value]
    raise TypeError(
        f"Unsupported non-JSON controlled-repair value: {type(value).__name__}"
    )


def _serialize_repair_mapping_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    raise TypeError(
        f"Unsupported non-string controlled-repair mapping key: {type(key).__name__}"
    )
