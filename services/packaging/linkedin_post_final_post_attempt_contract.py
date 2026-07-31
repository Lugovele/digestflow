"""Spec-only contracts for one standalone final-post attempt.

These contracts describe the request and result envelope for a future
single-attempt orchestration slice. They do not execute prompts, call providers,
parse responses, adapt payloads, run gates, evaluate quality, adjudicate
outcomes, persist data, or connect to runtime packaging.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
import math
from typing import Any


STAGE_CANDIDATE_WRITER_REQUEST = "candidate_writer_request"
STAGE_CANDIDATE_WRITER_EXECUTION = "candidate_writer_execution"
STAGE_CANDIDATE_WRITER_PARSE = "candidate_writer_parse"
STAGE_CANDIDATE_WRITER_ADAPTATION = "candidate_writer_adaptation"
STAGE_DETERMINISTIC_GATE = "deterministic_gate"
STAGE_SEMANTIC_GROUNDING_REQUEST = "semantic_grounding_request"
STAGE_SEMANTIC_GROUNDING_EXECUTION = "semantic_grounding_execution"
STAGE_SEMANTIC_GROUNDING_PARSE = "semantic_grounding_parse"
STAGE_SEMANTIC_GROUNDING_NORMALIZATION = "semantic_grounding_normalization"
STAGE_QUALITY_EVALUATOR_REQUEST = "quality_evaluator_request"
STAGE_QUALITY_EVALUATOR_EXECUTION = "quality_evaluator_execution"
STAGE_QUALITY_EVALUATOR_PARSE = "quality_evaluator_parse"
STAGE_QUALITY_REVIEW_NORMALIZATION = "quality_review_normalization"
STAGE_ATTEMPT_ADJUDICATION = "attempt_adjudication"
STAGE_ATTEMPT_OUTCOME = "attempt_outcome"

FINAL_POST_ATTEMPT_STAGE_ORDER = (
    STAGE_CANDIDATE_WRITER_REQUEST,
    STAGE_CANDIDATE_WRITER_EXECUTION,
    STAGE_CANDIDATE_WRITER_PARSE,
    STAGE_CANDIDATE_WRITER_ADAPTATION,
    STAGE_DETERMINISTIC_GATE,
    STAGE_SEMANTIC_GROUNDING_REQUEST,
    STAGE_SEMANTIC_GROUNDING_EXECUTION,
    STAGE_SEMANTIC_GROUNDING_PARSE,
    STAGE_SEMANTIC_GROUNDING_NORMALIZATION,
    STAGE_QUALITY_EVALUATOR_REQUEST,
    STAGE_QUALITY_EVALUATOR_EXECUTION,
    STAGE_QUALITY_EVALUATOR_PARSE,
    STAGE_QUALITY_REVIEW_NORMALIZATION,
    STAGE_ATTEMPT_ADJUDICATION,
    STAGE_ATTEMPT_OUTCOME,
)

STATUS_NOT_STARTED = "not_started"
STATUS_READY = "ready"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

FAILURE_CANDIDATE_WRITER_REQUEST = "candidate_writer_request_failure"
FAILURE_CANDIDATE_WRITER_PROVIDER = "candidate_writer_provider_failure"
FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE = "candidate_writer_empty_response"
FAILURE_CANDIDATE_WRITER_PARSE = "candidate_writer_parse_failure"
FAILURE_CANDIDATE_WRITER_ADAPTATION = "candidate_writer_adaptation_failure"
FAILURE_DETERMINISTIC_GATE = "deterministic_gate_failure"
FAILURE_SEMANTIC_GROUNDING_REQUEST = "semantic_grounding_request_failure"
FAILURE_SEMANTIC_GROUNDING_PROVIDER = "semantic_grounding_provider_failure"
FAILURE_SEMANTIC_GROUNDING_EMPTY_RESPONSE = "semantic_grounding_empty_response"
FAILURE_SEMANTIC_GROUNDING_PARSE = "semantic_grounding_parse_failure"
FAILURE_SEMANTIC_GROUNDING_NORMALIZATION = "semantic_grounding_normalization_failure"
FAILURE_SEMANTIC_GROUNDING = "semantic_grounding_failure"
FAILURE_QUALITY_EVALUATOR_REQUEST = "quality_evaluator_request_failure"
FAILURE_QUALITY_EVALUATOR_PROVIDER = "quality_evaluator_provider_failure"
FAILURE_QUALITY_EVALUATOR_EMPTY_RESPONSE = "quality_evaluator_empty_response"
FAILURE_QUALITY_EVALUATOR_PARSE = "quality_evaluator_parse_failure"
FAILURE_QUALITY_REVIEW_NORMALIZATION = "quality_review_normalization_failure"


@dataclass(frozen=True)
class FinalPostAttemptRequest:
    """Inputs required to run one future standalone final-post attempt."""

    candidate_writer_render: object | dict
    candidate_writer_prompt_text: str
    quality_rubric: object | dict
    quality_evaluator_prompt_text: str
    attempt_index: int
    max_attempts: int
    attempt_history: object | dict | None = None
    candidate_writer_provider: str | None = None
    candidate_writer_model: str | None = None
    candidate_writer_max_output_tokens: int | None = None
    semantic_grounding_prompt_text: str = ""
    semantic_grounding_provider: str | None = None
    semantic_grounding_model: str | None = None
    semantic_grounding_max_output_tokens: int | None = None
    quality_evaluator_provider: str | None = None
    quality_evaluator_model: str | None = None
    quality_evaluator_max_output_tokens: int | None = None
    created_at: str | None = None
    parent_attempt_index: int | None = None
    policy: object | dict | None = None
    alternative_model_available: bool = False
    target_model_provider: str | None = None
    target_model_name: str | None = None
    execution_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_writer_render": _serialize_attempt_value(
                self.candidate_writer_render
            ),
            "candidate_writer_prompt_text": _serialize_attempt_value(
                self.candidate_writer_prompt_text
            ),
            "quality_rubric": _serialize_attempt_value(self.quality_rubric),
            "quality_evaluator_prompt_text": _serialize_attempt_value(
                self.quality_evaluator_prompt_text
            ),
            "attempt_index": _serialize_attempt_value(self.attempt_index),
            "max_attempts": _serialize_attempt_value(self.max_attempts),
            "attempt_history": _serialize_attempt_value(self.attempt_history),
            "candidate_writer_provider": _serialize_attempt_value(
                self.candidate_writer_provider
            ),
            "candidate_writer_model": _serialize_attempt_value(
                self.candidate_writer_model
            ),
            "candidate_writer_max_output_tokens": _serialize_attempt_value(
                self.candidate_writer_max_output_tokens
            ),
            "semantic_grounding_prompt_text": _serialize_attempt_value(
                self.semantic_grounding_prompt_text
            ),
            "semantic_grounding_provider": _serialize_attempt_value(
                self.semantic_grounding_provider
            ),
            "semantic_grounding_model": _serialize_attempt_value(
                self.semantic_grounding_model
            ),
            "semantic_grounding_max_output_tokens": (
                _serialize_attempt_value(self.semantic_grounding_max_output_tokens)
            ),
            "quality_evaluator_provider": _serialize_attempt_value(
                self.quality_evaluator_provider
            ),
            "quality_evaluator_model": _serialize_attempt_value(
                self.quality_evaluator_model
            ),
            "quality_evaluator_max_output_tokens": (
                _serialize_attempt_value(self.quality_evaluator_max_output_tokens)
            ),
            "created_at": _serialize_attempt_value(self.created_at),
            "parent_attempt_index": _serialize_attempt_value(
                self.parent_attempt_index
            ),
            "policy": _serialize_attempt_value(self.policy),
            "alternative_model_available": _serialize_attempt_value(
                self.alternative_model_available
            ),
            "target_model_provider": _serialize_attempt_value(
                self.target_model_provider
            ),
            "target_model_name": _serialize_attempt_value(self.target_model_name),
            "execution_metadata": _serialize_attempt_value(self.execution_metadata),
        }


@dataclass(frozen=True)
class FinalPostAttemptStageStatus:
    """Status for one stage in a future standalone final-post attempt."""

    stage: str
    status: str
    error_code: str | None = None
    error_message: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": _serialize_attempt_value(self.stage),
            "status": _serialize_attempt_value(self.status),
            "error_code": _serialize_attempt_value(self.error_code),
            "error_message": _serialize_attempt_value(self.error_message),
            "metadata": _serialize_attempt_value(self.metadata),
        }


@dataclass(frozen=True)
class FinalPostStandaloneAttemptResult:
    """Audit-safe result envelope for one future standalone final-post attempt."""

    request: FinalPostAttemptRequest
    stage_statuses: tuple[FinalPostAttemptStageStatus, ...]
    completed_stage: str | None = None
    failure_stage: str | None = None
    failure_code: str | None = None
    failure_message: str = ""
    candidate_writer_raw_response: object | dict | None = None
    parsed_candidate: dict[str, Any] | None = None
    candidate_writer_output: object | dict | None = None
    deterministic_gate_output: object | dict | None = None
    semantic_grounding_prompt_render: object | dict | None = None
    semantic_grounding_raw_response: object | dict | None = None
    semantic_grounding_state: object | dict | None = None
    post_editorial_input: object | dict | None = None
    quality_evaluator_prompt_render: object | dict | None = None
    quality_evaluator_raw_response: object | dict | None = None
    quality_evaluation_state: object | dict | None = None
    final_attempt_outcome: object | dict | None = None
    candidate_writer_invocation_count: int = 0
    semantic_grounding_invocation_count: int = 0
    quality_evaluator_invocation_count: int = 0
    repair_invocation_count: int = 0
    audit_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.repair_invocation_count != 0:
            raise ValueError(
                "FinalPostStandaloneAttemptResult does not allow repair invocation."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": _serialize_attempt_value(self.request.to_dict()),
            "stage_order": _serialize_attempt_value(
                list(FINAL_POST_ATTEMPT_STAGE_ORDER)
            ),
            "stage_statuses": _serialize_attempt_value([
                stage_status.to_dict() for stage_status in self.stage_statuses
            ]),
            "completed_stage": _serialize_attempt_value(self.completed_stage),
            "failure_stage": _serialize_attempt_value(self.failure_stage),
            "failure_code": _serialize_attempt_value(self.failure_code),
            "failure_message": _serialize_attempt_value(self.failure_message),
            "candidate_writer_raw_response": _serialize_attempt_value(
                self.candidate_writer_raw_response
            ),
            "parsed_candidate": _serialize_attempt_value(self.parsed_candidate),
            "candidate_writer_output": _serialize_attempt_value(
                self.candidate_writer_output
            ),
            "deterministic_gate_output": _serialize_attempt_value(
                self.deterministic_gate_output
            ),
            "semantic_grounding_prompt_render": _serialize_attempt_value(
                self.semantic_grounding_prompt_render
            ),
            "semantic_grounding_raw_response": _serialize_attempt_value(
                self.semantic_grounding_raw_response
            ),
            "semantic_grounding_state": _serialize_attempt_value(
                self.semantic_grounding_state
            ),
            "post_editorial_input": _serialize_attempt_value(self.post_editorial_input),
            "quality_evaluator_prompt_render": _serialize_attempt_value(
                self.quality_evaluator_prompt_render
            ),
            "quality_evaluator_raw_response": _serialize_attempt_value(
                self.quality_evaluator_raw_response
            ),
            "quality_evaluation_state": _serialize_attempt_value(
                self.quality_evaluation_state
            ),
            "final_attempt_outcome": _serialize_attempt_value(
                self.final_attempt_outcome
            ),
            "candidate_writer_invocation_count": _serialize_attempt_value(
                self.candidate_writer_invocation_count
            ),
            "semantic_grounding_invocation_count": (
                _serialize_attempt_value(self.semantic_grounding_invocation_count)
            ),
            "quality_evaluator_invocation_count": (
                _serialize_attempt_value(self.quality_evaluator_invocation_count)
            ),
            "repair_invocation_count": _serialize_attempt_value(
                self.repair_invocation_count
            ),
            "audit_metadata": _serialize_attempt_value(self.audit_metadata),
        }


def _serialize_attempt_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(
                "Unsupported non-finite final-post attempt contract float."
            )
        return value
    if isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize_attempt_value(value.to_dict())
    if is_dataclass(value):
        return _serialize_attempt_value(asdict(value))
    if isinstance(value, dict):
        return {
            _serialize_attempt_mapping_key(key): _serialize_attempt_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_serialize_attempt_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_attempt_value(item) for item in value]
    raise TypeError(
        f"Unsupported non-JSON final-post attempt contract value: "
        f"{type(value).__name__}"
    )


def _serialize_attempt_mapping_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    raise TypeError(
        f"Unsupported non-string final-post attempt contract mapping key: "
        f"{type(key).__name__}"
    )
