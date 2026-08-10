"""Adapter from parsed Candidate Writer output to core CandidatePost handoff.

This module turns strict parsed Candidate Writer JSON into a canonical
CandidatePost-shaped dictionary and, when paired with a raw response, a
CandidateWriterOutput handoff. It does not parse JSON, call providers, run the
deterministic gate, evaluate quality, repair text, persist data, or connect to
runtime packaging.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from services.packaging.linkedin_post_candidate_post_contract import (
    CANDIDATE_POST_FIELDS,
    CandidatePostContractError,
    candidate_post_from_dict,
    candidate_post_to_dict,
)
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    ADAPTER_ERROR_INVALID_FIELD_TYPES,
    ADAPTER_ERROR_INVALID_FIELD_VALUES,
    ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
    ADAPTER_ERROR_PAYLOAD_CONTRACT_VIOLATION,
    ADAPTER_ERROR_TOP_LEVEL_NOT_OBJECT,
    ADAPTER_ERROR_UNKNOWN,
    FIELD_VIOLATION_ABOVE_MAX_LENGTH,
    FIELD_VIOLATION_EMPTY_STRING,
    FIELD_VIOLATION_WRONG_TYPE,
    CandidateWriterFieldViolation,
    CandidateWriterStructuralDiagnostics,
    build_adapter_structural_diagnostics,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput

if TYPE_CHECKING:
    from services.packaging.linkedin_post_candidate_writer_execution import (
        CandidateWriterRawResponse,
    )

__all__ = (
    "CANONICAL_CANDIDATE_POST_FIELDS",
    "CandidateWriterOutputAdaptationError",
    "adapt_candidate_writer_payload",
    "build_candidate_writer_output_from_parsed_response",
)


CANONICAL_CANDIDATE_POST_FIELDS = CANDIDATE_POST_FIELDS
CANONICAL_FINAL_POST_PAYLOAD_FIELDS = CANONICAL_CANDIDATE_POST_FIELDS
REQUIRED_CANDIDATE_POST_FIELDS = ("post_text",)
ERROR_INVALID_PARSED_CANDIDATE = "invalid_parsed_candidate"
ERROR_MISSING_REQUIRED_FIELD = "missing_required_field"
ERROR_INVALID_CANDIDATE_POST = "invalid_candidate_post"
ERROR_INVALID_FINAL_POST_PAYLOAD = ERROR_INVALID_CANDIDATE_POST
ERROR_RAW_RESPONSE_EXECUTION_ERROR = "raw_response_execution_error"
ERROR_INVALID_RAW_RESPONSE = "invalid_raw_response"

REQUIRED_RAW_RESPONSE_ATTRIBUTES = (
    "execution_error",
    "prompt_metadata",
    "raw_text",
    "provider",
    "model",
    "usage",
)
REQUIRED_PROMPT_METADATA_ATTRIBUTES = (
    "prompt_name",
    "prompt_version",
)


class CandidateWriterOutputAdaptationError(ValueError):
    """Raised when parsed Candidate Writer output cannot become a handoff."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        safe_details: dict[str, Any] | None = None,
        diagnostics: CandidateWriterStructuralDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_details = copy.deepcopy(safe_details) if safe_details else {}
        self.diagnostics = diagnostics


def adapt_candidate_writer_payload(parsed_candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical CandidatePost-compatible dictionary."""

    if not isinstance(parsed_candidate, dict):
        raise CandidateWriterOutputAdaptationError(
            ERROR_INVALID_PARSED_CANDIDATE,
            "parsed candidate must be a dictionary.",
            diagnostics=build_adapter_structural_diagnostics(
                adapter_error_code=ADAPTER_ERROR_TOP_LEVEL_NOT_OBJECT,
                parsed_candidate=parsed_candidate,
                required_fields=REQUIRED_CANDIDATE_POST_FIELDS,
                allowed_fields=CANONICAL_CANDIDATE_POST_FIELDS,
            ),
        )

    missing_fields = [
        field for field in REQUIRED_CANDIDATE_POST_FIELDS if field not in parsed_candidate
    ]
    if missing_fields:
        raise CandidateWriterOutputAdaptationError(
            ERROR_MISSING_REQUIRED_FIELD,
            f"parsed candidate is missing required fields: {missing_fields}.",
            safe_details={
                "error_code": ERROR_MISSING_REQUIRED_FIELD,
                "missing_fields": list(missing_fields),
                "required_fields": list(REQUIRED_CANDIDATE_POST_FIELDS),
            },
            diagnostics=build_adapter_structural_diagnostics(
                adapter_error_code=ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
                parsed_candidate=parsed_candidate,
                required_fields=REQUIRED_CANDIDATE_POST_FIELDS,
                allowed_fields=CANONICAL_CANDIDATE_POST_FIELDS,
            ),
        )

    unexpected_fields = sorted(set(parsed_candidate) - set(CANONICAL_CANDIDATE_POST_FIELDS))
    if unexpected_fields:
        raise CandidateWriterOutputAdaptationError(
            ERROR_INVALID_CANDIDATE_POST,
            f"parsed candidate contains unexpected fields: {unexpected_fields}.",
            safe_details={
                "error_code": ERROR_INVALID_CANDIDATE_POST,
                "unexpected_fields": list(unexpected_fields),
                **_safe_validation_details(parsed_candidate, ValueError("unexpected fields")),
            },
            diagnostics=build_adapter_structural_diagnostics(
                adapter_error_code=ADAPTER_ERROR_PAYLOAD_CONTRACT_VIOLATION,
                parsed_candidate=parsed_candidate,
                required_fields=REQUIRED_CANDIDATE_POST_FIELDS,
                allowed_fields=CANONICAL_CANDIDATE_POST_FIELDS,
                invalid_field_names=unexpected_fields,
            ),
        )

    try:
        candidate_post = candidate_post_from_dict(copy.deepcopy(parsed_candidate))
    except (TypeError, CandidatePostContractError) as exc:
        raise CandidateWriterOutputAdaptationError(
            ERROR_INVALID_CANDIDATE_POST,
            "parsed candidate does not satisfy CandidatePost structure.",
            safe_details=_safe_validation_details(parsed_candidate, exc),
            diagnostics=build_adapter_structural_diagnostics(
                adapter_error_code=_adapter_error_code_for_validation_failure(parsed_candidate),
                parsed_candidate=parsed_candidate,
                required_fields=REQUIRED_CANDIDATE_POST_FIELDS,
                allowed_fields=CANONICAL_CANDIDATE_POST_FIELDS,
                invalid_field_names=_invalid_field_names_for_validation_failure(parsed_candidate),
                field_violations=_field_violations_for_validation_failure(parsed_candidate),
            ),
        ) from exc

    return copy.deepcopy(candidate_post_to_dict(candidate_post))


def build_candidate_writer_output_from_parsed_response(
    *,
    parsed_candidate: dict[str, Any],
    raw_response: "CandidateWriterRawResponse",
) -> CandidateWriterOutput:
    """Build a CandidateWriterOutput from parsed output and raw metadata."""

    _require_attributes(raw_response, REQUIRED_RAW_RESPONSE_ATTRIBUTES, "raw response")

    if raw_response.execution_error:
        raise CandidateWriterOutputAdaptationError(
            ERROR_RAW_RESPONSE_EXECUTION_ERROR,
            "raw response has an execution error and cannot build candidate output.",
            diagnostics=build_adapter_structural_diagnostics(
                adapter_error_code=ADAPTER_ERROR_UNKNOWN,
                parsed_candidate=parsed_candidate,
                required_fields=REQUIRED_CANDIDATE_POST_FIELDS,
                allowed_fields=CANONICAL_CANDIDATE_POST_FIELDS,
            ),
        )

    payload = adapt_candidate_writer_payload(parsed_candidate)
    prompt_metadata = raw_response.prompt_metadata
    if prompt_metadata is not None:
        _require_attributes(prompt_metadata, REQUIRED_PROMPT_METADATA_ATTRIBUTES, "prompt metadata")

    return CandidateWriterOutput(
        payload=payload,
        raw_output=raw_response.raw_text,
        provider=raw_response.provider,
        model=raw_response.model,
        prompt_name=prompt_metadata.prompt_name if prompt_metadata is not None else None,
        prompt_version=prompt_metadata.prompt_version if prompt_metadata is not None else None,
        token_usage=copy.deepcopy(raw_response.usage),
        cost_metadata=None,
    )


def _safe_validation_details(value: dict[str, Any], exc: Exception) -> dict[str, Any]:
    post_text = value.get("post_text") if isinstance(value, dict) else None
    details: dict[str, Any] = {
        "error_code": ERROR_INVALID_CANDIDATE_POST,
        "validation_error": str(exc),
        "post_text": {
            "expected_type": "string",
            "max_chars": FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
            "input_type": type(post_text).__name__,
        },
    }
    if isinstance(post_text, str):
        details["post_text"]["input_length"] = len(post_text)
    return details


def _require_attributes(value: Any, attribute_names: tuple[str, ...], label: str) -> None:
    missing = [name for name in attribute_names if not hasattr(value, name)]
    if missing:
        raise CandidateWriterOutputAdaptationError(
            ERROR_INVALID_RAW_RESPONSE,
            f"{label} is missing required attributes: {missing}.",
            diagnostics=build_adapter_structural_diagnostics(
                adapter_error_code=ADAPTER_ERROR_UNKNOWN,
                parsed_candidate={},
                required_fields=(),
                allowed_fields=(),
            ),
        )


def _adapter_error_code_for_validation_failure(value: dict[str, Any]) -> str:
    failures = _field_failure_classifications(value)
    if any(item == "type" for item in failures.values()):
        return ADAPTER_ERROR_INVALID_FIELD_TYPES
    if failures:
        return ADAPTER_ERROR_INVALID_FIELD_VALUES
    return ADAPTER_ERROR_PAYLOAD_CONTRACT_VIOLATION


def _invalid_field_names_for_validation_failure(value: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_field_failure_classifications(value))


def _field_violations_for_validation_failure(value: dict[str, Any]) -> tuple[CandidateWriterFieldViolation, ...]:
    post_text = value.get("post_text") if isinstance(value, dict) else None
    if not isinstance(post_text, str):
        return (_field_violation("post_text", FIELD_VIOLATION_WRONG_TYPE, actual_type=type(post_text).__name__),)
    if not post_text.strip():
        return (_field_violation("post_text", FIELD_VIOLATION_EMPTY_STRING, actual_type="str", actual_length=len(post_text)),)
    if len(post_text) > FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS:
        return (
            _field_violation(
                "post_text",
                FIELD_VIOLATION_ABOVE_MAX_LENGTH,
                actual_type="str",
                actual_length=len(post_text),
                maximum_allowed=FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
            ),
        )
    return ()


def _field_violation(
    field_name: str,
    reason_code: str,
    *,
    actual_type: str | None = None,
    actual_length: int | None = None,
    minimum_required: int | None = None,
    maximum_allowed: int | None = None,
) -> CandidateWriterFieldViolation:
    return CandidateWriterFieldViolation(
        field_name=field_name,
        reason_code=reason_code,
        actual_type=actual_type,
        actual_length=actual_length,
        minimum_required=minimum_required,
        maximum_allowed=maximum_allowed,
    )


def _field_failure_classifications(value: dict[str, Any]) -> dict[str, str]:
    failures: dict[str, str] = {}
    post_text = value.get("post_text") if isinstance(value, dict) else None
    if not isinstance(post_text, str):
        failures["post_text"] = "type"
    elif not post_text.strip() or len(post_text) > FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS:
        failures["post_text"] = "value"
    return failures
