"""Adapter from parsed Candidate Writer output to PostFlow handoff contracts.

This module turns a parsed Candidate Writer dictionary into a canonical
FinalPostPayload-shaped dictionary and, when paired with a raw response, a
CandidateWriterOutput handoff. It does not parse JSON, call providers, run the
deterministic gate, evaluate quality, repair text, persist data, or connect to
runtime packaging.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput
from services.packaging.linkedin_post_pipeline import (
    FinalPostPayload,
    LinkedInPostPipelineContractError,
    REQUIRED_QUALITY_CHECKS,
    build_final_post_payload_constraints,
    final_post_payload_to_dict,
    validate_final_post_payload,
)

if TYPE_CHECKING:
    from services.packaging.linkedin_post_candidate_writer_execution import (
        CandidateWriterRawResponse,
    )

__all__ = (
    "CandidateWriterOutputAdaptationError",
    "adapt_candidate_writer_payload",
    "build_candidate_writer_output_from_parsed_response",
)


CANONICAL_FINAL_POST_PAYLOAD_FIELDS = (
    "post_text",
    "hook_variants",
    "cta_variants",
    "hashtags",
    "quality_checks",
    "carousel_outline",
)
REQUIRED_FINAL_POST_PAYLOAD_FIELDS = (
    "post_text",
    "hook_variants",
    "cta_variants",
    "hashtags",
    "quality_checks",
)
OPTIONAL_FINAL_POST_PAYLOAD_DEFAULTS = {
    "carousel_outline": [],
}

ERROR_INVALID_PARSED_CANDIDATE = "invalid_parsed_candidate"
ERROR_MISSING_REQUIRED_FIELD = "missing_required_field"
ERROR_INVALID_FINAL_POST_PAYLOAD = "invalid_final_post_payload"
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
        validation_detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.validation_detail = copy.deepcopy(validation_detail)


def adapt_candidate_writer_payload(parsed_candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical FinalPostPayload-compatible dictionary."""

    if not isinstance(parsed_candidate, dict):
        raise CandidateWriterOutputAdaptationError(
            ERROR_INVALID_PARSED_CANDIDATE,
            "parsed candidate must be a dictionary.",
        )

    missing_fields = [
        field
        for field in REQUIRED_FINAL_POST_PAYLOAD_FIELDS
        if field not in parsed_candidate
    ]
    if missing_fields:
        raise CandidateWriterOutputAdaptationError(
            ERROR_MISSING_REQUIRED_FIELD,
            f"parsed candidate is missing required fields: {missing_fields}.",
        )

    canonical_payload = {
        field: _copy_canonical_field(parsed_candidate[field], field)
        for field in REQUIRED_FINAL_POST_PAYLOAD_FIELDS
    }
    for field, default in OPTIONAL_FINAL_POST_PAYLOAD_DEFAULTS.items():
        canonical_payload[field] = copy.deepcopy(parsed_candidate.get(field, default))

    try:
        final_post_payload = FinalPostPayload(**canonical_payload)
        validate_final_post_payload(final_post_payload)
    except (TypeError, LinkedInPostPipelineContractError) as exc:
        raise CandidateWriterOutputAdaptationError(
            ERROR_INVALID_FINAL_POST_PAYLOAD,
            "parsed candidate does not satisfy FinalPostPayload structure.",
            validation_detail=_final_post_payload_validation_detail(
                exc,
                canonical_payload,
            ),
        ) from exc

    return copy.deepcopy(final_post_payload_to_dict(final_post_payload))


def build_candidate_writer_output_from_parsed_response(
    *,
    parsed_candidate: dict[str, Any],
    raw_response: "CandidateWriterRawResponse",
) -> CandidateWriterOutput:
    """Build a CandidateWriterOutput from parsed output and raw metadata."""

    _require_attributes(
        raw_response,
        REQUIRED_RAW_RESPONSE_ATTRIBUTES,
        "raw response",
    )

    if raw_response.execution_error:
        raise CandidateWriterOutputAdaptationError(
            ERROR_RAW_RESPONSE_EXECUTION_ERROR,
            "raw response has an execution error and cannot build candidate output.",
        )

    payload = adapt_candidate_writer_payload(parsed_candidate)
    prompt_metadata = raw_response.prompt_metadata
    if prompt_metadata is not None:
        _require_attributes(
            prompt_metadata,
            REQUIRED_PROMPT_METADATA_ATTRIBUTES,
            "prompt metadata",
        )

    return CandidateWriterOutput(
        payload=payload,
        raw_output=raw_response.raw_text,
        provider=raw_response.provider,
        model=raw_response.model,
        prompt_name=prompt_metadata.prompt_name if prompt_metadata is not None else None,
        prompt_version=(
            prompt_metadata.prompt_version if prompt_metadata is not None else None
        ),
        token_usage=copy.deepcopy(raw_response.usage),
        cost_metadata=None,
    )


def _copy_canonical_field(value: Any, field_name: str) -> Any:
    if field_name == "quality_checks" and isinstance(value, dict):
        return {
            key: copy.deepcopy(value[key])
            for key in REQUIRED_QUALITY_CHECKS
            if key in value
        }
    return copy.deepcopy(value)


def _final_post_payload_validation_detail(
    exc: BaseException,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    constraints = build_final_post_payload_constraints()
    post_text_constraints = constraints["post_text"]
    max_chars = post_text_constraints["max_chars"]
    message = str(exc)
    expected_post_text_length_message = (
        f"FinalPostPayload.post_text must not exceed {max_chars} characters."
    )
    post_text = payload.get("post_text")
    if message == expected_post_text_length_message and isinstance(post_text, str):
        actual_chars = len(post_text)
        return {
            "field_path": "FinalPostPayload.post_text",
            "field_name": "post_text",
            "message": message,
            "max_chars": max_chars,
            "actual_chars": actual_chars,
            "excess_chars": max(0, actual_chars - max_chars),
        }
    return None


def _require_attributes(value: Any, attribute_names: tuple[str, ...], label: str) -> None:
    missing = [
        attribute_name
        for attribute_name in attribute_names
        if not hasattr(value, attribute_name)
    ]
    if missing:
        raise CandidateWriterOutputAdaptationError(
            ERROR_INVALID_RAW_RESPONSE,
            f"{label} is missing required attributes: {missing}.",
        )
