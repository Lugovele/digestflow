"""Sanitized structural diagnostics for Repair Writer adaptation.

This module observes Repair Writer parser/adaptation inputs and failures. It
does not parse, normalize, validate, repair, execute providers, or change the
canonical CandidatePost contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_post_candidate_post_contract import (
    CANDIDATE_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_candidate_writer_output_adapter import (
    CandidateWriterOutputAdaptationError,
)
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    FIELD_VIOLATION_ABOVE_MAX_LENGTH,
    FIELD_VIOLATION_EMPTY_STRING,
    FIELD_VIOLATION_WRONG_TYPE,
    sanitize_field_names,
)


POST_TEXT_MISSING = "POST_TEXT_MISSING"
POST_TEXT_WRONG_TYPE = "POST_TEXT_WRONG_TYPE"
POST_TEXT_EMPTY = "POST_TEXT_EMPTY"
POST_TEXT_BLANK = "POST_TEXT_BLANK"
POST_TEXT_TOO_LONG = "POST_TEXT_TOO_LONG"
CANDIDATE_POST_OTHER_VALIDATION_FAILURE = "CANDIDATE_POST_OTHER_VALIDATION_FAILURE"
UNKNOWN = "UNKNOWN"

STRUCTURAL_FAILURE_CATEGORIES = (
    POST_TEXT_MISSING,
    POST_TEXT_WRONG_TYPE,
    POST_TEXT_EMPTY,
    POST_TEXT_BLANK,
    POST_TEXT_TOO_LONG,
    CANDIDATE_POST_OTHER_VALIDATION_FAILURE,
    UNKNOWN,
)


@dataclass(frozen=True)
class RepairWriterStructuralDiagnostics:
    parsed_top_level_type: str
    parsed_top_level_keys: tuple[str, ...]
    post_text_present: bool
    post_text_type: str | None
    post_text_is_string: bool
    post_text_character_count: int | None
    post_text_is_empty: bool | None
    post_text_is_blank: bool | None
    post_text_within_candidate_max_length: bool | None
    candidate_post_max_length: int
    candidate_validation_error_code: str | None
    candidate_validation_error_field: str | None
    candidate_validation_error_kind: str | None
    structural_failure_category: str
    raw_response_character_count: int | None = None
    starts_with_json_object: bool | None = None
    ends_with_json_object: bool | None = None
    starts_with_code_fence: bool | None = None
    ends_with_code_fence: bool | None = None

    def __post_init__(self) -> None:
        if self.structural_failure_category not in STRUCTURAL_FAILURE_CATEGORIES:
            raise ValueError("unsupported structural failure category")

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed_top_level_type": self.parsed_top_level_type,
            "parsed_top_level_keys": list(self.parsed_top_level_keys),
            "post_text_present": self.post_text_present,
            "post_text_type": self.post_text_type,
            "post_text_is_string": self.post_text_is_string,
            "post_text_character_count": self.post_text_character_count,
            "post_text_is_empty": self.post_text_is_empty,
            "post_text_is_blank": self.post_text_is_blank,
            "post_text_within_candidate_max_length": (
                self.post_text_within_candidate_max_length
            ),
            "candidate_post_max_length": self.candidate_post_max_length,
            "candidate_validation_error_code": self.candidate_validation_error_code,
            "candidate_validation_error_field": self.candidate_validation_error_field,
            "candidate_validation_error_kind": self.candidate_validation_error_kind,
            "structural_failure_category": self.structural_failure_category,
            "raw_response_character_count": self.raw_response_character_count,
            "starts_with_json_object": self.starts_with_json_object,
            "ends_with_json_object": self.ends_with_json_object,
            "starts_with_code_fence": self.starts_with_code_fence,
            "ends_with_code_fence": self.ends_with_code_fence,
        }


def build_repair_writer_structural_diagnostics(
    *,
    parsed_repair_candidate: Any,
    adaptation_error: CandidateWriterOutputAdaptationError | None = None,
    repair_raw_response: Any | None = None,
) -> RepairWriterStructuralDiagnostics:
    is_mapping = isinstance(parsed_repair_candidate, dict)
    post_text_present = is_mapping and "post_text" in parsed_repair_candidate
    post_text = parsed_repair_candidate.get("post_text") if is_mapping else None
    post_text_is_string = isinstance(post_text, str)
    response_shape = _raw_response_shape(repair_raw_response)
    return RepairWriterStructuralDiagnostics(
        parsed_top_level_type=type(parsed_repair_candidate).__name__,
        parsed_top_level_keys=_safe_top_level_keys(parsed_repair_candidate),
        post_text_present=post_text_present,
        post_text_type=type(post_text).__name__ if post_text_present else None,
        post_text_is_string=post_text_is_string,
        post_text_character_count=len(post_text) if post_text_is_string else None,
        post_text_is_empty=(post_text == "") if post_text_is_string else None,
        post_text_is_blank=(
            bool(post_text) and not post_text.strip()
            if post_text_is_string
            else None
        ),
        post_text_within_candidate_max_length=(
            len(post_text) <= CANDIDATE_POST_TEXT_MAX_CHARS
            if post_text_is_string
            else None
        ),
        candidate_post_max_length=CANDIDATE_POST_TEXT_MAX_CHARS,
        candidate_validation_error_code=(
            adaptation_error.code if adaptation_error is not None else None
        ),
        candidate_validation_error_field=_validation_error_field(adaptation_error),
        candidate_validation_error_kind=_validation_error_kind(adaptation_error),
        structural_failure_category=_structural_failure_category(
            parsed_repair_candidate,
            adaptation_error,
        ),
        **response_shape,
    )


def _safe_top_level_keys(parsed_repair_candidate: Any) -> tuple[str, ...]:
    if not isinstance(parsed_repair_candidate, dict):
        return ()
    return sanitize_field_names(parsed_repair_candidate.keys()).values


def _raw_response_shape(repair_raw_response: Any | None) -> dict[str, Any]:
    if repair_raw_response is None or not hasattr(repair_raw_response, "raw_text"):
        return {
            "raw_response_character_count": None,
            "starts_with_json_object": None,
            "ends_with_json_object": None,
            "starts_with_code_fence": None,
            "ends_with_code_fence": None,
        }
    text = str(repair_raw_response.raw_text or "")
    stripped = text.strip()
    return {
        "raw_response_character_count": len(text),
        "starts_with_json_object": stripped.startswith("{"),
        "ends_with_json_object": stripped.endswith("}"),
        "starts_with_code_fence": stripped.startswith("```"),
        "ends_with_code_fence": stripped.endswith("```"),
    }


def _validation_error_field(
    adaptation_error: CandidateWriterOutputAdaptationError | None,
) -> str | None:
    diagnostics = getattr(adaptation_error, "diagnostics", None)
    violations = getattr(diagnostics, "field_violations", ()) if diagnostics else ()
    if violations:
        return violations[0].field_name
    invalid_fields = getattr(diagnostics, "invalid_field_names", ()) if diagnostics else ()
    if invalid_fields:
        return invalid_fields[0]
    missing_fields = getattr(diagnostics, "missing_required_fields", ()) if diagnostics else ()
    if missing_fields:
        return missing_fields[0]
    return None


def _validation_error_kind(
    adaptation_error: CandidateWriterOutputAdaptationError | None,
) -> str | None:
    diagnostics = getattr(adaptation_error, "diagnostics", None)
    return getattr(diagnostics, "adapter_error_code", None) if diagnostics else None


def _structural_failure_category(
    parsed_repair_candidate: Any,
    adaptation_error: CandidateWriterOutputAdaptationError | None,
) -> str:
    if not isinstance(parsed_repair_candidate, dict):
        return CANDIDATE_POST_OTHER_VALIDATION_FAILURE
    if "post_text" not in parsed_repair_candidate:
        return POST_TEXT_MISSING
    post_text = parsed_repair_candidate["post_text"]
    if not isinstance(post_text, str):
        return POST_TEXT_WRONG_TYPE
    if post_text == "":
        return POST_TEXT_EMPTY
    if not post_text.strip():
        return POST_TEXT_BLANK
    if len(post_text) > CANDIDATE_POST_TEXT_MAX_CHARS:
        return POST_TEXT_TOO_LONG
    diagnostics = getattr(adaptation_error, "diagnostics", None)
    violations = getattr(diagnostics, "field_violations", ()) if diagnostics else ()
    if violations:
        reason_code = violations[0].reason_code
        if reason_code == FIELD_VIOLATION_WRONG_TYPE:
            return POST_TEXT_WRONG_TYPE
        if reason_code == FIELD_VIOLATION_EMPTY_STRING:
            return POST_TEXT_EMPTY
        if reason_code == FIELD_VIOLATION_ABOVE_MAX_LENGTH:
            return POST_TEXT_TOO_LONG
    if adaptation_error is not None:
        return CANDIDATE_POST_OTHER_VALIDATION_FAILURE
    return UNKNOWN
