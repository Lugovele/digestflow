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
COMPLETE_PLAIN_JSON = "COMPLETE_PLAIN_JSON"
COMPLETE_FENCED_JSON = "COMPLETE_FENCED_JSON"
TRUNCATED_INSIDE_JSON = "TRUNCATED_INSIDE_JSON"
TRUNCATED_AFTER_JSON_BEFORE_FENCE = "TRUNCATED_AFTER_JSON_BEFORE_FENCE"
MALFORMED_FENCE_NOT_TRUNCATED = "MALFORMED_FENCE_NOT_TRUNCATED"
NON_JSON_RESPONSE = "NON_JSON_RESPONSE"

STRUCTURAL_FAILURE_CATEGORIES = (
    POST_TEXT_MISSING,
    POST_TEXT_WRONG_TYPE,
    POST_TEXT_EMPTY,
    POST_TEXT_BLANK,
    POST_TEXT_TOO_LONG,
    CANDIDATE_POST_OTHER_VALIDATION_FAILURE,
    UNKNOWN,
)
RESPONSE_STRUCTURE_CLASSIFICATIONS = (
    COMPLETE_PLAIN_JSON,
    COMPLETE_FENCED_JSON,
    TRUNCATED_INSIDE_JSON,
    TRUNCATED_AFTER_JSON_BEFORE_FENCE,
    MALFORMED_FENCE_NOT_TRUNCATED,
    NON_JSON_RESPONSE,
    UNKNOWN,
)
PREFIX_EXCERPT_MAX_CHARS = 80
SUFFIX_EXCERPT_MAX_CHARS = 120


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
    raw_response_prefix_excerpt: str | None = None
    raw_response_suffix_excerpt: str | None = None
    opening_fence_language: str | None = None
    contains_json_object_start_after_fence: bool | None = None
    contains_json_object_end: bool | None = None
    json_brace_balance: int | None = None
    json_string_appears_unterminated: bool | None = None
    markdown_fence_count: int | None = None
    response_structure_classification: str | None = None

    def __post_init__(self) -> None:
        if self.structural_failure_category not in STRUCTURAL_FAILURE_CATEGORIES:
            raise ValueError("unsupported structural failure category")
        if (
            self.response_structure_classification is not None
            and self.response_structure_classification not in RESPONSE_STRUCTURE_CLASSIFICATIONS
        ):
            raise ValueError("unsupported response structure classification")

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
            "raw_response_prefix_excerpt": self.raw_response_prefix_excerpt,
            "raw_response_suffix_excerpt": self.raw_response_suffix_excerpt,
            "opening_fence_language": self.opening_fence_language,
            "contains_json_object_start_after_fence": (
                self.contains_json_object_start_after_fence
            ),
            "contains_json_object_end": self.contains_json_object_end,
            "json_brace_balance": self.json_brace_balance,
            "json_string_appears_unterminated": (
                self.json_string_appears_unterminated
            ),
            "markdown_fence_count": self.markdown_fence_count,
            "response_structure_classification": (
                self.response_structure_classification
            ),
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
            "raw_response_prefix_excerpt": None,
            "raw_response_suffix_excerpt": None,
            "opening_fence_language": None,
            "contains_json_object_start_after_fence": None,
            "contains_json_object_end": None,
            "json_brace_balance": None,
            "json_string_appears_unterminated": None,
            "markdown_fence_count": None,
            "response_structure_classification": None,
        }
    text = str(repair_raw_response.raw_text or "")
    return build_repair_writer_response_structure_diagnostics(
        text,
        provider_output_limit_reached=_provider_output_limit_reached(
            repair_raw_response
        ),
    )


def build_repair_writer_response_structure_diagnostics(
    raw_text: str,
    *,
    provider_output_limit_reached: bool | None = None,
) -> dict[str, Any]:
    text = str(raw_text or "")
    stripped = text.strip()
    json_text = _json_region_for_shape(stripped)
    scan = _scan_json_shape(json_text)
    return {
        "raw_response_character_count": len(text),
        "starts_with_json_object": stripped.startswith("{"),
        "ends_with_json_object": stripped.endswith("}"),
        "starts_with_code_fence": stripped.startswith("```"),
        "ends_with_code_fence": stripped.endswith("```"),
        "raw_response_prefix_excerpt": _safe_excerpt(text, PREFIX_EXCERPT_MAX_CHARS),
        "raw_response_suffix_excerpt": _safe_suffix_excerpt(
            text,
            SUFFIX_EXCERPT_MAX_CHARS,
        ),
        "opening_fence_language": _opening_fence_language(stripped),
        "contains_json_object_start_after_fence": (
            _body_after_opening_fence(stripped).lstrip().startswith("{")
            if stripped.startswith("```")
            else False
        ),
        "contains_json_object_end": scan["contains_json_object_end"],
        "json_brace_balance": scan["brace_balance"],
        "json_string_appears_unterminated": scan["unterminated_string"],
        "markdown_fence_count": stripped.count("```"),
        "response_structure_classification": _classify_response_structure(
            stripped,
            scan,
            provider_output_limit_reached=provider_output_limit_reached,
        ),
    }


def _provider_output_limit_reached(repair_raw_response: Any) -> bool | None:
    metadata = getattr(repair_raw_response, "provider_response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("provider_output_limit_reached")
    return value if value is None or isinstance(value, bool) else None


def _json_region_for_shape(stripped: str) -> str:
    if not stripped.startswith("```"):
        return stripped
    return _body_after_opening_fence(stripped).removesuffix("```").strip()


def _body_after_opening_fence(stripped: str) -> str:
    if "\n" not in stripped:
        return ""
    return stripped.split("\n", 1)[1]


def _opening_fence_language(stripped: str) -> str | None:
    if not stripped.startswith("```"):
        return None
    first_line = stripped.splitlines()[0].strip()
    if not first_line.startswith("```"):
        return None
    language = first_line[3:]
    if language in {"", "json"}:
        return language
    return "other"


def _scan_json_shape(text: str) -> dict[str, Any]:
    balance = 0
    in_string = False
    escaped = False
    contains_end = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
            contains_end = True
    return {
        "brace_balance": balance,
        "unterminated_string": in_string,
        "contains_json_object_end": contains_end,
    }


def _classify_response_structure(
    stripped: str,
    scan: dict[str, Any],
    *,
    provider_output_limit_reached: bool | None,
) -> str:
    if not stripped:
        return NON_JSON_RESPONSE
    if _is_complete_plain_json(stripped, scan):
        return COMPLETE_PLAIN_JSON
    if _is_complete_fenced_json(stripped, scan):
        return COMPLETE_FENCED_JSON
    if provider_output_limit_reached is True:
        if stripped.startswith("```") and scan["brace_balance"] == 0 and scan["contains_json_object_end"]:
            return TRUNCATED_AFTER_JSON_BEFORE_FENCE
        if stripped.startswith("{") or stripped.startswith("```"):
            return TRUNCATED_INSIDE_JSON
    if "```" in stripped:
        return MALFORMED_FENCE_NOT_TRUNCATED
    if not stripped.startswith("{"):
        return NON_JSON_RESPONSE
    return UNKNOWN


def _is_complete_plain_json(stripped: str, scan: dict[str, Any]) -> bool:
    return (
        stripped.startswith("{")
        and stripped.endswith("}")
        and scan["brace_balance"] == 0
        and not scan["unterminated_string"]
    )


def _is_complete_fenced_json(stripped: str, scan: dict[str, Any]) -> bool:
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return False
    if stripped.count("```") != 2:
        return False
    opening = stripped.splitlines()[0].strip() if stripped.splitlines() else ""
    if opening not in {"```", "```json"}:
        return False
    body = _body_after_opening_fence(stripped).removesuffix("```").strip()
    return (
        body.startswith("{")
        and body.endswith("}")
        and scan["brace_balance"] == 0
        and not scan["unterminated_string"]
    )


def _safe_excerpt(text: str, max_chars: int) -> str:
    excerpt = _mask_json_string_content(str(text or ""))
    excerpt = excerpt.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    return excerpt[:max_chars]


def _safe_suffix_excerpt(text: str, max_chars: int) -> str:
    excerpt = _mask_json_string_content(str(text or ""))
    excerpt = excerpt.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    return excerpt[-max_chars:]


def _mask_json_string_content(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    emitted_marker = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                result.append('"')
                in_string = False
                emitted_marker = False
            elif not emitted_marker:
                result.append("...")
                emitted_marker = True
            continue
        if char.isalnum() or char in {"_", "-"}:
            if not emitted_marker:
                result.append("...")
                emitted_marker = True
            continue
        emitted_marker = False
        result.append(char)
        if char == '"':
            in_string = True
            emitted_marker = False
    return "".join(result)


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
