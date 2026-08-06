"""Bounded diagnostics for Candidate Writer structural failures.

The diagnostics in this module are structural only. They intentionally retain
field names, type names, and lengths, but never raw response text, prompt text,
provider payloads, or Candidate Writer field values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DIAGNOSTIC_SCHEMA_VERSION = "1.0"
METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS = (
    "candidate_writer_structural_diagnostics"
)

FAILURE_STAGE_CANDIDATE_WRITER_PARSE = "candidate_writer_parse"
FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION = "candidate_writer_adaptation"
ALLOWED_FAILURE_STAGES = (
    FAILURE_STAGE_CANDIDATE_WRITER_PARSE,
    FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
)

PARSER_ERROR_EXECUTION_FAILED = "execution_failed"
PARSER_ERROR_EMPTY_RAW_RESPONSE = "empty_raw_response"
PARSER_ERROR_MALFORMED_FENCE = "malformed_fence"
PARSER_ERROR_MALFORMED_JSON = "malformed_json"
PARSER_ERROR_NON_OBJECT_JSON = "non_object_json"
ALLOWED_PARSER_ERROR_CODES = (
    PARSER_ERROR_EXECUTION_FAILED,
    PARSER_ERROR_EMPTY_RAW_RESPONSE,
    PARSER_ERROR_MALFORMED_FENCE,
    PARSER_ERROR_MALFORMED_JSON,
    PARSER_ERROR_NON_OBJECT_JSON,
)

ADAPTER_ERROR_TOP_LEVEL_NOT_OBJECT = "top_level_not_object"
ADAPTER_ERROR_MISSING_REQUIRED_FIELDS = "missing_required_fields"
ADAPTER_ERROR_INVALID_FIELD_TYPES = "invalid_field_types"
ADAPTER_ERROR_INVALID_FIELD_VALUES = "invalid_field_values"
ADAPTER_ERROR_PAYLOAD_CONTRACT_VIOLATION = "payload_contract_violation"
ADAPTER_ERROR_UNKNOWN = "unknown"
ALLOWED_ADAPTER_ERROR_CODES = (
    ADAPTER_ERROR_TOP_LEVEL_NOT_OBJECT,
    ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
    ADAPTER_ERROR_INVALID_FIELD_TYPES,
    ADAPTER_ERROR_INVALID_FIELD_VALUES,
    ADAPTER_ERROR_PAYLOAD_CONTRACT_VIOLATION,
    ADAPTER_ERROR_UNKNOWN,
)

MAX_DIAGNOSTIC_LIST_ITEMS = 20
MAX_DIAGNOSTIC_FIELD_NAME_LENGTH = 80
SAFE_TYPE_NAME_MAX_LENGTH = 80
SAFE_TOP_LEVEL_TYPE_NAMES = (
    "NoneType",
    "bool",
    "dict",
    "float",
    "int",
    "list",
    "str",
    "tuple",
)
SUSPICIOUS_FIELD_NAME_FRAGMENTS = (
    "api_key",
    "authorization",
    "credential",
    "header",
    "password",
    "prompt",
    "provider_payload",
    "raw_response",
    "secret",
    "token",
)


@dataclass(frozen=True)
class CandidateWriterStructuralDiagnostics:
    failure_stage: str
    parser_error_code: str | None
    adapter_error_code: str | None
    top_level_json_type: str | None
    received_top_level_keys: tuple[str, ...]
    missing_required_fields: tuple[str, ...]
    unexpected_fields: tuple[str, ...]
    invalid_field_names: tuple[str, ...]
    candidate_text_length: int | None
    diagnostics_truncated: bool
    redacted_key_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "top_level_json_type",
            _safe_type_name(self.top_level_json_type),
        )
        if self.failure_stage not in ALLOWED_FAILURE_STAGES:
            raise ValueError("failure_stage is not supported")
        if (
            self.parser_error_code is not None
            and self.parser_error_code not in ALLOWED_PARSER_ERROR_CODES
        ):
            raise ValueError("parser_error_code is not supported")
        if (
            self.adapter_error_code is not None
            and self.adapter_error_code not in ALLOWED_ADAPTER_ERROR_CODES
        ):
            raise ValueError("adapter_error_code is not supported")
        if self.failure_stage == FAILURE_STAGE_CANDIDATE_WRITER_PARSE:
            if self.parser_error_code is None or self.adapter_error_code is not None:
                raise ValueError("parse diagnostics must only carry parser_error_code")
        if self.failure_stage == FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION:
            if self.adapter_error_code is None or self.parser_error_code is not None:
                raise ValueError("adaptation diagnostics must only carry adapter_error_code")
        if self.candidate_text_length is not None:
            if (
                isinstance(self.candidate_text_length, bool)
                or not isinstance(self.candidate_text_length, int)
                or self.candidate_text_length < 0
            ):
                raise ValueError("candidate_text_length must be a non-negative integer")
        if isinstance(self.redacted_key_count, bool) or self.redacted_key_count < 0:
            raise ValueError("redacted_key_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "failure_stage": self.failure_stage,
            "parser_error_code": self.parser_error_code,
            "adapter_error_code": self.adapter_error_code,
            "top_level_json_type": self.top_level_json_type,
            "received_top_level_keys": list(self.received_top_level_keys),
            "missing_required_fields": list(self.missing_required_fields),
            "unexpected_fields": list(self.unexpected_fields),
            "invalid_field_names": list(self.invalid_field_names),
            "candidate_text_length": self.candidate_text_length,
            "diagnostics_truncated": self.diagnostics_truncated,
            "redacted_key_count": self.redacted_key_count,
        }


def build_parser_structural_diagnostics(
    *,
    parser_error_code: str,
    candidate_text: Any,
    top_level_json_type: str | None = None,
) -> CandidateWriterStructuralDiagnostics:
    return CandidateWriterStructuralDiagnostics(
        failure_stage=FAILURE_STAGE_CANDIDATE_WRITER_PARSE,
        parser_error_code=parser_error_code,
        adapter_error_code=None,
        top_level_json_type=_safe_type_name(top_level_json_type),
        received_top_level_keys=(),
        missing_required_fields=(),
        unexpected_fields=(),
        invalid_field_names=(),
        candidate_text_length=_candidate_text_length(candidate_text),
        diagnostics_truncated=False,
        redacted_key_count=0,
    )


def build_adapter_structural_diagnostics(
    *,
    adapter_error_code: str,
    parsed_candidate: Any,
    required_fields: Iterable[str],
    allowed_fields: Iterable[str],
    invalid_field_names: Iterable[str] = (),
) -> CandidateWriterStructuralDiagnostics:
    received_keys: Iterable[Any] = ()
    missing_fields: Iterable[str] = ()
    unexpected_fields: Iterable[Any] = ()
    top_level_type = _type_name(parsed_candidate)
    if isinstance(parsed_candidate, dict):
        received_keys = parsed_candidate.keys()
        required = tuple(str(field) for field in required_fields)
        allowed = set(str(field) for field in allowed_fields)
        missing_fields = tuple(field for field in required if field not in parsed_candidate)
        unexpected_fields = tuple(key for key in parsed_candidate if str(key) not in allowed)

    received = sanitize_field_names(received_keys)
    missing = sanitize_field_names(missing_fields)
    unexpected = sanitize_field_names(unexpected_fields)
    invalid = sanitize_field_names(invalid_field_names)
    diagnostics_truncated = any(
        result.diagnostics_truncated
        for result in (received, missing, unexpected, invalid)
    )
    redacted_key_count = sum(
        result.redacted_key_count
        for result in (received, missing, unexpected, invalid)
    )
    return CandidateWriterStructuralDiagnostics(
        failure_stage=FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
        parser_error_code=None,
        adapter_error_code=adapter_error_code,
        top_level_json_type=top_level_type,
        received_top_level_keys=received.values,
        missing_required_fields=missing.values,
        unexpected_fields=unexpected.values,
        invalid_field_names=invalid.values,
        candidate_text_length=None,
        diagnostics_truncated=diagnostics_truncated,
        redacted_key_count=redacted_key_count,
    )


@dataclass(frozen=True)
class SanitizedFieldNames:
    values: tuple[str, ...]
    diagnostics_truncated: bool
    redacted_key_count: int


def sanitize_field_names(values: Iterable[Any]) -> SanitizedFieldNames:
    safe_values: set[str] = set()
    redacted_count = 0
    for value in values:
        safe_value = _safe_field_name(value)
        if safe_value is None:
            redacted_count += 1
            continue
        safe_values.add(safe_value)
    sorted_values = tuple(sorted(safe_values))
    truncated_values = sorted_values[:MAX_DIAGNOSTIC_LIST_ITEMS]
    omitted_count = max(0, len(sorted_values) - len(truncated_values))
    return SanitizedFieldNames(
        values=truncated_values,
        diagnostics_truncated=bool(redacted_count or omitted_count),
        redacted_key_count=redacted_count + omitted_count,
    )


def structural_diagnostics_from_dict(
    value: Any,
) -> CandidateWriterStructuralDiagnostics | None:
    if not isinstance(value, dict):
        return None
    received = sanitize_field_names(_field_name_values(value.get("received_top_level_keys")))
    missing = sanitize_field_names(_field_name_values(value.get("missing_required_fields")))
    unexpected = sanitize_field_names(_field_name_values(value.get("unexpected_fields")))
    invalid = sanitize_field_names(_field_name_values(value.get("invalid_field_names")))
    diagnostics_truncated = bool(value.get("diagnostics_truncated")) or any(
        result.diagnostics_truncated
        for result in (received, missing, unexpected, invalid)
    )
    redacted_key_count = (
        _safe_non_negative_int(value.get("redacted_key_count"))
        + sum(
            result.redacted_key_count
            for result in (received, missing, unexpected, invalid)
        )
    )
    try:
        return CandidateWriterStructuralDiagnostics(
            failure_stage=value.get("failure_stage"),
            parser_error_code=value.get("parser_error_code"),
            adapter_error_code=value.get("adapter_error_code"),
            top_level_json_type=_safe_type_name(value.get("top_level_json_type")),
            received_top_level_keys=received.values,
            missing_required_fields=missing.values,
            unexpected_fields=unexpected.values,
            invalid_field_names=invalid.values,
            candidate_text_length=value.get("candidate_text_length"),
            diagnostics_truncated=diagnostics_truncated,
            redacted_key_count=redacted_key_count,
        )
    except (TypeError, ValueError):
        return None


def _safe_field_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > MAX_DIAGNOSTIC_FIELD_NAME_LENGTH:
        return None
    if any(ord(character) < 32 for character in stripped):
        return None
    lowered = stripped.lower()
    if any(fragment in lowered for fragment in SUSPICIOUS_FIELD_NAME_FRAGMENTS):
        return None
    return stripped


def _field_name_values(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple, set)):
        return value
    return ()


def _candidate_text_length(value: Any) -> int | None:
    if value is None:
        return None
    return len(str(value))


def _safe_non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, integer)


def _safe_type_name(value: str | None) -> str | None:
    if value is None or not isinstance(value, str):
        return None
    stripped = str(value).strip()[:SAFE_TYPE_NAME_MAX_LENGTH]
    if stripped not in SAFE_TOP_LEVEL_TYPE_NAMES:
        return None
    return stripped or None


def _type_name(value: Any) -> str | None:
    return _safe_type_name(type(value).__name__)
