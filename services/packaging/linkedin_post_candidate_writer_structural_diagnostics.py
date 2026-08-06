"""Bounded diagnostics for Candidate Writer structural failures.

The diagnostics in this module are structural only. They intentionally retain
field names, type names, and lengths, but never raw response text, prompt text,
provider payloads, or Candidate Writer field values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DIAGNOSTIC_SCHEMA_VERSION = "1.1"
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

FIELD_VIOLATION_WRONG_TYPE = "wrong_type"
FIELD_VIOLATION_EMPTY_STRING = "empty_string"
FIELD_VIOLATION_ABOVE_MAX_LENGTH = "above_max_length"
FIELD_VIOLATION_BELOW_MIN_COUNT = "below_min_count"
FIELD_VIOLATION_INVALID_STRING_LIST_ITEM = "invalid_string_list_item"
FIELD_VIOLATION_MISSING_REQUIRED_BOOLEAN_KEY = "missing_required_boolean_key"
FIELD_VIOLATION_NON_BOOLEAN_REQUIRED_KEY = "non_boolean_required_key"
ALLOWED_FIELD_VIOLATION_REASON_CODES = (
    FIELD_VIOLATION_WRONG_TYPE,
    FIELD_VIOLATION_EMPTY_STRING,
    FIELD_VIOLATION_ABOVE_MAX_LENGTH,
    FIELD_VIOLATION_BELOW_MIN_COUNT,
    FIELD_VIOLATION_INVALID_STRING_LIST_ITEM,
    FIELD_VIOLATION_MISSING_REQUIRED_BOOLEAN_KEY,
    FIELD_VIOLATION_NON_BOOLEAN_REQUIRED_KEY,
)

PARSER_DETAIL_UNEXPECTED_END_OF_INPUT = "unexpected_end_of_input"
PARSER_DETAIL_EXTRA_DATA_AFTER_JSON = "extra_data_after_json"
PARSER_DETAIL_INVALID_CONTROL_CHARACTER = "invalid_control_character"
PARSER_DETAIL_UNTERMINATED_STRING = "unterminated_string"
PARSER_DETAIL_EXPECTED_PROPERTY_NAME = "expected_property_name"
PARSER_DETAIL_TRAILING_COMMA = "trailing_comma"
PARSER_DETAIL_EXPECTED_VALUE = "expected_value"
PARSER_DETAIL_EXPECTED_COLON = "expected_colon"
PARSER_DETAIL_EXPECTED_COMMA_DELIMITER = "expected_comma_delimiter"
PARSER_DETAIL_INVALID_ESCAPE = "invalid_escape"
PARSER_DETAIL_NON_STANDARD_NUMBER = "non_standard_number"
PARSER_DETAIL_UNKNOWN_JSON_SYNTAX = "unknown_json_syntax"
ALLOWED_PARSER_ERROR_DETAIL_CODES = (
    PARSER_DETAIL_UNEXPECTED_END_OF_INPUT,
    PARSER_DETAIL_EXTRA_DATA_AFTER_JSON,
    PARSER_DETAIL_INVALID_CONTROL_CHARACTER,
    PARSER_DETAIL_UNTERMINATED_STRING,
    PARSER_DETAIL_EXPECTED_PROPERTY_NAME,
    PARSER_DETAIL_TRAILING_COMMA,
    PARSER_DETAIL_EXPECTED_VALUE,
    PARSER_DETAIL_EXPECTED_COLON,
    PARSER_DETAIL_EXPECTED_COMMA_DELIMITER,
    PARSER_DETAIL_INVALID_ESCAPE,
    PARSER_DETAIL_NON_STANDARD_NUMBER,
    PARSER_DETAIL_UNKNOWN_JSON_SYNTAX,
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
class CandidateWriterFieldViolation:
    field_name: str
    reason_code: str
    actual_type: str | None
    actual_length: int | None
    minimum_required: int | None
    maximum_allowed: int | None

    def __post_init__(self) -> None:
        safe_field_name = _safe_field_name(self.field_name)
        if safe_field_name is None:
            raise ValueError("field_name is not safe")
        if self.reason_code not in ALLOWED_FIELD_VIOLATION_REASON_CODES:
            raise ValueError("reason_code is not supported")
        object.__setattr__(self, "field_name", safe_field_name)
        object.__setattr__(self, "actual_type", _safe_type_name(self.actual_type))
        for field_name in (
            "actual_length",
            "minimum_required",
            "maximum_allowed",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "reason_code": self.reason_code,
            "actual_type": self.actual_type,
            "actual_length": self.actual_length,
            "minimum_required": self.minimum_required,
            "maximum_allowed": self.maximum_allowed,
        }


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
    field_violations: tuple[CandidateWriterFieldViolation, ...] = ()
    parser_error_detail_code: str | None = None
    parser_error_line: int | None = None
    parser_error_column: int | None = None
    parser_error_position: int | None = None

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
        if (
            self.parser_error_detail_code is not None
            and self.parser_error_detail_code not in ALLOWED_PARSER_ERROR_DETAIL_CODES
        ):
            raise ValueError("parser_error_detail_code is not supported")
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
        for field_name in (
            "parser_error_line",
            "parser_error_column",
            "parser_error_position",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        object.__setattr__(
            self,
            "field_violations",
            _safe_field_violations(self.field_violations),
        )

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
            "field_violations": [
                violation.to_dict() for violation in self.field_violations
            ],
            "parser_error_detail_code": self.parser_error_detail_code,
            "parser_error_line": self.parser_error_line,
            "parser_error_column": self.parser_error_column,
            "parser_error_position": self.parser_error_position,
        }


def build_parser_structural_diagnostics(
    *,
    parser_error_code: str,
    candidate_text: Any,
    top_level_json_type: str | None = None,
    parser_error_detail_code: str | None = None,
    parser_error_line: int | None = None,
    parser_error_column: int | None = None,
    parser_error_position: int | None = None,
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
        field_violations=(),
        parser_error_detail_code=parser_error_detail_code,
        parser_error_line=parser_error_line,
        parser_error_column=parser_error_column,
        parser_error_position=parser_error_position,
    )


def build_adapter_structural_diagnostics(
    *,
    adapter_error_code: str,
    parsed_candidate: Any,
    required_fields: Iterable[str],
    allowed_fields: Iterable[str],
    invalid_field_names: Iterable[str] = (),
    field_violations: Iterable[CandidateWriterFieldViolation] = (),
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
    raw_violations = tuple(field_violations)
    violations = _safe_field_violations(raw_violations)
    diagnostics_truncated = any(
        result.diagnostics_truncated
        for result in (received, missing, unexpected, invalid)
    ) or len(raw_violations) > len(violations)
    redacted_key_count = sum(
        result.redacted_key_count
        for result in (received, missing, unexpected, invalid)
    ) + max(0, len(raw_violations) - len(violations))
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
        field_violations=violations,
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
    violations, violation_truncated, violation_redacted_count = (
        _safe_field_violations_from_dicts(value.get("field_violations"))
    )
    diagnostics_truncated = bool(value.get("diagnostics_truncated")) or any(
        result.diagnostics_truncated
        for result in (received, missing, unexpected, invalid)
    ) or violation_truncated
    redacted_key_count = (
        _safe_non_negative_int(value.get("redacted_key_count"))
        + sum(
            result.redacted_key_count
            for result in (received, missing, unexpected, invalid)
        )
        + violation_redacted_count
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
            field_violations=violations,
            parser_error_detail_code=value.get("parser_error_detail_code"),
            parser_error_line=_safe_optional_non_negative_int(
                value.get("parser_error_line")
            ),
            parser_error_column=_safe_optional_non_negative_int(
                value.get("parser_error_column")
            ),
            parser_error_position=_safe_optional_non_negative_int(
                value.get("parser_error_position")
            ),
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


def _safe_optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    if integer < 0:
        return None
    return integer


def _safe_field_violations(
    violations: Iterable[CandidateWriterFieldViolation],
) -> tuple[CandidateWriterFieldViolation, ...]:
    safe: list[CandidateWriterFieldViolation] = []
    for violation in violations:
        if not isinstance(violation, CandidateWriterFieldViolation):
            continue
        safe.append(violation)
    return tuple(
        sorted(
            safe,
            key=lambda violation: (violation.field_name, violation.reason_code),
        )[:MAX_DIAGNOSTIC_LIST_ITEMS]
    )


def _safe_field_violations_from_dicts(
    value: Any,
) -> tuple[tuple[CandidateWriterFieldViolation, ...], bool, int]:
    if not isinstance(value, (list, tuple)):
        return (), False, 0
    violations: list[CandidateWriterFieldViolation] = []
    redacted_count = 0
    for item in value:
        if not isinstance(item, dict):
            redacted_count += 1
            continue
        try:
            violations.append(
                CandidateWriterFieldViolation(
                    field_name=item.get("field_name"),
                    reason_code=item.get("reason_code"),
                    actual_type=item.get("actual_type"),
                    actual_length=_safe_optional_non_negative_int(
                        item.get("actual_length")
                    ),
                    minimum_required=_safe_optional_non_negative_int(
                        item.get("minimum_required")
                    ),
                    maximum_allowed=_safe_optional_non_negative_int(
                        item.get("maximum_allowed")
                    ),
                )
            )
        except (TypeError, ValueError):
            redacted_count += 1
    bounded = _safe_field_violations(violations)
    omitted_count = max(0, len(violations) - len(bounded))
    return bounded, bool(redacted_count or omitted_count), redacted_count + omitted_count


def _safe_type_name(value: str | None) -> str | None:
    if value is None or not isinstance(value, str):
        return None
    stripped = str(value).strip()[:SAFE_TYPE_NAME_MAX_LENGTH]
    if stripped not in SAFE_TOP_LEVEL_TYPE_NAMES:
        return None
    return stripped or None


def _type_name(value: Any) -> str | None:
    return _safe_type_name(type(value).__name__)
