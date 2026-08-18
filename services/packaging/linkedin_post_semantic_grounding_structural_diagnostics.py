"""Sanitized structural diagnostics for Semantic Grounding raw responses.

This module observes provider response shape only. It does not parse, normalize,
repair, execute providers, classify claims, or change Semantic Grounding parser
acceptance rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


COMPLETE_PLAIN_JSON = "COMPLETE_PLAIN_JSON"
COMPLETE_FENCED_JSON = "COMPLETE_FENCED_JSON"
TRUNCATED_INSIDE_JSON = "TRUNCATED_INSIDE_JSON"
TRUNCATED_AFTER_JSON_BEFORE_FENCE = "TRUNCATED_AFTER_JSON_BEFORE_FENCE"
MALFORMED_FENCE_NOT_TRUNCATED = "MALFORMED_FENCE_NOT_TRUNCATED"
NON_JSON_RESPONSE = "NON_JSON_RESPONSE"
MULTIPLE_JSON_BLOCKS = "MULTIPLE_JSON_BLOCKS"
EXTRA_PROSE_AROUND_JSON = "EXTRA_PROSE_AROUND_JSON"
UNKNOWN = "UNKNOWN"
RESPONSE_STRUCTURE_CLASSIFICATIONS = (
    COMPLETE_PLAIN_JSON,
    COMPLETE_FENCED_JSON,
    TRUNCATED_INSIDE_JSON,
    TRUNCATED_AFTER_JSON_BEFORE_FENCE,
    MALFORMED_FENCE_NOT_TRUNCATED,
    NON_JSON_RESPONSE,
    MULTIPLE_JSON_BLOCKS,
    EXTRA_PROSE_AROUND_JSON,
    UNKNOWN,
)
PREFIX_EXCERPT_MAX_CHARS = 80
SUFFIX_EXCERPT_MAX_CHARS = 120
PROVIDER_METADATA_FIELDS = (
    "provider_finish_reason",
    "provider_stop_reason",
    "provider_max_output_tokens",
    "provider_output_limit_reached",
    "provider_prompt_tokens",
    "provider_visible_output_tokens",
    "provider_hidden_output_tokens",
    "provider_combined_output_tokens",
    "provider_output_budget_utilization_percent",
)

PROVIDER_ERROR_DIAGNOSTIC_FIELDS = (
    "provider_error_type",
    "provider_error_code",
    "provider_http_status",
    "provider_error_category",
    "provider_error_retryable",
    "provider_endpoint_family",
    "provider_model",
    "provider_error_message_safe",
)


@dataclass(frozen=True)
class SemanticGroundingResponseStructureDiagnostics:
    raw_response_character_count: int
    raw_response_prefix_excerpt: str
    raw_response_suffix_excerpt: str
    starts_with_json_object: bool
    ends_with_json_object: bool
    starts_with_code_fence: bool
    opening_fence_language: str | None
    contains_json_object_start_after_fence: bool
    contains_json_object_end: bool
    ends_with_code_fence: bool
    json_brace_balance: int
    json_string_appears_unterminated: bool
    markdown_fence_count: int
    response_structure_classification: str

    def __post_init__(self) -> None:
        if self.response_structure_classification not in RESPONSE_STRUCTURE_CLASSIFICATIONS:
            raise ValueError("unsupported Semantic Grounding response classification")

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_response_character_count": self.raw_response_character_count,
            "raw_response_prefix_excerpt": self.raw_response_prefix_excerpt,
            "raw_response_suffix_excerpt": self.raw_response_suffix_excerpt,
            "starts_with_json_object": self.starts_with_json_object,
            "ends_with_json_object": self.ends_with_json_object,
            "starts_with_code_fence": self.starts_with_code_fence,
            "opening_fence_language": self.opening_fence_language,
            "contains_json_object_start_after_fence": (
                self.contains_json_object_start_after_fence
            ),
            "contains_json_object_end": self.contains_json_object_end,
            "ends_with_code_fence": self.ends_with_code_fence,
            "json_brace_balance": self.json_brace_balance,
            "json_string_appears_unterminated": self.json_string_appears_unterminated,
            "markdown_fence_count": self.markdown_fence_count,
            "response_structure_classification": self.response_structure_classification,
        }


def build_semantic_grounding_response_structure_diagnostics(
    raw_text: str,
    *,
    provider_output_limit_reached: bool | None = None,
) -> SemanticGroundingResponseStructureDiagnostics:
    text = str(raw_text or "")
    stripped = text.strip()
    json_region = _json_region_for_shape(stripped)
    scan = _scan_json_shape(json_region)
    return SemanticGroundingResponseStructureDiagnostics(
        raw_response_character_count=len(text),
        raw_response_prefix_excerpt=_safe_excerpt(text, PREFIX_EXCERPT_MAX_CHARS),
        raw_response_suffix_excerpt=_safe_suffix_excerpt(
            text,
            SUFFIX_EXCERPT_MAX_CHARS,
        ),
        starts_with_json_object=stripped.startswith("{"),
        ends_with_json_object=stripped.endswith("}"),
        starts_with_code_fence=stripped.startswith("```"),
        opening_fence_language=_opening_fence_language(stripped),
        contains_json_object_start_after_fence=(
            _body_after_opening_fence(stripped).lstrip().startswith("{")
            if stripped.startswith("```")
            else False
        ),
        contains_json_object_end=scan["contains_json_object_end"],
        ends_with_code_fence=stripped.endswith("```"),
        json_brace_balance=scan["brace_balance"],
        json_string_appears_unterminated=scan["unterminated_string"],
        markdown_fence_count=stripped.count("```"),
        response_structure_classification=_classify_response_structure(
            stripped,
            scan,
            provider_output_limit_reached=provider_output_limit_reached,
        ),
    )


def build_semantic_grounding_raw_response_diagnostics(raw_response: Any) -> dict[str, Any]:
    raw_text = str(getattr(raw_response, "raw_text", "") or "")
    metadata = _safe_provider_response_metadata(
        getattr(raw_response, "provider_response_metadata", None)
    )
    diagnostics: dict[str, Any] = {
        **build_semantic_grounding_response_structure_diagnostics(
            raw_text,
            provider_output_limit_reached=metadata.get("provider_output_limit_reached"),
        ).to_dict(),
        **metadata,
    }
    usage = _safe_token_usage(getattr(raw_response, "usage", None))
    if usage:
        diagnostics["usage"] = usage
    provider_error = _safe_provider_error_diagnostics(
        getattr(raw_response, "execution_diagnostics", None)
    )
    if provider_error:
        diagnostics["provider_error_diagnostics"] = provider_error
    return diagnostics


def _safe_provider_response_metadata(metadata: Any) -> dict[str, Any]:
    result: dict[str, Any] = {key: None for key in PROVIDER_METADATA_FIELDS}
    if not isinstance(metadata, dict):
        return result
    for key in PROVIDER_METADATA_FIELDS:
        value = metadata.get(key)
        if key == "provider_output_limit_reached":
            if value is None or isinstance(value, bool):
                result[key] = value
        elif key == "provider_output_budget_utilization_percent":
            if value is None:
                result[key] = None
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                result[key] = round(float(value), 2)
        elif key in {"provider_finish_reason", "provider_stop_reason"}:
            if value is None or isinstance(value, str):
                result[key] = value[:120] if isinstance(value, str) else None
        elif value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            result[key] = value
    return result


def _safe_token_usage(usage: Any) -> dict[str, int | None]:
    if not isinstance(usage, dict):
        return {}
    safe: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            safe[key] = value
    return safe


def _safe_provider_error_diagnostics(diagnostics: Any) -> dict[str, Any]:
    if not isinstance(diagnostics, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in PROVIDER_ERROR_DIAGNOSTIC_FIELDS:
        value = diagnostics.get(key)
        if value is None:
            continue
        if key == "provider_error_retryable" and isinstance(value, bool):
            safe[key] = value
        elif key == "provider_http_status" and isinstance(value, int) and not isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, str):
            safe[key] = _safe_text(value)
    return safe


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
    first_line = stripped.splitlines()[0].strip() if stripped.splitlines() else ""
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
    if _has_multiple_json_objects(stripped):
        return MULTIPLE_JSON_BLOCKS
    if _is_complete_plain_json(stripped, scan):
        return COMPLETE_PLAIN_JSON
    if _is_complete_fenced_json(stripped, scan):
        return COMPLETE_FENCED_JSON
    if _has_extra_prose_around_json(stripped):
        return EXTRA_PROSE_AROUND_JSON
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


def _has_multiple_json_objects(stripped: str) -> bool:
    if stripped.startswith("```"):
        return stripped.count("```") > 2
    first_end = stripped.find("}")
    if first_end == -1:
        return False
    return stripped[first_end + 1 :].lstrip().startswith("{")


def _has_extra_prose_around_json(stripped: str) -> bool:
    if not stripped:
        return False
    if not stripped.startswith(("{", "```")) and "{" in stripped:
        return True
    if stripped.startswith("{") and "}" in stripped:
        return bool(stripped[stripped.rfind("}") + 1 :].strip())
    if stripped.startswith("```") and stripped.endswith("```"):
        return False
    return False


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


def _safe_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    lowered = text.lower()
    if any(marker in lowered for marker in ("sk-", "authorization", "api_key", "secret", "password", "credential", "bearer")):
        return "redacted safe error"
    return text[:240]
