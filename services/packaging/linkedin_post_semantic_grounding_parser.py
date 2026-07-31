"""Strict parser for raw semantic grounding evaluator responses."""
from __future__ import annotations

import json
from typing import Any

from services.packaging.linkedin_post_semantic_grounding_contract import (
    SemanticGroundingReviewResult,
    normalize_semantic_grounding_review_result,
)
from services.packaging.linkedin_post_semantic_grounding_execution import (
    SemanticGroundingRawResponse,
)


ERROR_EXECUTION_FAILED = "execution_failed"
ERROR_EMPTY_RAW_RESPONSE = "empty_raw_response"
ERROR_MALFORMED_FENCE = "malformed_fence"
ERROR_MALFORMED_JSON = "malformed_json"
ERROR_NON_OBJECT_JSON = "non_object_json"
ERROR_NORMALIZATION_FAILED = "normalization_failed"


class SemanticGroundingResponseParseError(ValueError):
    """Raised when a semantic grounding raw response cannot be parsed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_semantic_grounding_raw_response(
    raw_response: SemanticGroundingRawResponse,
) -> dict[str, Any]:
    if raw_response.execution_error:
        raise SemanticGroundingResponseParseError(
            ERROR_EXECUTION_FAILED,
            "semantic grounding execution failed before parsing.",
        )
    raw_text = raw_response.raw_text
    if raw_text is None or not str(raw_text).strip():
        raise SemanticGroundingResponseParseError(
            ERROR_EMPTY_RAW_RESPONSE,
            "semantic grounding raw response is empty.",
        )

    json_text = _extract_supported_json_text(str(raw_text))
    try:
        payload = json.loads(
            json_text,
            parse_constant=_reject_non_standard_number,
        )
    except json.JSONDecodeError as exc:
        raise SemanticGroundingResponseParseError(
            ERROR_MALFORMED_JSON,
            "semantic grounding raw response is not valid JSON.",
        ) from exc
    except ValueError as exc:
        raise SemanticGroundingResponseParseError(
            ERROR_MALFORMED_JSON,
            "semantic grounding raw response contains non-standard JSON numbers.",
        ) from exc

    if not isinstance(payload, dict):
        raise SemanticGroundingResponseParseError(
            ERROR_NON_OBJECT_JSON,
            "semantic grounding raw response must be a JSON object.",
        )
    return payload


def parse_and_normalize_semantic_grounding_response(
    raw_response: SemanticGroundingRawResponse,
    *,
    selected_evidence_ids: tuple[str, ...] | list[str],
) -> SemanticGroundingReviewResult:
    parsed = parse_semantic_grounding_raw_response(raw_response)
    try:
        return normalize_semantic_grounding_review_result(
            parsed,
            selected_evidence_ids=selected_evidence_ids,
        )
    except ValueError as exc:
        raise SemanticGroundingResponseParseError(
            ERROR_NORMALIZATION_FAILED,
            "semantic grounding response failed review normalization.",
        ) from exc


def _extract_supported_json_text(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        return _extract_fenced_json_text(stripped)
    if not stripped.startswith("{"):
        if "```" in stripped:
            raise SemanticGroundingResponseParseError(
                ERROR_MALFORMED_FENCE,
                "semantic grounding raw response has malformed markdown fencing.",
            )
        return stripped
    return stripped


def _extract_fenced_json_text(stripped: str) -> str:
    lines = stripped.splitlines()
    if len(lines) < 3:
        raise SemanticGroundingResponseParseError(
            ERROR_MALFORMED_FENCE,
            "semantic grounding raw response has incomplete markdown fencing.",
        )
    opening = lines[0].strip()
    if opening not in {"```", "```json"}:
        raise SemanticGroundingResponseParseError(
            ERROR_MALFORMED_FENCE,
            "semantic grounding raw response uses unsupported markdown fencing.",
        )
    if lines[-1].strip() != "```":
        raise SemanticGroundingResponseParseError(
            ERROR_MALFORMED_FENCE,
            "semantic grounding raw response has malformed markdown fencing.",
        )
    body_lines = lines[1:-1]
    if any(line.strip().startswith("```") for line in body_lines):
        raise SemanticGroundingResponseParseError(
            ERROR_MALFORMED_FENCE,
            "semantic grounding raw response has nested markdown fencing.",
        )
    body = "\n".join(body_lines).strip()
    if not body:
        raise SemanticGroundingResponseParseError(
            ERROR_EMPTY_RAW_RESPONSE,
            "semantic grounding fenced raw response is empty.",
        )
    return body


def _reject_non_standard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")
