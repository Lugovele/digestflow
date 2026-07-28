"""Strict parser for raw LinkedIn quality evaluator responses.

This module converts a provider raw response into a plain dictionary and can
hand that dictionary to the existing quality-review normalizer. It does not
call providers, render prompts, route decisions, repair text, persist data, or
connect to runtime packaging.
"""
from __future__ import annotations

import json
from typing import Any

from services.packaging.linkedin_post_quality_evaluator_execution import (
    QualityEvaluatorRawResponse,
)
from services.packaging.linkedin_post_quality_review_contract import (
    normalize_quality_review_result,
)


ERROR_EXECUTION_FAILED = "execution_failed"
ERROR_EMPTY_RAW_RESPONSE = "empty_raw_response"
ERROR_MALFORMED_FENCE = "malformed_fence"
ERROR_MALFORMED_JSON = "malformed_json"
ERROR_NON_OBJECT_JSON = "non_object_json"
ERROR_NORMALIZATION_FAILED = "normalization_failed"


class QualityEvaluatorResponseParseError(ValueError):
    """Raised when a quality evaluator raw response cannot be parsed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_quality_evaluator_raw_response(
    raw_response: QualityEvaluatorRawResponse,
) -> dict[str, Any]:
    """Parse raw evaluator text into an independent top-level JSON object."""

    if raw_response.execution_error:
        raise QualityEvaluatorResponseParseError(
            ERROR_EXECUTION_FAILED,
            "quality evaluator execution failed before parsing.",
        )

    raw_text = raw_response.raw_text
    if raw_text is None or not str(raw_text).strip():
        raise QualityEvaluatorResponseParseError(
            ERROR_EMPTY_RAW_RESPONSE,
            "quality evaluator raw response is empty.",
        )

    json_text = _extract_supported_json_text(str(raw_text))
    try:
        payload = json.loads(
            json_text,
            parse_constant=_reject_non_standard_number,
        )
    except json.JSONDecodeError as exc:
        raise QualityEvaluatorResponseParseError(
            ERROR_MALFORMED_JSON,
            "quality evaluator raw response is not valid JSON.",
        ) from exc
    except ValueError as exc:
        raise QualityEvaluatorResponseParseError(
            ERROR_MALFORMED_JSON,
            "quality evaluator raw response contains non-standard JSON numbers.",
        ) from exc

    if not isinstance(payload, dict):
        raise QualityEvaluatorResponseParseError(
            ERROR_NON_OBJECT_JSON,
            "quality evaluator raw response must be a JSON object.",
        )
    return payload


def parse_and_normalize_quality_evaluator_response(
    raw_response: QualityEvaluatorRawResponse,
) -> dict[str, Any]:
    """Parse raw evaluator text and return the canonical quality-review dict."""

    parsed = parse_quality_evaluator_raw_response(raw_response)
    try:
        return normalize_quality_review_result(parsed)
    except ValueError as exc:
        raise QualityEvaluatorResponseParseError(
            ERROR_NORMALIZATION_FAILED,
            "quality evaluator response failed quality-review normalization.",
        ) from exc


def _extract_supported_json_text(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        return _extract_fenced_json_text(stripped)
    if not stripped.startswith("{"):
        if "```" in stripped:
            raise QualityEvaluatorResponseParseError(
                ERROR_MALFORMED_FENCE,
                "quality evaluator raw response has malformed markdown fencing.",
            )
        return stripped
    return stripped


def _extract_fenced_json_text(stripped: str) -> str:
    lines = stripped.splitlines()
    if len(lines) < 3:
        raise QualityEvaluatorResponseParseError(
            ERROR_MALFORMED_FENCE,
            "quality evaluator raw response has incomplete markdown fencing.",
        )

    opening = lines[0].strip()
    if opening not in {"```", "```json"}:
        raise QualityEvaluatorResponseParseError(
            ERROR_MALFORMED_FENCE,
            "quality evaluator raw response uses unsupported markdown fencing.",
        )
    if lines[-1].strip() != "```":
        raise QualityEvaluatorResponseParseError(
            ERROR_MALFORMED_FENCE,
            "quality evaluator raw response has malformed markdown fencing.",
        )

    body_lines = lines[1:-1]
    if any(line.strip().startswith("```") for line in body_lines):
        raise QualityEvaluatorResponseParseError(
            ERROR_MALFORMED_FENCE,
            "quality evaluator raw response has nested markdown fencing.",
        )
    body = "\n".join(body_lines).strip()
    if not body:
        raise QualityEvaluatorResponseParseError(
            ERROR_EMPTY_RAW_RESPONSE,
            "quality evaluator fenced raw response is empty.",
        )
    return body


def _reject_non_standard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")
