"""Strict parser for raw LinkedIn Candidate Writer responses.

This module converts a provider raw response into a plain dictionary. It does
not adapt the parsed object into FinalPostPayload, build CandidateWriterOutput,
run deterministic gates, evaluate quality, repair text, persist data, or
connect to runtime packaging.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.packaging.linkedin_post_candidate_writer_execution import (
        CandidateWriterRawResponse,
    )


ERROR_EXECUTION_FAILED = "execution_failed"
ERROR_EMPTY_RAW_RESPONSE = "empty_raw_response"
ERROR_MALFORMED_FENCE = "malformed_fence"
ERROR_MALFORMED_JSON = "malformed_json"
ERROR_NON_OBJECT_JSON = "non_object_json"


class CandidateWriterResponseParseError(ValueError):
    """Raised when a Candidate Writer raw response cannot be parsed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_candidate_writer_raw_response(
    raw_response: CandidateWriterRawResponse,
) -> dict[str, Any]:
    """Parse raw Candidate Writer text into an independent JSON object."""

    if raw_response.execution_error:
        raise CandidateWriterResponseParseError(
            ERROR_EXECUTION_FAILED,
            "candidate writer execution failed before parsing.",
        )

    raw_text = raw_response.raw_text
    if raw_text is None or not str(raw_text).strip():
        raise CandidateWriterResponseParseError(
            ERROR_EMPTY_RAW_RESPONSE,
            "candidate writer raw response is empty.",
        )

    json_text = _extract_supported_json_text(str(raw_text))
    try:
        payload = json.loads(
            json_text,
            parse_constant=_reject_non_standard_number,
        )
    except json.JSONDecodeError as exc:
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_JSON,
            "candidate writer raw response is not valid JSON.",
        ) from exc
    except ValueError as exc:
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_JSON,
            "candidate writer raw response contains non-standard JSON numbers.",
        ) from exc

    if not isinstance(payload, dict):
        raise CandidateWriterResponseParseError(
            ERROR_NON_OBJECT_JSON,
            "candidate writer raw response must be a JSON object.",
        )
    return payload


def _extract_supported_json_text(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        return _extract_fenced_json_text(stripped)
    if not stripped.startswith("{"):
        if "```" in stripped:
            raise CandidateWriterResponseParseError(
                ERROR_MALFORMED_FENCE,
                "candidate writer raw response has malformed markdown fencing.",
            )
        return stripped
    return stripped


def _extract_fenced_json_text(stripped: str) -> str:
    lines = stripped.splitlines()
    if len(lines) < 3:
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_FENCE,
            "candidate writer raw response has incomplete markdown fencing.",
        )

    opening = lines[0].strip()
    if opening not in {"```", "```json"}:
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_FENCE,
            "candidate writer raw response uses unsupported markdown fencing.",
        )
    if lines[-1].strip() != "```":
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_FENCE,
            "candidate writer raw response has malformed markdown fencing.",
        )

    body_lines = lines[1:-1]
    if any(line.strip().startswith("```") for line in body_lines):
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_FENCE,
            "candidate writer raw response has nested markdown fencing.",
        )
    body = "\n".join(body_lines).strip()
    if not body:
        raise CandidateWriterResponseParseError(
            ERROR_EMPTY_RAW_RESPONSE,
            "candidate writer fenced raw response is empty.",
        )
    return body


def _reject_non_standard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")
