"""Strict parser for raw LinkedIn Candidate Writer responses.

This module converts a provider raw response into a plain dictionary. It does
not adapt the parsed object into FinalPostPayload, build CandidateWriterOutput,
run deterministic gates, evaluate quality, repair text, persist data, or
connect to runtime packaging.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    CandidateWriterStructuralDiagnostics,
    PARSER_DETAIL_EXPECTED_COLON,
    PARSER_DETAIL_EXPECTED_COMMA_DELIMITER,
    PARSER_DETAIL_EXPECTED_PROPERTY_NAME,
    PARSER_DETAIL_EXPECTED_VALUE,
    PARSER_DETAIL_EXTRA_DATA_AFTER_JSON,
    PARSER_DETAIL_INVALID_CONTROL_CHARACTER,
    PARSER_DETAIL_INVALID_ESCAPE,
    PARSER_DETAIL_NON_STANDARD_NUMBER,
    PARSER_DETAIL_TRAILING_COMMA,
    PARSER_DETAIL_UNEXPECTED_END_OF_INPUT,
    PARSER_DETAIL_UNTERMINATED_STRING,
    PARSER_DETAIL_UNKNOWN_JSON_SYNTAX,
    build_parser_structural_diagnostics,
)

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

    def __init__(
        self,
        code: str,
        message: str,
        *,
        diagnostics: CandidateWriterStructuralDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics


def parse_candidate_writer_raw_response(
    raw_response: CandidateWriterRawResponse,
) -> dict[str, Any]:
    """Parse raw Candidate Writer text into an independent JSON object."""

    if raw_response.execution_error:
        raise CandidateWriterResponseParseError(
            ERROR_EXECUTION_FAILED,
            "candidate writer execution failed before parsing.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_EXECUTION_FAILED,
                candidate_text=getattr(raw_response, "raw_text", None),
            ),
        )

    raw_text = raw_response.raw_text
    if raw_text is None or not str(raw_text).strip():
        raise CandidateWriterResponseParseError(
            ERROR_EMPTY_RAW_RESPONSE,
            "candidate writer raw response is empty.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_EMPTY_RAW_RESPONSE,
                candidate_text=raw_text,
            ),
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
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_MALFORMED_JSON,
                candidate_text=raw_text,
                parser_error_detail_code=_json_error_detail_code(json_text, exc),
                parser_error_line=exc.lineno,
                parser_error_column=exc.colno,
                parser_error_position=exc.pos,
            ),
        ) from exc
    except ValueError as exc:
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_JSON,
            "candidate writer raw response contains non-standard JSON numbers.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_MALFORMED_JSON,
                candidate_text=raw_text,
                parser_error_detail_code=PARSER_DETAIL_NON_STANDARD_NUMBER,
            ),
        ) from exc

    if not isinstance(payload, dict):
        raise CandidateWriterResponseParseError(
            ERROR_NON_OBJECT_JSON,
            "candidate writer raw response must be a JSON object.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_NON_OBJECT_JSON,
                candidate_text=raw_text,
                top_level_json_type=type(payload).__name__,
            ),
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
                diagnostics=build_parser_structural_diagnostics(
                    parser_error_code=ERROR_MALFORMED_FENCE,
                    candidate_text=raw_text,
                ),
            )
        return stripped
    return stripped


def _extract_fenced_json_text(stripped: str) -> str:
    lines = stripped.splitlines()
    if len(lines) < 3:
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_FENCE,
            "candidate writer raw response has incomplete markdown fencing.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_MALFORMED_FENCE,
                candidate_text=stripped,
            ),
        )

    opening = lines[0].strip()
    if opening not in {"```", "```json"}:
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_FENCE,
            "candidate writer raw response uses unsupported markdown fencing.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_MALFORMED_FENCE,
                candidate_text=stripped,
            ),
        )
    if lines[-1].strip() != "```":
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_FENCE,
            "candidate writer raw response has malformed markdown fencing.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_MALFORMED_FENCE,
                candidate_text=stripped,
            ),
        )

    body_lines = lines[1:-1]
    if any(line.strip().startswith("```") for line in body_lines):
        raise CandidateWriterResponseParseError(
            ERROR_MALFORMED_FENCE,
            "candidate writer raw response has nested markdown fencing.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_MALFORMED_FENCE,
                candidate_text=stripped,
            ),
        )
    body = "\n".join(body_lines).strip()
    if not body:
        raise CandidateWriterResponseParseError(
            ERROR_EMPTY_RAW_RESPONSE,
            "candidate writer fenced raw response is empty.",
            diagnostics=build_parser_structural_diagnostics(
                parser_error_code=ERROR_EMPTY_RAW_RESPONSE,
                candidate_text=stripped,
            ),
        )
    return body


def _reject_non_standard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def _json_error_detail_code(json_text: str, exc: json.JSONDecodeError) -> str:
    message = exc.msg.lower()
    stripped = json_text.strip()
    if "trailing comma" in message:
        return PARSER_DETAIL_TRAILING_COMMA
    if "unterminated string" in message:
        return PARSER_DETAIL_UNTERMINATED_STRING
    if "invalid control character" in message:
        return PARSER_DETAIL_INVALID_CONTROL_CHARACTER
    if "invalid \\escape" in message:
        return PARSER_DETAIL_INVALID_ESCAPE
    if "extra data" in message:
        return PARSER_DETAIL_EXTRA_DATA_AFTER_JSON
    if "expecting property name enclosed in double quotes" in message:
        if _looks_like_trailing_comma(stripped, exc.pos):
            return PARSER_DETAIL_TRAILING_COMMA
        return PARSER_DETAIL_EXPECTED_PROPERTY_NAME
    if "expecting ':' delimiter" in message:
        return PARSER_DETAIL_EXPECTED_COLON
    if "expecting ',' delimiter" in message:
        if exc.pos >= max(0, len(stripped) - 1):
            return PARSER_DETAIL_UNEXPECTED_END_OF_INPUT
        return PARSER_DETAIL_EXPECTED_COMMA_DELIMITER
    if "expecting value" in message:
        if exc.pos >= len(stripped):
            return PARSER_DETAIL_UNEXPECTED_END_OF_INPUT
        return PARSER_DETAIL_EXPECTED_VALUE
    if exc.pos >= len(stripped):
        return PARSER_DETAIL_UNEXPECTED_END_OF_INPUT
    return PARSER_DETAIL_UNKNOWN_JSON_SYNTAX


def _looks_like_trailing_comma(json_text: str, position: int) -> bool:
    if position <= 0:
        return False
    prefix = json_text[:position].rstrip()
    return prefix.endswith(",")
