"""Length-only model repair for overlong Candidate Writer post_text.

This module owns the narrow repair boundary between Candidate Writer adaptation
and the deterministic gate. It does not slice strings, retry recursively,
repair JSON, substitute providers, run semantic grounding, evaluate quality, or
persist data.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import is_dataclass
from dataclasses import asdict
import json
from typing import Any

from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterRawResponse,
    build_candidate_writer_execution_request,
    execute_candidate_writer_prompt,
)
from services.packaging.linkedin_post_candidate_writer_output_adapter import (
    CANONICAL_FINAL_POST_PAYLOAD_FIELDS,
    REQUIRED_FINAL_POST_PAYLOAD_FIELDS,
    CandidateWriterOutputAdaptationError,
    adapt_candidate_writer_payload,
)
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    ADAPTER_ERROR_INVALID_FIELD_VALUES,
    FIELD_VIOLATION_ABOVE_MAX_LENGTH,
    FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
    FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MAX_CHARS,
    FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MIN_CHARS,
)
from services.packaging.linkedin_post_prompt_renderers import CandidateWriterPromptRender


PROMPT_CANDIDATE_WRITER_LENGTH_REPAIR = "candidate_writer_length_repair"
PROMPT_CANDIDATE_WRITER_LENGTH_REPAIR_VERSION = "1.0"
PROMPT_CANDIDATE_WRITER_LENGTH_REPAIR_PATH = (
    "prompts/linkedin/candidate_writer_length_repair.txt"
)

DEFAULT_CANDIDATE_WRITER_LENGTH_REPAIR_MAX_OUTPUT_TOKENS = 800

FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_REQUEST = (
    "candidate_writer_length_repair_request_failure"
)
FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_PROVIDER = (
    "candidate_writer_length_repair_provider_failure"
)
FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_EMPTY_RESPONSE = (
    "candidate_writer_length_repair_empty_response"
)
FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_PARSE = (
    "candidate_writer_length_repair_parse_failure"
)
FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_VALIDATION = (
    "candidate_writer_length_repair_validation_failure"
)

REPAIR_RESPONSE_ALLOWED_KEYS = ("post_text",)


@dataclass(frozen=True)
class CandidateWriterLengthRepairEligibility:
    eligible: bool
    reason: str
    original_post_text_length: int | None = None
    maximum_allowed: int = FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "original_post_text_length": self.original_post_text_length,
            "maximum_allowed": self.maximum_allowed,
        }


@dataclass(frozen=True)
class CandidateWriterLengthRepairResult:
    repaired_candidate: dict[str, Any] | None
    provider_reply: CandidateWriterRawResponse | None
    eligibility: CandidateWriterLengthRepairEligibility
    repair_executed: bool
    failure_code: str | None = None
    failure_message: str = ""
    repaired_post_text_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repaired_candidate": copy.deepcopy(self.repaired_candidate),
            "provider_reply_metadata": _provider_reply_metadata(self.provider_reply),
            "eligibility": self.eligibility.to_dict(),
            "repair_executed": self.repair_executed,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "repaired_post_text_length": self.repaired_post_text_length,
        }


def assess_candidate_writer_length_repair_eligibility(
    *,
    parsed_candidate: dict[str, Any],
    adaptation_error: CandidateWriterOutputAdaptationError,
) -> CandidateWriterLengthRepairEligibility:
    """Return whether an adaptation failure is exactly post_text overlength."""

    if not isinstance(parsed_candidate, dict):
        return _ineligible("parsed candidate is not a dictionary")

    allowed_fields = set(CANONICAL_FINAL_POST_PAYLOAD_FIELDS)
    missing_fields = [
        field for field in REQUIRED_FINAL_POST_PAYLOAD_FIELDS if field not in parsed_candidate
    ]
    if missing_fields:
        return _ineligible("parsed candidate is missing required fields")

    unexpected_fields = [field for field in parsed_candidate if field not in allowed_fields]
    if unexpected_fields:
        return _ineligible("parsed candidate has unexpected fields")

    post_text = parsed_candidate.get("post_text")
    if not isinstance(post_text, str):
        return _ineligible("post_text is not a string")
    if not post_text.strip():
        return _ineligible("post_text is empty", original_length=len(post_text))
    if len(post_text) <= FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS:
        return _ineligible("post_text is not above maximum", original_length=len(post_text))

    diagnostics = getattr(adaptation_error, "diagnostics", None)
    if diagnostics is None:
        return _ineligible("missing adaptation diagnostics", original_length=len(post_text))
    if diagnostics.failure_stage != FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION:
        return _ineligible("failure is not adaptation", original_length=len(post_text))
    if diagnostics.adapter_error_code != ADAPTER_ERROR_INVALID_FIELD_VALUES:
        return _ineligible("adapter failure is not invalid field values", original_length=len(post_text))
    if tuple(diagnostics.missing_required_fields):
        return _ineligible("diagnostics include missing fields", original_length=len(post_text))
    if tuple(diagnostics.unexpected_fields):
        return _ineligible("diagnostics include unexpected fields", original_length=len(post_text))
    if tuple(diagnostics.invalid_field_names) != ("post_text",):
        return _ineligible("invalid fields are not exactly post_text", original_length=len(post_text))

    violations = tuple(diagnostics.field_violations)
    if len(violations) != 1:
        return _ineligible("field violations are not singular", original_length=len(post_text))
    violation = violations[0]
    if (
        violation.field_name != "post_text"
        or violation.reason_code != FIELD_VIOLATION_ABOVE_MAX_LENGTH
    ):
        return _ineligible("sole violation is not post_text above_max_length", original_length=len(post_text))

    probe_candidate = copy.deepcopy(parsed_candidate)
    probe_candidate["post_text"] = "Valid short candidate post text."
    try:
        adapt_candidate_writer_payload(probe_candidate)
    except CandidateWriterOutputAdaptationError:
        return _ineligible("non-post_text fields do not satisfy FinalPostPayload", original_length=len(post_text))

    return CandidateWriterLengthRepairEligibility(
        eligible=True,
        reason="post_text above maximum is the sole adaptation violation",
        original_post_text_length=len(post_text),
    )


def execute_candidate_writer_length_repair(
    *,
    parsed_candidate: dict[str, Any],
    adaptation_error: CandidateWriterOutputAdaptationError,
    post_brief: object | dict,
    angle_decision: object | dict,
    provider: str,
    model: str,
    prompt_text: str,
    client: Any | None = None,
    max_output_tokens: int = DEFAULT_CANDIDATE_WRITER_LENGTH_REPAIR_MAX_OUTPUT_TOKENS,
    thinking_mode: str | None = None,
) -> CandidateWriterLengthRepairResult:
    eligibility = assess_candidate_writer_length_repair_eligibility(
        parsed_candidate=parsed_candidate,
        adaptation_error=adaptation_error,
    )
    if not eligibility.eligible:
        return CandidateWriterLengthRepairResult(
            repaired_candidate=None,
            provider_reply=None,
            eligibility=eligibility,
            repair_executed=False,
        )

    try:
        render = render_candidate_writer_length_repair_prompt_input(
            parsed_candidate=parsed_candidate,
            post_brief=post_brief,
            angle_decision=angle_decision,
        )
        request = build_candidate_writer_execution_request(
            render,
            prompt_text=prompt_text,
            provider=provider,
            model=model,
            max_output_tokens=max_output_tokens,
            thinking_mode=thinking_mode,
            execution_metadata={"candidate_writer_length_repair": True},
        )
    except (TypeError, ValueError) as exc:
        return CandidateWriterLengthRepairResult(
            repaired_candidate=None,
            provider_reply=None,
            eligibility=eligibility,
            repair_executed=False,
            failure_code=FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_REQUEST,
            failure_message=str(exc),
        )
    provider_reply = execute_candidate_writer_prompt(request, client=client)
    if provider_reply.execution_error:
        return CandidateWriterLengthRepairResult(
            repaired_candidate=None,
            provider_reply=provider_reply,
            eligibility=eligibility,
            repair_executed=True,
            failure_code=_execution_failure_code(provider_reply),
            failure_message=provider_reply.execution_error,
        )

    try:
        repaired_post_text = parse_candidate_writer_length_repair_response(
            provider_reply
        )
    except ValueError as exc:
        failure_code = (
            FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_VALIDATION
            if "remains above maximum" in str(exc)
            else FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_PARSE
        )
        return CandidateWriterLengthRepairResult(
            repaired_candidate=None,
            provider_reply=provider_reply,
            eligibility=eligibility,
            repair_executed=True,
            failure_code=failure_code,
            failure_message=str(exc),
        )

    repaired_candidate = copy.deepcopy(parsed_candidate)
    repaired_candidate["post_text"] = repaired_post_text
    try:
        adapt_candidate_writer_payload(repaired_candidate)
    except CandidateWriterOutputAdaptationError as exc:
        return CandidateWriterLengthRepairResult(
            repaired_candidate=None,
            provider_reply=provider_reply,
            eligibility=eligibility,
            repair_executed=True,
            failure_code=FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_VALIDATION,
            failure_message=str(exc),
            repaired_post_text_length=len(repaired_post_text),
        )

    return CandidateWriterLengthRepairResult(
        repaired_candidate=repaired_candidate,
        provider_reply=provider_reply,
        eligibility=eligibility,
        repair_executed=True,
        repaired_post_text_length=len(repaired_post_text),
    )


def parse_candidate_writer_length_repair_response(
    provider_reply: CandidateWriterRawResponse,
) -> str:
    raw_text = provider_reply.raw_text
    if provider_reply.execution_error:
        raise ValueError("candidate writer length repair execution failed")
    if raw_text is None or not str(raw_text).strip():
        raise ValueError("candidate writer length repair response is empty")
    stripped = str(raw_text).strip()
    if not stripped.startswith("{"):
        raise ValueError("candidate writer length repair response must be JSON")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("candidate writer length repair response is malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("candidate writer length repair response must be an object")
    if tuple(sorted(payload)) != REPAIR_RESPONSE_ALLOWED_KEYS:
        raise ValueError("candidate writer length repair response must contain only post_text")
    post_text = payload.get("post_text")
    if not isinstance(post_text, str) or not post_text.strip():
        raise ValueError("candidate writer length repair post_text must be a non-empty string")
    if len(post_text) > FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS:
        raise ValueError("candidate writer length repair post_text remains above maximum")
    return post_text


def render_candidate_writer_length_repair_prompt_input(
    *,
    parsed_candidate: dict[str, Any],
    post_brief: object | dict,
    angle_decision: object | dict,
) -> CandidateWriterPromptRender:
    post_brief_payload = _serialize(post_brief)
    angle_decision_payload = _serialize(angle_decision)
    selected_evidence = _selected_evidence_from_post_brief(post_brief_payload)
    authorial_voice_directive = _authorial_voice_directive_from_angle(
        angle_decision_payload
    )
    variables = {
        "current_post_text_json": _stable_json(parsed_candidate["post_text"]),
        "length_constraints_json": _stable_json(
            {
                "actual_chars": len(parsed_candidate["post_text"]),
                "hard_max_chars": FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
                "prompt_target_min_chars": (
                    FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MIN_CHARS
                ),
                "prompt_target_max_chars": (
                    FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MAX_CHARS
                ),
            }
        ),
        "post_brief_json": _stable_json(post_brief_payload),
        "angle_decision_json": _stable_json(angle_decision_payload),
        "authorial_voice_directive_json": _stable_json(authorial_voice_directive),
        "selected_evidence_json": _stable_json(selected_evidence),
        "preservation_rules_json": _stable_json(
            {
                "change_allowed": ["post_text"],
                "preserve_exactly": [
                    field
                    for field in CANONICAL_FINAL_POST_PAYLOAD_FIELDS
                    if field != "post_text"
                ],
                "do_not_add_facts": True,
                "do_not_output_evidence_ids_in_human_text": True,
            }
        ),
    }
    return CandidateWriterPromptRender(
        prompt_name=PROMPT_CANDIDATE_WRITER_LENGTH_REPAIR,
        prompt_version=PROMPT_CANDIDATE_WRITER_LENGTH_REPAIR_VERSION,
        prompt_path=PROMPT_CANDIDATE_WRITER_LENGTH_REPAIR_PATH,
        variables=variables,
        input_text=_build_input_text(variables),
    )


def _ineligible(
    reason: str,
    *,
    original_length: int | None = None,
) -> CandidateWriterLengthRepairEligibility:
    return CandidateWriterLengthRepairEligibility(
        eligible=False,
        reason=reason,
        original_post_text_length=original_length,
    )


def _execution_failure_code(provider_reply: CandidateWriterRawResponse) -> str:
    if provider_reply.execution_error == "empty provider response":
        return FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_EMPTY_RESPONSE
    if str(provider_reply.execution_error or "").startswith(("missing ", "invalid ")):
        return FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_REQUEST
    return FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_PROVIDER


def _provider_reply_metadata(
    provider_reply: CandidateWriterRawResponse | None,
) -> dict[str, Any] | None:
    if provider_reply is None:
        return None
    metadata = {
        "provider": provider_reply.provider,
        "model": provider_reply.model,
        "usage": copy.deepcopy(provider_reply.usage),
        "execution_error": provider_reply.execution_error,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _selected_evidence_from_post_brief(post_brief: Any) -> list[dict[str, Any]]:
    if not isinstance(post_brief, dict):
        raise TypeError("post_brief must serialize to a dictionary.")
    evidence_to_use = post_brief.get("evidence_to_use")
    if not isinstance(evidence_to_use, list):
        raise TypeError("post_brief.evidence_to_use must serialize to a list.")
    selected: list[dict[str, Any]] = []
    for item in evidence_to_use:
        if not isinstance(item, dict):
            raise TypeError("post_brief.evidence_to_use items must be dictionaries.")
        selected.append(
            {
                "evidence_id": item.get("evidence_id"),
                "evidence_text": item.get("evidence_text"),
                "role_in_post": item.get("role_in_post"),
            }
        )
    return selected


def _authorial_voice_directive_from_angle(angle_decision: Any) -> dict[str, Any]:
    if not isinstance(angle_decision, dict):
        raise TypeError("angle_decision must serialize to a dictionary.")
    directive = angle_decision.get("authorial_voice_directive")
    if not isinstance(directive, dict):
        raise TypeError("angle_decision.authorial_voice_directive must serialize to a dictionary.")
    return copy.deepcopy(directive)


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize(value.to_dict())
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    return copy.deepcopy(value)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _build_input_text(variables: dict[str, str]) -> str:
    sections = (
        ("CURRENT_POST_TEXT_JSON", variables["current_post_text_json"]),
        ("LENGTH_CONSTRAINTS_JSON", variables["length_constraints_json"]),
        ("POST_BRIEF_JSON", variables["post_brief_json"]),
        ("ANGLE_DECISION_JSON", variables["angle_decision_json"]),
        (
            "AUTHORIAL_VOICE_DIRECTIVE_JSON",
            variables["authorial_voice_directive_json"],
        ),
        ("SELECTED_EVIDENCE_JSON", variables["selected_evidence_json"]),
        ("PRESERVATION_RULES_JSON", variables["preservation_rules_json"]),
    )
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)
