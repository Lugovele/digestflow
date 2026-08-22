"""Pure prompt input renderers for future final LinkedIn post flow agents."""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
import json
from typing import Any

from services.packaging.linkedin_post_editorial_boundary import PostEditorialInput
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_input_builders import CandidateWriterInput
from services.packaging.linkedin_post_pipeline import (
    AUTHORIAL_FIRST_PERSON_ALLOWED_NOT_REQUIRED,
)
from services.packaging.linkedin_post_pipeline import (
    AUTHORIAL_PERSONAL_PRESENCE_REQUIREMENTS,
)
from services.packaging.linkedin_post_pipeline import build_final_post_payload_constraints
from services.packaging.linkedin_post_quality_rubric_contract import (
    QualityEvaluatorRubricPayload,
)
from services.packaging.linkedin_post_semantic_grounding_contract import (
    build_semantic_grounding_prompt_rules,
)


FINAL_POST_PAYLOAD_PROMPT_FIELDS = (
    "post_text",
    "hook_variants",
    "cta_variants",
    "hashtags",
    "quality_checks",
    "carousel_outline",
)

SELECTED_EVIDENCE_PROMPT_FIELDS = (
    "evidence_id",
    "evidence_text",
    "role_in_post",
)

REPAIR_POST_BRIEF_PROMPT_FIELDS = (
    "opening_direction",
    "pattern_interrupt",
    "core_point",
    "evidence_to_use",
    "practical_point",
    "ending_direction",
    "cta_direction",
)

REPAIR_ANGLE_DECISION_PROMPT_FIELDS = (
    "controlling_angle",
    "reader_problem",
    "author_position",
    "main_tension",
    "supporting_evidence_ids",
    "angle_to_avoid",
)

PERSONAL_PRESENCE_INSTRUCTIONS = {
    "explicit_author_owned_statement_required": (
        "Include exactly one naturally integrated author-owned interpretive "
        "statement derived from authorial_observation, rejected_reading, or "
        "why_distinction_matters. Align it with AngleDecision.controlling_angle. "
        "An impersonal editorial judgment is not sufficient. Do not force stock "
        "wording, do not require first person at the opening, and do not add "
        "more than one ownership statement. Do not introduce new facts or "
        "invent personal experience, professional authority, client/customer "
        "evidence, biography, or emotional history."
    ),
    "author_owned_statement_allowed": (
        "An author-owned interpretive statement is allowed when natural, but "
        "not required. If used, derive it from authorial_observation, "
        "rejected_reading, or why_distinction_matters and avoid invented "
        "experience, authority, client/customer evidence, biography, or "
        "emotional history."
    ),
    "editorial_stance_only": (
        "Use an editorial stance without adding explicit personal-presence "
        "wording. Preserve the authorial judgment from authorial_observation, "
        "rejected_reading, or why_distinction_matters, but do not add invented "
        "experience, authority, client/customer evidence, biography, or "
        "emotional history."
    ),
}


@dataclass(frozen=True)
class CandidateWriterPromptRender:
    prompt_name: str | None
    prompt_version: str | None
    prompt_path: str | None
    variables: dict[str, str]
    input_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_path": self.prompt_path,
            "variables": dict(self.variables),
            "input_text": self.input_text,
        }


@dataclass(frozen=True)
class QualityEvaluatorPromptRender:
    prompt_name: str | None
    prompt_version: str | None
    prompt_path: str | None
    variables: dict[str, str]
    input_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_path": self.prompt_path,
            "variables": dict(self.variables),
            "input_text": self.input_text,
        }


@dataclass(frozen=True)
class SemanticGroundingPromptRender:
    prompt_name: str | None
    prompt_version: str | None
    prompt_path: str | None
    variables: dict[str, str]
    input_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_path": self.prompt_path,
            "variables": dict(self.variables),
            "input_text": self.input_text,
        }


@dataclass(frozen=True)
class RepairWriterPromptRender:
    prompt_name: str | None
    prompt_version: str | None
    prompt_path: str | None
    variables: dict[str, str]
    input_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_path": self.prompt_path,
            "variables": dict(self.variables),
            "input_text": self.input_text,
        }


def render_candidate_writer_prompt_input(
    candidate_input: CandidateWriterInput,
) -> CandidateWriterPromptRender:
    candidate_input_dict = candidate_input.to_dict()
    authorial_voice_directive = _authorial_voice_directive_for_prompt(
        candidate_input_dict["angle_decision"]
    )
    variables = {
        "post_brief_json": _stable_json(candidate_input_dict["post_brief"]),
        "angle_decision_json": _stable_json(candidate_input_dict["angle_decision"]),
        "authorial_voice_directive_json": _stable_json(authorial_voice_directive),
        "personal_presence_instruction": _personal_presence_instruction_for_prompt(
            authorial_voice_directive
        ),
        "selected_evidence_json": _stable_json(
            candidate_input_dict["selected_evidence"]
        ),
        "final_post_payload_constraints_json": _stable_json(
            build_final_post_payload_constraints()
        ),
        "candidate_writer_input_json": _stable_json(candidate_input_dict),
    }
    prompt_metadata = candidate_input.prompt_metadata

    return CandidateWriterPromptRender(
        prompt_name=prompt_metadata.prompt_name if prompt_metadata else None,
        prompt_version=prompt_metadata.prompt_version if prompt_metadata else None,
        prompt_path=prompt_metadata.prompt_path if prompt_metadata else None,
        variables=variables,
        input_text=_build_input_text(variables),
    )


def render_repair_writer_prompt_input(
    *,
    original_candidate_payload: dict[str, Any],
    post_brief: object | dict,
    angle_decision: object | dict,
    selected_evidence: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    deterministic_findings: dict[str, Any],
    quality_findings: dict[str, Any],
    repair_instruction: dict[str, Any],
    attempt_index: int,
    max_attempts: int,
    prompt_metadata: PromptMetadata | None = None,
) -> RepairWriterPromptRender:
    selected_evidence_ids = _selected_evidence_ids_for_repair_prompt(selected_evidence)
    authorial_voice_directive = _authorial_voice_directive_for_prompt(angle_decision)
    variables = {
        "original_candidate_payload_json": _stable_json(
            _candidate_payload_for_quality_prompt(original_candidate_payload)
        ),
        "post_brief_json": _stable_json(
            _post_brief_for_repair_prompt(post_brief, selected_evidence_ids)
        ),
        "angle_decision_json": _stable_json(
            _angle_decision_for_repair_prompt(angle_decision, selected_evidence_ids)
        ),
        "authorial_voice_directive_json": _stable_json(authorial_voice_directive),
        "personal_presence_instruction": _personal_presence_instruction_for_prompt(
            authorial_voice_directive
        ),
        "selected_evidence_json": _stable_json(
            _selected_evidence_for_quality_prompt(selected_evidence)
        ),
        "deterministic_findings_json": _stable_json(
            _repair_dict(deterministic_findings, "deterministic_findings")
        ),
        "quality_findings_json": _stable_json(
            _repair_dict(quality_findings, "quality_findings")
        ),
        "repair_instruction_json": _stable_json(
            _repair_dict(repair_instruction, "repair_instruction")
        ),
        "repair_attempt_json": _stable_json(
            {
                "attempt_index": attempt_index,
                "max_attempts": max_attempts,
                "repair_type": repair_instruction.get("repair_type", "editorial"),
            }
        ),
    }
    return RepairWriterPromptRender(
        prompt_name=prompt_metadata.prompt_name if prompt_metadata else None,
        prompt_version=prompt_metadata.prompt_version if prompt_metadata else None,
        prompt_path=prompt_metadata.prompt_path if prompt_metadata else None,
        variables=variables,
        input_text=_build_repair_writer_input_text(variables),
    )


def render_quality_evaluator_prompt_input(
    editorial_input: PostEditorialInput,
    quality_rubric: QualityEvaluatorRubricPayload,
    prompt_metadata: PromptMetadata | None = None,
) -> QualityEvaluatorPromptRender:
    variables = {
        "candidate_payload_json": _stable_json(
            _candidate_payload_for_quality_prompt(editorial_input.candidate_payload)
        ),
        "post_brief_json": _stable_json(
            _serialize_render_value(editorial_input.post_brief)
        ),
        "angle_decision_json": _stable_json(
            _serialize_render_value(editorial_input.angle_decision)
        ),
        "authorial_voice_directive_json": _stable_json(
            _authorial_voice_directive_for_prompt(editorial_input.angle_decision)
        ),
        "selected_evidence_json": _stable_json(
            _selected_evidence_for_quality_prompt(editorial_input.selected_evidence)
        ),
        "quality_rubric_json": _stable_json(quality_rubric.to_prompt_dict()),
    }

    return QualityEvaluatorPromptRender(
        prompt_name=prompt_metadata.prompt_name if prompt_metadata else None,
        prompt_version=prompt_metadata.prompt_version if prompt_metadata else None,
        prompt_path=prompt_metadata.prompt_path if prompt_metadata else None,
        variables=variables,
        input_text=_build_quality_evaluator_input_text(variables),
    )


def render_semantic_grounding_prompt_input(
    editorial_input: PostEditorialInput,
    prompt_metadata: PromptMetadata | None = None,
) -> SemanticGroundingPromptRender:
    selected_evidence_ids = _selected_evidence_ids_for_repair_prompt(
        editorial_input.selected_evidence
    )
    variables = {
        "candidate_payload_json": _stable_json(
            _candidate_payload_for_quality_prompt(editorial_input.candidate_payload)
        ),
        "post_brief_json": _stable_json(
            _post_brief_for_repair_prompt(
                editorial_input.post_brief,
                selected_evidence_ids,
            )
        ),
        "angle_decision_json": _stable_json(
            _angle_decision_for_repair_prompt(
                editorial_input.angle_decision,
                selected_evidence_ids,
            )
        ),
        "selected_evidence_json": _stable_json(
            _selected_evidence_for_quality_prompt(editorial_input.selected_evidence)
        ),
        "semantic_grounding_rules_json": _stable_json(
            build_semantic_grounding_prompt_rules()
        ),
    }

    return SemanticGroundingPromptRender(
        prompt_name=prompt_metadata.prompt_name if prompt_metadata else None,
        prompt_version=prompt_metadata.prompt_version if prompt_metadata else None,
        prompt_path=prompt_metadata.prompt_path if prompt_metadata else None,
        variables=variables,
        input_text=_build_semantic_grounding_input_text(variables),
    )


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _serialize_render_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return _serialize_render_value(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize_render_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_render_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_render_value(item) for item in value]
    return value


def _authorial_voice_directive_for_prompt(angle_decision: Any) -> dict[str, Any]:
    serialized = _serialize_render_value(angle_decision)
    if not isinstance(serialized, dict):
        raise TypeError("angle_decision must serialize to a dictionary.")
    directive = serialized.get("authorial_voice_directive")
    if directive is None:
        raise TypeError(
            "angle_decision.authorial_voice_directive must be present and serialize "
            "to a dictionary."
        )
    if not isinstance(directive, dict):
        raise TypeError(
            "angle_decision.authorial_voice_directive must serialize to a dictionary."
        )
    _validate_authorial_voice_directive_for_prompt(directive)
    return directive


def _validate_authorial_voice_directive_for_prompt(
    directive: dict[str, Any],
) -> None:
    for field_name in (
        "authorial_observation",
        "rejected_reading",
        "why_distinction_matters",
        "personal_presence_requirement",
        "first_person_policy",
    ):
        field_value = directive.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise TypeError(
                "angle_decision.authorial_voice_directive."
                f"{field_name} must be present and serialize to a non-empty string."
            )
    if directive["personal_presence_requirement"] not in AUTHORIAL_PERSONAL_PRESENCE_REQUIREMENTS:
        raise ValueError(
            "angle_decision.authorial_voice_directive.personal_presence_requirement "
            "must be a supported personal-presence policy."
        )
    if directive["first_person_policy"] != AUTHORIAL_FIRST_PERSON_ALLOWED_NOT_REQUIRED:
        raise ValueError(
            "angle_decision.authorial_voice_directive.first_person_policy "
            f"must be {AUTHORIAL_FIRST_PERSON_ALLOWED_NOT_REQUIRED}."
        )
    forbidden_claims = directive.get("forbidden_author_claims")
    if not isinstance(forbidden_claims, (list, tuple)) or not forbidden_claims:
        raise TypeError(
            "angle_decision.authorial_voice_directive.forbidden_author_claims "
            "must serialize to a non-empty list or tuple."
        )
    for claim in forbidden_claims:
        if not isinstance(claim, str) or not claim.strip():
            raise TypeError(
                "angle_decision.authorial_voice_directive.forbidden_author_claims "
                "must contain non-empty strings."
            )


def _personal_presence_instruction_for_prompt(
    authorial_voice_directive: dict[str, Any],
) -> str:
    requirement = authorial_voice_directive.get("personal_presence_requirement")
    if not isinstance(requirement, str) or not requirement.strip():
        raise TypeError(
            "angle_decision.authorial_voice_directive.personal_presence_requirement "
            "must be present and serialize to a non-empty string."
        )
    try:
        return PERSONAL_PRESENCE_INSTRUCTIONS[requirement]
    except KeyError as exc:
        raise ValueError(
            "angle_decision.authorial_voice_directive.personal_presence_requirement "
            "must be a supported personal-presence policy."
        ) from exc


def _candidate_payload_for_quality_prompt(candidate_payload: Any) -> dict[str, Any]:
    serialized = _serialize_render_value(candidate_payload)
    if not isinstance(serialized, dict):
        raise TypeError("candidate_payload must serialize to a dictionary.")

    return {
        field_name: serialized[field_name]
        for field_name in FINAL_POST_PAYLOAD_PROMPT_FIELDS
        if field_name in serialized
    }


def _selected_evidence_for_quality_prompt(selected_evidence: Any) -> list[dict[str, Any]]:
    serialized = _serialize_render_value(selected_evidence)
    if not isinstance(serialized, list):
        raise TypeError("selected_evidence must serialize to a list.")

    evidence_items = []
    for item in serialized:
        if not isinstance(item, dict):
            raise TypeError("selected_evidence items must serialize to dictionaries.")
        evidence_items.append(
            {
                field_name: item[field_name]
                for field_name in SELECTED_EVIDENCE_PROMPT_FIELDS
                if field_name in item
            }
        )
    return evidence_items


def _selected_evidence_ids_for_repair_prompt(selected_evidence: Any) -> set[str]:
    return {
        item["evidence_id"]
        for item in _selected_evidence_for_quality_prompt(selected_evidence)
        if "evidence_id" in item
    }


def _post_brief_for_repair_prompt(
    post_brief: Any,
    selected_evidence_ids: set[str],
) -> dict[str, Any]:
    serialized = _serialize_render_value(post_brief)
    if not isinstance(serialized, dict):
        raise TypeError("post_brief must serialize to a dictionary.")

    prompt_brief = {
        field_name: serialized[field_name]
        for field_name in REPAIR_POST_BRIEF_PROMPT_FIELDS
        if field_name in serialized and field_name != "evidence_to_use"
    }
    if "evidence_to_use" in serialized:
        prompt_brief["evidence_to_use"] = _selected_evidence_to_use_for_repair_prompt(
            serialized["evidence_to_use"],
            selected_evidence_ids,
        )
    return prompt_brief


def _selected_evidence_to_use_for_repair_prompt(
    evidence_to_use: Any,
    selected_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(evidence_to_use, list):
        raise TypeError("post_brief.evidence_to_use must serialize to a list.")

    evidence_items = []
    for item in evidence_to_use:
        if not isinstance(item, dict):
            raise TypeError(
                "post_brief.evidence_to_use items must serialize to dictionaries."
            )
        if item.get("evidence_id") not in selected_evidence_ids:
            continue
        evidence_items.append(
            {
                field_name: item[field_name]
                for field_name in SELECTED_EVIDENCE_PROMPT_FIELDS
                if field_name in item
            }
        )
    return evidence_items


def _angle_decision_for_repair_prompt(
    angle_decision: Any,
    selected_evidence_ids: set[str],
) -> dict[str, Any]:
    serialized = _serialize_render_value(angle_decision)
    if not isinstance(serialized, dict):
        raise TypeError("angle_decision must serialize to a dictionary.")

    prompt_decision = {
        field_name: serialized[field_name]
        for field_name in REPAIR_ANGLE_DECISION_PROMPT_FIELDS
        if field_name in serialized and field_name != "supporting_evidence_ids"
    }
    if "supporting_evidence_ids" in serialized:
        prompt_decision["supporting_evidence_ids"] = [
            evidence_id
            for evidence_id in serialized["supporting_evidence_ids"]
            if evidence_id in selected_evidence_ids
        ]
    return prompt_decision


def _repair_dict(value: Any, label: str) -> dict[str, Any]:
    serialized = _serialize_render_value(value)
    if not isinstance(serialized, dict):
        raise TypeError(f"{label} must serialize to a dictionary.")
    return serialized


def _build_input_text(variables: dict[str, str]) -> str:
    sections = [
        ("POST_BRIEF_JSON", variables["post_brief_json"]),
        ("ANGLE_DECISION_JSON", variables["angle_decision_json"]),
        (
            "AUTHORIAL_VOICE_DIRECTIVE_JSON",
            variables["authorial_voice_directive_json"],
        ),
        (
            "PERSONAL_PRESENCE_INSTRUCTION",
            variables["personal_presence_instruction"],
        ),
        ("SELECTED_EVIDENCE_JSON", variables["selected_evidence_json"]),
        (
            "FINAL_POST_PAYLOAD_CONSTRAINTS_JSON",
            variables["final_post_payload_constraints_json"],
        ),
        ("CANDIDATE_WRITER_INPUT_JSON", variables["candidate_writer_input_json"]),
    ]
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)


def _build_repair_writer_input_text(variables: dict[str, str]) -> str:
    sections = [
        ("ORIGINAL_CANDIDATE_PAYLOAD_JSON", variables["original_candidate_payload_json"]),
        ("POST_BRIEF_JSON", variables["post_brief_json"]),
        ("ANGLE_DECISION_JSON", variables["angle_decision_json"]),
        (
            "AUTHORIAL_VOICE_DIRECTIVE_JSON",
            variables["authorial_voice_directive_json"],
        ),
        (
            "PERSONAL_PRESENCE_INSTRUCTION",
            variables["personal_presence_instruction"],
        ),
        ("SELECTED_EVIDENCE_JSON", variables["selected_evidence_json"]),
        ("DETERMINISTIC_FINDINGS_JSON", variables["deterministic_findings_json"]),
        ("QUALITY_FINDINGS_JSON", variables["quality_findings_json"]),
        ("REPAIR_INSTRUCTION_JSON", variables["repair_instruction_json"]),
        ("REPAIR_ATTEMPT_JSON", variables["repair_attempt_json"]),
    ]
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)


def _build_quality_evaluator_input_text(variables: dict[str, str]) -> str:
    sections = [
        ("CANDIDATE_PAYLOAD_JSON", variables["candidate_payload_json"]),
        ("POST_BRIEF_JSON", variables["post_brief_json"]),
        ("ANGLE_DECISION_JSON", variables["angle_decision_json"]),
        (
            "AUTHORIAL_VOICE_DIRECTIVE_JSON",
            variables["authorial_voice_directive_json"],
        ),
        ("SELECTED_EVIDENCE_JSON", variables["selected_evidence_json"]),
        ("QUALITY_RUBRIC_JSON", variables["quality_rubric_json"]),
    ]
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)


def _build_semantic_grounding_input_text(variables: dict[str, str]) -> str:
    sections = [
        ("CANDIDATE_PAYLOAD_JSON", variables["candidate_payload_json"]),
        ("POST_BRIEF_JSON", variables["post_brief_json"]),
        ("ANGLE_DECISION_JSON", variables["angle_decision_json"]),
        ("SELECTED_EVIDENCE_JSON", variables["selected_evidence_json"]),
        (
            "SEMANTIC_GROUNDING_RULES_JSON",
            variables["semantic_grounding_rules_json"],
        ),
    ]
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)
