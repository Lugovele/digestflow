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
from services.packaging.linkedin_post_quality_rubric_contract import (
    QualityEvaluatorRubricPayload,
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


def render_candidate_writer_prompt_input(
    candidate_input: CandidateWriterInput,
) -> CandidateWriterPromptRender:
    candidate_input_dict = candidate_input.to_dict()
    variables = {
        "post_brief_json": _stable_json(candidate_input_dict["post_brief"]),
        "angle_decision_json": _stable_json(candidate_input_dict["angle_decision"]),
        "selected_evidence_json": _stable_json(
            candidate_input_dict["selected_evidence"]
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


def _build_input_text(variables: dict[str, str]) -> str:
    sections = [
        ("POST_BRIEF_JSON", variables["post_brief_json"]),
        ("ANGLE_DECISION_JSON", variables["angle_decision_json"]),
        ("SELECTED_EVIDENCE_JSON", variables["selected_evidence_json"]),
        ("CANDIDATE_WRITER_INPUT_JSON", variables["candidate_writer_input_json"]),
    ]
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)


def _build_quality_evaluator_input_text(variables: dict[str, str]) -> str:
    sections = [
        ("CANDIDATE_PAYLOAD_JSON", variables["candidate_payload_json"]),
        ("POST_BRIEF_JSON", variables["post_brief_json"]),
        ("ANGLE_DECISION_JSON", variables["angle_decision_json"]),
        ("SELECTED_EVIDENCE_JSON", variables["selected_evidence_json"]),
        ("QUALITY_RUBRIC_JSON", variables["quality_rubric_json"]),
    ]
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)
