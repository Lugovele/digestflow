"""Pure input builders for future final LinkedIn post flow agents."""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from typing import Any

from services.packaging.linkedin_post_editorial_boundary import PromptMetadata


@dataclass(frozen=True)
class CandidateWriterInput:
    post_brief: object | dict
    angle_decision: object | dict
    selected_evidence: tuple[dict, ...]
    prompt_metadata: PromptMetadata | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_brief": _serialize_input_value(self.post_brief),
            "angle_decision": _serialize_input_value(self.angle_decision),
            "selected_evidence": _serialize_input_value(self.selected_evidence),
            "prompt_metadata": (
                self.prompt_metadata.to_dict() if self.prompt_metadata else None
            ),
        }


def build_candidate_writer_input(
    post_brief: object | dict,
    angle_decision: object | dict,
    prompt_metadata: PromptMetadata | None = None,
) -> CandidateWriterInput:
    return CandidateWriterInput(
        post_brief=post_brief,
        angle_decision=angle_decision,
        selected_evidence=tuple(
            _normalize_selected_evidence_item(item)
            for item in _get_required_value(post_brief, "evidence_to_use")
        ),
        prompt_metadata=prompt_metadata,
    )


def _normalize_selected_evidence_item(item: object | dict) -> dict[str, str]:
    return {
        "evidence_id": _get_required_value(item, "evidence_id"),
        "evidence_text": _get_required_value(item, "evidence_text"),
        "role_in_post": _get_required_value(item, "role_in_post"),
    }


def _get_required_value(value: object | dict, field_name: str) -> Any:
    if isinstance(value, dict):
        return value[field_name]
    return getattr(value, field_name)


def _serialize_input_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return _serialize_input_value(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize_input_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_input_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_input_value(item) for item in value]
    return value
