"""Pure prompt input renderers for future final LinkedIn post flow agents."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from services.packaging.linkedin_post_flow_input_builders import CandidateWriterInput


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


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _build_input_text(variables: dict[str, str]) -> str:
    sections = [
        ("POST_BRIEF_JSON", variables["post_brief_json"]),
        ("ANGLE_DECISION_JSON", variables["angle_decision_json"]),
        ("SELECTED_EVIDENCE_JSON", variables["selected_evidence_json"]),
        ("CANDIDATE_WRITER_INPUT_JSON", variables["candidate_writer_input_json"]),
    ]
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)
