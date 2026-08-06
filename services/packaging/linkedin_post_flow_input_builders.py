"""Pure input builders for future final LinkedIn post flow agents."""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
import copy
from typing import Any

from services.packaging.linkedin_post_editorial_boundary import (
    PostEditorialInput,
    PostGenerationMetadata,
    PromptMetadata,
)
from services.packaging.linkedin_post_flow_handoffs import (
    CandidateWriterOutput,
    DeterministicGateOutput,
)
from services.packaging.linkedin_post_pipeline import (
    AuthorialVoiceDirective,
    validate_authorial_voice_directive,
)


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
    _validate_angle_decision_authorial_voice_directive(angle_decision)
    return CandidateWriterInput(
        post_brief=post_brief,
        angle_decision=angle_decision,
        selected_evidence=tuple(
            _normalize_selected_evidence_item(item)
            for item in _get_required_value(post_brief, "evidence_to_use")
        ),
        prompt_metadata=prompt_metadata,
    )


def build_post_editorial_input(
    *,
    post_brief: object | dict,
    angle_decision: object | dict,
    candidate_output: CandidateWriterOutput,
    gate_output: DeterministicGateOutput,
    generation_metadata: PostGenerationMetadata | None = None,
    prompt_metadata: PromptMetadata | None = None,
) -> PostEditorialInput:
    _validate_angle_decision_authorial_voice_directive(angle_decision)
    _require_passing_gate(gate_output)
    _require_matching_candidate_payload(candidate_output, gate_output)

    selected_evidence = _selected_evidence_from_post_brief(post_brief)
    selected_evidence_ids = tuple(item["evidence_id"] for item in selected_evidence)
    if selected_evidence_ids != gate_output.selected_evidence_ids:
        raise ValueError(
            "PostBrief.evidence_to_use evidence IDs must match "
            "DeterministicGateOutput.selected_evidence_ids."
        )

    resolved_generation_metadata = _build_generation_metadata(
        candidate_output,
        generation_metadata,
    )
    resolved_prompt_metadata = _build_prompt_metadata(
        candidate_output,
        prompt_metadata,
    )

    return PostEditorialInput(
        candidate_payload=copy.deepcopy(gate_output.payload),
        post_brief=_serialize_input_value(post_brief),
        angle_decision=_serialize_input_value(angle_decision),
        selected_evidence=[copy.deepcopy(item) for item in selected_evidence],
        final_payload_validation_passed=gate_output.validation_passed,
        final_payload_validation_error=gate_output.validation_error,
        diagnostics=gate_output.diagnostics,
        repair_reasons=list(gate_output.diagnostics.repair_reasons),
        generation_metadata=resolved_generation_metadata,
        prompt_metadata=resolved_prompt_metadata,
    )



def _validate_angle_decision_authorial_voice_directive(
    angle_decision: object | dict,
) -> None:
    directive = _get_required_value(
        angle_decision,
        "authorial_voice_directive",
        "AngleDecision",
    )
    if isinstance(directive, AuthorialVoiceDirective):
        validate_authorial_voice_directive(directive)
        return
    if not isinstance(directive, dict):
        validate_authorial_voice_directive(directive)
        return
    forbidden_author_claims = directive.get("forbidden_author_claims")
    if isinstance(forbidden_author_claims, (list, tuple)):
        normalized_forbidden_author_claims = tuple(forbidden_author_claims)
    else:
        normalized_forbidden_author_claims = forbidden_author_claims
    validate_authorial_voice_directive(
        AuthorialVoiceDirective(
            authorial_observation=directive.get("authorial_observation"),
            rejected_reading=directive.get("rejected_reading"),
            why_distinction_matters=directive.get("why_distinction_matters"),
            personal_presence_requirement=directive.get("personal_presence_requirement"),
            first_person_policy=directive.get("first_person_policy"),
            forbidden_author_claims=normalized_forbidden_author_claims,
        )
    )

def _normalize_selected_evidence_item(item: object | dict) -> dict[str, str]:
    normalized = {
        "evidence_id": _get_required_value(item, "evidence_id", "PostBrief.evidence_to_use"),
        "evidence_text": _get_required_value(item, "evidence_text", "PostBrief.evidence_to_use"),
        "role_in_post": _get_required_value(item, "role_in_post", "PostBrief.evidence_to_use"),
    }
    for field_name, field_value in normalized.items():
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"PostBrief.evidence_to_use.{field_name} must be a non-empty string."
            )
    return normalized


def _selected_evidence_from_post_brief(post_brief: object | dict) -> tuple[dict[str, str], ...]:
    evidence_to_use = _get_field(post_brief, "evidence_to_use")
    if not isinstance(evidence_to_use, (list, tuple)) or not evidence_to_use:
        raise ValueError("PostBrief.evidence_to_use must be a non-empty list or tuple.")
    selected_evidence = tuple(
        _normalize_selected_evidence_item(item)
        for item in evidence_to_use
    )
    evidence_ids = [item["evidence_id"] for item in selected_evidence]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("PostBrief.evidence_to_use evidence IDs must be unique.")
    return selected_evidence


def _get_required_value(value: object | dict, field_name: str, owner_name: str = "value") -> Any:
    field_value = _get_field(value, field_name)
    if field_value is None:
        raise ValueError(f"{owner_name}.{field_name} must be present.")
    return field_value


def _get_field(value: object | dict, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _require_passing_gate(gate_output: DeterministicGateOutput) -> None:
    if gate_output.validation_passed is not True:
        raise ValueError("DeterministicGateOutput.validation_passed must be True.")
    if gate_output.diagnostics.schema_validation_passed is not True:
        raise ValueError("Deterministic diagnostics schema validation must pass.")
    if gate_output.diagnostics.system_linkedin_ready is not True:
        raise ValueError("Deterministic diagnostics system_linkedin_ready must be True.")
    if gate_output.diagnostics.repair_reasons:
        raise ValueError("Deterministic diagnostics repair_reasons must be empty.")


def _require_matching_candidate_payload(
    candidate_output: CandidateWriterOutput,
    gate_output: DeterministicGateOutput,
) -> None:
    if candidate_output.payload != gate_output.payload:
        raise ValueError(
            "CandidateWriterOutput.payload must match DeterministicGateOutput.payload."
        )


def _build_generation_metadata(
    candidate_output: CandidateWriterOutput,
    supplied_metadata: PostGenerationMetadata | None,
) -> PostGenerationMetadata:
    provider = _require_metadata_string(candidate_output.provider, "provider")
    model = _require_metadata_string(candidate_output.model, "model")
    if supplied_metadata is not None:
        if supplied_metadata.provider != provider or supplied_metadata.model != model:
            raise ValueError(
                "PostGenerationMetadata provider/model must match CandidateWriterOutput."
            )
        token_usage = _resolve_generation_metadata_value(
            candidate_output.token_usage,
            supplied_metadata.token_usage,
            "token_usage",
        )
        cost_metadata = _resolve_generation_metadata_value(
            candidate_output.cost_metadata,
            supplied_metadata.cost_metadata,
            "cost_metadata",
        )
        return PostGenerationMetadata(
            provider=provider,
            model=model,
            run_id=supplied_metadata.run_id,
            created_at=supplied_metadata.created_at,
            token_usage=token_usage,
            cost_metadata=cost_metadata,
        )

    return PostGenerationMetadata(
        provider=provider,
        model=model,
        run_id=None,
        created_at=None,
        token_usage=copy.deepcopy(candidate_output.token_usage),
        cost_metadata=copy.deepcopy(candidate_output.cost_metadata),
    )


def _resolve_generation_metadata_value(
    candidate_value: dict | None,
    supplied_value: dict | None,
    field_name: str,
) -> dict | None:
    if candidate_value is not None and supplied_value is not None:
        if candidate_value != supplied_value:
            raise ValueError(
                f"CandidateWriterOutput {field_name} conflicts with "
                f"PostGenerationMetadata.{field_name}."
            )
        return copy.deepcopy(candidate_value)
    if candidate_value is not None:
        return copy.deepcopy(candidate_value)
    return copy.deepcopy(supplied_value)


def _build_prompt_metadata(
    candidate_output: CandidateWriterOutput,
    supplied_metadata: PromptMetadata | None,
) -> PromptMetadata:
    prompt_name = _require_metadata_string(candidate_output.prompt_name, "prompt_name")
    prompt_version = _require_metadata_string(
        candidate_output.prompt_version,
        "prompt_version",
    )
    if supplied_metadata is not None:
        if (
            supplied_metadata.prompt_name != prompt_name
            or supplied_metadata.prompt_version != prompt_version
        ):
            raise ValueError(
                "PromptMetadata name/version must match CandidateWriterOutput."
            )
        return PromptMetadata(
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_path=supplied_metadata.prompt_path,
        )
    return PromptMetadata(
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        prompt_path=None,
    )


def _require_metadata_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CandidateWriterOutput.{field_name} must be a non-empty string.")
    return value


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
