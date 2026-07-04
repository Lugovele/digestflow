"""Boundary objects for future LinkedIn post editorial prompt flow."""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from typing import Any

from services.packaging.linkedin_final_post_diagnostics import FinalPostDiagnostics


@dataclass(frozen=True)
class PostGenerationMetadata:
    provider: str
    model: str
    run_id: str | None
    created_at: str | None
    token_usage: dict | None
    cost_metadata: dict | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "token_usage": _serialize_boundary_value(self.token_usage),
            "cost_metadata": _serialize_boundary_value(self.cost_metadata),
        }


@dataclass(frozen=True)
class PromptMetadata:
    prompt_name: str
    prompt_version: str
    prompt_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_path": self.prompt_path,
        }


@dataclass(frozen=True)
class PostEditorialInput:
    candidate_payload: dict
    post_brief: object | dict
    angle_decision: object | dict
    selected_evidence: list[dict]
    final_payload_validation_passed: bool
    final_payload_validation_error: str
    diagnostics: FinalPostDiagnostics
    repair_reasons: list[str]
    generation_metadata: PostGenerationMetadata
    prompt_metadata: PromptMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_payload": _serialize_boundary_value(self.candidate_payload),
            "post_brief": _serialize_boundary_value(self.post_brief),
            "angle_decision": _serialize_boundary_value(self.angle_decision),
            "selected_evidence": _serialize_boundary_value(self.selected_evidence),
            "final_payload_validation_passed": self.final_payload_validation_passed,
            "final_payload_validation_error": self.final_payload_validation_error,
            "diagnostics": self.diagnostics.to_dict(),
            "repair_reasons": list(self.repair_reasons),
            "generation_metadata": self.generation_metadata.to_dict(),
            "prompt_metadata": self.prompt_metadata.to_dict(),
        }


def _serialize_boundary_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return _serialize_boundary_value(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize_boundary_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_boundary_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_boundary_value(item) for item in value]
    return value
