"""Pure deterministic gate for final LinkedIn post payload candidates."""
from __future__ import annotations

from typing import Any

from services.packaging.linkedin_final_post_diagnostics import (
    diagnose_final_post_payload,
)
from services.packaging.linkedin_post_flow_handoffs import (
    CandidateWriterOutput,
    DeterministicGateOutput,
)
from services.packaging.linkedin_post_pipeline import (
    FinalPostPayload,
    LinkedInPostPipelineContractError,
    validate_final_post_payload,
)


def run_final_post_deterministic_gate(
    candidate: dict | CandidateWriterOutput,
    *,
    selected_evidence_ids: list[str] | tuple[str, ...],
) -> DeterministicGateOutput:
    payload = _extract_payload(candidate)
    validation_passed, validation_error = _validate_payload(payload)
    diagnostics = diagnose_final_post_payload(
        payload,
        selected_evidence_ids=list(selected_evidence_ids),
        schema_validation_passed=validation_passed,
        schema_validation_error=validation_error,
    )

    return DeterministicGateOutput(
        payload=payload,
        validation_passed=validation_passed,
        validation_error=validation_error,
        diagnostics=diagnostics,
        selected_evidence_ids=tuple(selected_evidence_ids),
    )


def _extract_payload(candidate: dict | CandidateWriterOutput) -> dict:
    if isinstance(candidate, CandidateWriterOutput):
        return candidate.payload
    return candidate


def _validate_payload(payload: dict) -> tuple[bool, str]:
    try:
        validate_final_post_payload(_final_post_payload_from_dict(payload))
    except (KeyError, TypeError, LinkedInPostPipelineContractError) as exc:
        return False, str(exc)
    return True, ""


def _final_post_payload_from_dict(payload: dict[str, Any]) -> FinalPostPayload:
    return FinalPostPayload(
        post_text=payload["post_text"],
        hook_variants=payload["hook_variants"],
        cta_variants=payload["cta_variants"],
        hashtags=payload["hashtags"],
        quality_checks=payload["quality_checks"],
        carousel_outline=payload.get("carousel_outline", []),
    )
