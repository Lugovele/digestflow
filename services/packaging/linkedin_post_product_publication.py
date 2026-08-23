"""Production publication boundary for accepted clean LinkedIn PostFlow output.

This module connects the existing deterministic PostFlow attempt executor to
the existing ContentPackage result-page model. It does not implement a second
packaging pipeline, repair loop, prompt executor, provider adapter, or UI path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings

from apps.digests.models import Digest
from apps.packaging.models import ContentPackage
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    FinalPostAttemptOutcome,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FinalPostAttemptRequest,
)
from services.packaging.linkedin_post_final_post_attempt_execution import (
    execute_final_post_standalone_attempt,
)
from services.packaging.linkedin_post_flow_contracts import FinalPostAttemptHistory
from services.packaging.linkedin_post_flow_input_builders import (
    build_candidate_writer_input,
)
from services.packaging.linkedin_post_pipeline import (
    build_angle_decision_from_contextual_evidence_pack,
    build_article_evidence_pack_from_pipeline_input,
    build_contextual_evidence_pack_from_article_evidence_pack,
    build_pipeline_input_from_digest,
    build_post_brief_from_angle_decision,
)
from services.packaging.linkedin_post_prompt_registry import (
    PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF,
    PROMPT_FINAL_POST_QUALITY_EVALUATOR,
    get_prompt_contract,
    prompt_contract_to_prompt_metadata,
)
from services.packaging.linkedin_post_prompt_renderers import (
    render_candidate_writer_prompt_input,
)
from services.packaging.linkedin_post_publication_persistence import (
    persist_accepted_linkedin_publication_from_outcome,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    get_quality_evaluator_rubric_payload,
)


SEMANTIC_GROUNDING_PROMPT_PATH = (
    "prompts/linkedin/final_post_semantic_grounding_evaluator.txt"
)


@dataclass(frozen=True)
class FinalPostProviderClients:
    """Optional provider-client injection for tests and local controlled runs."""

    candidate_writer_client: Any | None = None
    semantic_grounding_client: Any | None = None
    quality_evaluator_client: Any | None = None


def generate_accepted_linkedin_content_package_for_digest(
    digest: Digest,
    *,
    author_profile: dict[str, Any] | None = None,
    provider_clients: FinalPostProviderClients | None = None,
) -> tuple[ContentPackage, dict[str, Any]]:
    """Run the clean final-post attempt and persist only accepted output.

    The clean accepted result is authoritative for the result page. Any existing
    package row for this digest is cleared before the attempt so a failed or
    non-accepted clean flow cannot leave stale legacy content visible.
    """

    ContentPackage.objects.filter(digest=digest).delete()

    prepared = _build_product_attempt_request(
        digest=digest,
        author_profile=author_profile,
    )
    clients = provider_clients or FinalPostProviderClients()
    result = execute_final_post_standalone_attempt(
        prepared["request"],
        post_brief=prepared["post_brief"],
        angle_decision=prepared["angle_decision"],
        selected_evidence_ids=prepared["selected_evidence_ids"],
        candidate_writer_client=clients.candidate_writer_client,
        semantic_grounding_client=clients.semantic_grounding_client,
        quality_evaluator_client=clients.quality_evaluator_client,
    )

    attempt_outcome = getattr(result, "final_attempt_outcome", None)
    if not isinstance(attempt_outcome, FinalPostAttemptOutcome):
        raise RuntimeError(_clean_flow_failure_message(result))
    if attempt_outcome.outcome != OUTCOME_ACCEPTED:
        raise RuntimeError(f"Clean final-post flow did not accept: {attempt_outcome.outcome}")

    content_package = persist_accepted_linkedin_publication_from_outcome(
        digest=digest,
        attempt_outcome=attempt_outcome,
        deterministic_gate_passed=_gate_passed(result),
        semantic_grounding_passed=_semantic_grounding_passed(result),
        quality_passed=_quality_passed(result),
    )
    if content_package is None:
        raise RuntimeError("Clean final-post flow did not produce a publication package.")

    return content_package, _packaging_debug_from_result(result)


def _build_product_attempt_request(
    *,
    digest: Digest,
    author_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    pipeline_input = build_pipeline_input_from_digest(
        digest,
        author_profile=author_profile,
    )
    article_evidence_pack = build_article_evidence_pack_from_pipeline_input(
        pipeline_input
    )
    contextual_evidence_pack = build_contextual_evidence_pack_from_article_evidence_pack(
        article_evidence_pack
    )
    angle_decision = build_angle_decision_from_contextual_evidence_pack(
        contextual_evidence_pack
    )
    post_brief = build_post_brief_from_angle_decision(
        contextual_evidence_pack,
        angle_decision,
    )
    selected_evidence_ids = tuple(
        item.evidence_id for item in post_brief.evidence_to_use
    )

    candidate_contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)
    quality_contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)
    candidate_input = build_candidate_writer_input(
        post_brief,
        angle_decision,
        prompt_metadata=prompt_contract_to_prompt_metadata(candidate_contract),
    )
    candidate_render = render_candidate_writer_prompt_input(candidate_input)

    request = FinalPostAttemptRequest(
        candidate_writer_render=candidate_render,
        candidate_writer_prompt_text=_read_prompt_text(candidate_contract.prompt_path),
        quality_rubric=get_quality_evaluator_rubric_payload(),
        quality_evaluator_prompt_text=_read_prompt_text(quality_contract.prompt_path),
        attempt_index=0,
        max_attempts=1,
        attempt_history=FinalPostAttemptHistory(attempts=[]),
        semantic_grounding_prompt_text=_read_prompt_text(
            SEMANTIC_GROUNDING_PROMPT_PATH
        ),
        execution_metadata={
            "production_runtime": True,
            "digest_id": digest.id,
            "clean_postflow_publication": True,
        },
    )
    return {
        "request": request,
        "post_brief": post_brief,
        "angle_decision": angle_decision,
        "selected_evidence_ids": selected_evidence_ids,
    }


def _read_prompt_text(prompt_path: str) -> str:
    path = Path(settings.BASE_DIR) / prompt_path
    if not path.exists():
        raise ValueError(f"prompt file does not exist: {prompt_path}")
    prompt_text = path.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise ValueError(f"prompt file is empty: {prompt_path}")
    return prompt_text


def _clean_flow_failure_message(result: Any) -> str:
    failure_code = getattr(result, "failure_code", None)
    failure_stage = getattr(result, "failure_stage", None)
    failure_message = getattr(result, "failure_message", "")
    if failure_code or failure_stage or failure_message:
        return (
            "Clean final-post flow failed before acceptance"
            f" ({failure_stage or 'unknown_stage'}"
            f"/{failure_code or 'unknown_code'}): {failure_message}"
        )
    return "Clean final-post flow finished without an accepted outcome."


def _gate_passed(result: Any) -> bool:
    gate = getattr(result, "deterministic_gate_output", None)
    diagnostics = getattr(gate, "diagnostics", None)
    return (
        getattr(gate, "validation_passed", False) is True
        and getattr(diagnostics, "deterministic_checks_passed", False) is True
        and getattr(diagnostics, "system_linkedin_ready", False) is True
    )


def _semantic_grounding_passed(result: Any) -> bool:
    semantic_state = getattr(result, "semantic_grounding_state", None)
    grounding_review = getattr(semantic_state, "grounding_review", None)
    return getattr(grounding_review, "passed", False) is True


def _quality_passed(result: Any) -> bool:
    quality_state = getattr(result, "quality_evaluation_state", None)
    quality_review = getattr(quality_state, "quality_review", None)
    return isinstance(quality_review, dict) and quality_review.get("pass") is True


def _packaging_debug_from_result(result: Any) -> dict[str, Any]:
    return {
        "provider": "clean_postflow",
        "is_mock": False,
        "fallback_reason": "",
        "tokens": _combined_token_usage(result),
        "estimated_cost_usd": None,
        "final_post_flow": {
            "status": "accepted",
            "outcome": getattr(
                getattr(result, "final_attempt_outcome", None),
                "outcome",
                None,
            ),
            "completed_stage": getattr(result, "completed_stage", None),
            "failure_stage": getattr(result, "failure_stage", None),
            "failure_code": getattr(result, "failure_code", None),
            "candidate_writer_invocation_count": getattr(
                result,
                "candidate_writer_invocation_count",
                0,
            ),
            "semantic_grounding_invocation_count": getattr(
                result,
                "semantic_grounding_invocation_count",
                0,
            ),
            "quality_evaluator_invocation_count": getattr(
                result,
                "quality_evaluator_invocation_count",
                0,
            ),
            "repair_invocation_count": getattr(result, "repair_invocation_count", 0),
        },
    }


def _combined_token_usage(result: Any) -> dict[str, int] | None:
    usages = [
        getattr(getattr(result, "candidate_writer_raw_response", None), "usage", None),
        getattr(getattr(result, "semantic_grounding_raw_response", None), "usage", None),
        getattr(getattr(result, "quality_evaluator_raw_response", None), "usage", None),
    ]
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    saw_usage = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        saw_usage = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return totals if saw_usage else None
