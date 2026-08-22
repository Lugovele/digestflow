"""Contracts for semantic grounding review of a final LinkedIn post candidate.

This module defines claim-fidelity structures only. It does not execute
providers, render prompts, score editorial quality, route attempts, repair
text, persist data, or connect to runtime packaging.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


GROUNDING_STATUS_PASS = "pass"
GROUNDING_STATUS_FAIL = "fail"
GROUNDING_STATUS_NOT_READY = "not_ready"
GROUNDING_STATUS_NEEDS_HUMAN_REVIEW = "needs_human_review"

SEMANTIC_GROUNDING_STATUSES = (
    GROUNDING_STATUS_PASS,
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_NOT_READY,
    GROUNDING_STATUS_NEEDS_HUMAN_REVIEW,
)

CLAIM_TYPE_METRIC_OR_DATE = "metric_or_date"
CLAIM_TYPE_ATTRIBUTED_SOURCE_CLAIM = "attributed_source_claim"
CLAIM_TYPE_MARKET_CONDITION = "market_condition"
CLAIM_TYPE_FORECAST_OR_PROJECTION = "forecast_or_projection"
CLAIM_TYPE_CAUSAL_CLAIM = "causal_claim"
CLAIM_TYPE_COMPARISON_OR_CONTRAST = "comparison_or_contrast"
CLAIM_TYPE_AUTHOR_INTERPRETATION = "author_interpretation"
CLAIM_TYPE_RECOMMENDATION = "recommendation"
CLAIM_TYPE_PERSONAL_EXPERIENCE_OR_CASE = "personal_experience_or_case"
CLAIM_TYPE_CTA_OR_RHETORICAL = "cta_or_rhetorical"

SEMANTIC_GROUNDING_CLAIM_TYPES = (
    CLAIM_TYPE_METRIC_OR_DATE,
    CLAIM_TYPE_ATTRIBUTED_SOURCE_CLAIM,
    CLAIM_TYPE_MARKET_CONDITION,
    CLAIM_TYPE_FORECAST_OR_PROJECTION,
    CLAIM_TYPE_CAUSAL_CLAIM,
    CLAIM_TYPE_COMPARISON_OR_CONTRAST,
    CLAIM_TYPE_AUTHOR_INTERPRETATION,
    CLAIM_TYPE_RECOMMENDATION,
    CLAIM_TYPE_PERSONAL_EXPERIENCE_OR_CASE,
    CLAIM_TYPE_CTA_OR_RHETORICAL,
)

SUPPORT_STATUS_NOT_CLAIM = "not_claim"
SUPPORT_STATUS_SUPPORTED = "supported"
SUPPORT_STATUS_SUPPORTED_WITH_REQUIRED_QUALIFICATION = (
    "supported_with_required_qualification"
)
SUPPORT_STATUS_PARTIALLY_SUPPORTED = "partially_supported"
SUPPORT_STATUS_MISSING_REQUIRED_QUALIFICATION = "missing_required_qualification"
SUPPORT_STATUS_CAUSAL_OVERREACH = "causal_overreach"
SUPPORT_STATUS_UNSUPPORTED = "unsupported"
SUPPORT_STATUS_CONTRADICTED = "contradicted"
SUPPORT_STATUS_NOT_EVALUABLE = "not_evaluable"

SEMANTIC_GROUNDING_SUPPORT_STATUSES = (
    SUPPORT_STATUS_NOT_CLAIM,
    SUPPORT_STATUS_SUPPORTED,
    SUPPORT_STATUS_SUPPORTED_WITH_REQUIRED_QUALIFICATION,
    SUPPORT_STATUS_PARTIALLY_SUPPORTED,
    SUPPORT_STATUS_MISSING_REQUIRED_QUALIFICATION,
    SUPPORT_STATUS_CAUSAL_OVERREACH,
    SUPPORT_STATUS_UNSUPPORTED,
    SUPPORT_STATUS_CONTRADICTED,
    SUPPORT_STATUS_NOT_EVALUABLE,
)

SUPPORT_STATUS_PRECEDENCE = {
    SUPPORT_STATUS_CONTRADICTED: 90,
    SUPPORT_STATUS_CAUSAL_OVERREACH: 80,
    SUPPORT_STATUS_MISSING_REQUIRED_QUALIFICATION: 70,
    SUPPORT_STATUS_UNSUPPORTED: 60,
    SUPPORT_STATUS_NOT_EVALUABLE: 50,
    SUPPORT_STATUS_PARTIALLY_SUPPORTED: 40,
    SUPPORT_STATUS_SUPPORTED_WITH_REQUIRED_QUALIFICATION: 20,
    SUPPORT_STATUS_SUPPORTED: 10,
    SUPPORT_STATUS_NOT_CLAIM: 0,
}

SEVERITY_INFO = "info"
SEVERITY_MINOR = "minor"
SEVERITY_MAJOR = "major"
SEVERITY_BLOCKING = "blocking"

SEMANTIC_GROUNDING_SEVERITIES = (
    SEVERITY_INFO,
    SEVERITY_MINOR,
    SEVERITY_MAJOR,
    SEVERITY_BLOCKING,
)

SEMANTIC_GROUNDING_STATUS_SEVERITY_MATRIX = {
    SUPPORT_STATUS_NOT_CLAIM: (SEVERITY_INFO,),
    SUPPORT_STATUS_SUPPORTED: (SEVERITY_INFO,),
    SUPPORT_STATUS_SUPPORTED_WITH_REQUIRED_QUALIFICATION: (SEVERITY_INFO,),
    SUPPORT_STATUS_PARTIALLY_SUPPORTED: (SEVERITY_MINOR, SEVERITY_MAJOR),
    SUPPORT_STATUS_MISSING_REQUIRED_QUALIFICATION: (
        SEVERITY_MAJOR,
        SEVERITY_BLOCKING,
    ),
    SUPPORT_STATUS_CAUSAL_OVERREACH: (SEVERITY_MAJOR, SEVERITY_BLOCKING),
    SUPPORT_STATUS_UNSUPPORTED: (SEVERITY_MAJOR, SEVERITY_BLOCKING),
    SUPPORT_STATUS_CONTRADICTED: (SEVERITY_BLOCKING,),
    SUPPORT_STATUS_NOT_EVALUABLE: (SEVERITY_MAJOR, SEVERITY_BLOCKING),
}

SEMANTIC_GROUNDING_SUPPORTED_STATUSES = (
    SUPPORT_STATUS_SUPPORTED,
    SUPPORT_STATUS_SUPPORTED_WITH_REQUIRED_QUALIFICATION,
)
SEMANTIC_GROUNDING_ADVISORY_STATUSES = (
    SUPPORT_STATUS_PARTIALLY_SUPPORTED,
)
SEMANTIC_GROUNDING_BLOCKING_ELIGIBLE_STATUSES = (
    SUPPORT_STATUS_PARTIALLY_SUPPORTED,
    SUPPORT_STATUS_MISSING_REQUIRED_QUALIFICATION,
    SUPPORT_STATUS_CAUSAL_OVERREACH,
    SUPPORT_STATUS_UNSUPPORTED,
    SUPPORT_STATUS_CONTRADICTED,
    SUPPORT_STATUS_NOT_EVALUABLE,
)
SEMANTIC_GROUNDING_BLOCKING_SEVERITIES = (
    SEVERITY_MAJOR,
    SEVERITY_BLOCKING,
)
SEMANTIC_GROUNDING_NOT_A_CLAIM_STATUSES = (
    SUPPORT_STATUS_NOT_CLAIM,
)


def is_valid_status_severity_pair(support_status: str, severity: str) -> bool:
    return severity in SEMANTIC_GROUNDING_STATUS_SEVERITY_MATRIX.get(
        support_status,
        (),
    )


def is_blocking_claim_state(support_status: str, severity: str) -> bool:
    return (
        support_status in SEMANTIC_GROUNDING_BLOCKING_ELIGIBLE_STATUSES
        and severity in SEMANTIC_GROUNDING_BLOCKING_SEVERITIES
    )


def build_semantic_grounding_prompt_rules() -> dict[str, Any]:
    """Return the canonical model-facing semantic-grounding decision rules."""

    return {
        "status_severity_matrix": {
            status: list(severities)
            for status, severities in SEMANTIC_GROUNDING_STATUS_SEVERITY_MATRIX.items()
        },
        "supported_statuses": list(SEMANTIC_GROUNDING_SUPPORTED_STATUSES),
        "advisory_statuses": list(SEMANTIC_GROUNDING_ADVISORY_STATUSES),
        "blocking_eligible_statuses": list(
            SEMANTIC_GROUNDING_BLOCKING_ELIGIBLE_STATUSES
        ),
        "blocking_severities": list(SEMANTIC_GROUNDING_BLOCKING_SEVERITIES),
        "not_a_claim_statuses": list(SEMANTIC_GROUNDING_NOT_A_CLAIM_STATUSES),
        "decision_rules": [
            "Use only status/severity pairs from status_severity_matrix.",
            "supported and supported_with_required_qualification must use info severity.",
            "partially_supported with minor severity is advisory and does not fail grounding.",
            "partially_supported with major severity is a blocking claim.",
            "missing_required_qualification, causal_overreach, unsupported, contradicted, and not_evaluable are blocking when paired with a blocking severity.",
            "failed_claim_ids must exactly match blocking claim IDs in claim_reviews order.",
            "pass can be true only when there are no blocking claims, no automatic_fail_reason, no human-review requirement, no repairable flag, and no repair instructions.",
            "automatic_fail_reason requires at least one blocking claim.",
            "repairable can be true only for failed, non-human-review grounding results with blocking claims.",
            "Every repairable blocking claim requires a non-empty repair_hint and a top-level repair instruction beginning with '<claim_id>:'.",
        ],
        "qualification_invariants": [
            "projected remains projected",
            "likely remains attributed likelihood",
            "may remains possibility",
            "risk remains risk",
            "analysis remains attributed analysis",
        ],
        "causal_fidelity": [
            "Do not turn coexistence into cause.",
            "Do not turn forecasts into outcomes.",
            "Do not turn positioning or risk into stability, recovery, optimism, or growth.",
        ],
        "selected_evidence_only": True,
    }


@dataclass(frozen=True)
class SemanticGroundingClaimReview:
    claim_id: str
    field_name: str
    value_index: int | None
    claim_text: str
    claim_type: str
    support_status: str
    severity: str
    supported_evidence_ids: tuple[str, ...]
    required_qualifications: tuple[str, ...] = ()
    missing_qualifications: tuple[str, ...] = ()
    rationale: str = ""
    repair_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "field_name": self.field_name,
            "value_index": self.value_index,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "support_status": self.support_status,
            "severity": self.severity,
            "supported_evidence_ids": list(self.supported_evidence_ids),
            "required_qualifications": list(self.required_qualifications),
            "missing_qualifications": list(self.missing_qualifications),
            "rationale": self.rationale,
            "repair_hint": self.repair_hint,
        }


@dataclass(frozen=True)
class SemanticGroundingReviewResult:
    passed: bool
    claim_reviews: tuple[SemanticGroundingClaimReview, ...]
    blocking_claim_ids: tuple[str, ...]
    automatic_fail_reason: str = ""
    requires_human_review: bool = False
    human_review_reason: str = ""
    repairable: bool = False
    repair_instructions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "claim_reviews": [claim.to_dict() for claim in self.claim_reviews],
            "blocking_claim_ids": list(self.blocking_claim_ids),
            "automatic_fail_reason": self.automatic_fail_reason,
            "requires_human_review": self.requires_human_review,
            "human_review_reason": self.human_review_reason,
            "repairable": self.repairable,
            "repair_instructions": list(self.repair_instructions),
        }


@dataclass(frozen=True)
class FinalPostSemanticGroundingState:
    status: str
    grounding_review: SemanticGroundingReviewResult | None
    error_code: str | None = None
    error_message: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "grounding_review": (
                self.grounding_review.to_dict() if self.grounding_review else None
            ),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "metadata": copy.deepcopy(self.metadata),
        }


def normalize_semantic_grounding_review_result(
    payload: dict[str, Any],
    *,
    selected_evidence_ids: tuple[str, ...] | list[str],
) -> SemanticGroundingReviewResult:
    """Normalize a strict semantic-grounding response into the domain contract."""

    if not isinstance(payload, dict):
        raise ValueError("semantic grounding review must be a JSON object.")

    selected_ids = tuple(selected_evidence_ids)
    selected_id_set = set(selected_ids)
    claim_reviews = _normalize_claim_reviews(payload.get("claims"), selected_id_set)
    failed_claim_ids = _string_tuple(payload.get("failed_claim_ids"), "failed_claim_ids")
    known_claim_ids = {claim.claim_id for claim in claim_reviews}
    for claim_id in failed_claim_ids:
        if claim_id not in known_claim_ids:
            raise ValueError(
                f"semantic grounding failed_claim_ids references unknown claim: {claim_id}"
            )

    passed = _required_bool(payload, "pass")
    automatic_fail_reason = _optional_string(payload, "automatic_fail_reason")
    requires_human_review = _optional_bool(payload, "requires_human_review", False)
    human_review_reason = _optional_string(payload, "human_review_reason")
    repairable = _optional_bool(payload, "repairable", False)
    repair_instructions = _string_tuple(
        payload.get("repair_instructions", ()),
        "repair_instructions",
    )

    blocking_claim_ids = tuple(
        claim.claim_id
        for claim in claim_reviews
        if is_blocking_claim_state(claim.support_status, claim.severity)
    )
    claim_by_id = {claim.claim_id: claim for claim in claim_reviews}
    for claim_id in failed_claim_ids:
        if not _claim_blocks_grounding(claim_by_id[claim_id]):
            raise ValueError(
                "semantic grounding failed_claim_ids must reference blocking claims."
            )
    if failed_claim_ids != blocking_claim_ids:
        raise ValueError(
            "semantic grounding failed_claim_ids must exactly match blocking claims."
        )
    if blocking_claim_ids and passed:
        raise ValueError("semantic grounding pass cannot be true with failed claims.")
    if automatic_fail_reason and passed:
        raise ValueError(
            "semantic grounding pass cannot be true with automatic_fail_reason."
        )
    if automatic_fail_reason and not blocking_claim_ids:
        raise ValueError(
            "semantic grounding automatic_fail_reason requires blocking claims."
        )
    if requires_human_review and passed:
        raise ValueError(
            "semantic grounding pass cannot be true when human review is required."
        )
    if requires_human_review and not human_review_reason:
        raise ValueError(
            "semantic grounding human_review_reason is required when human review is required."
        )
    if repairable and passed:
        raise ValueError(
            "semantic grounding pass cannot be true when repairable is true."
        )
    if repair_instructions and passed:
        raise ValueError(
            "semantic grounding pass cannot be true with repair_instructions."
        )
    if repair_instructions and not repairable:
        raise ValueError(
            "semantic grounding repair_instructions must be empty when repairable is false."
        )
    if repairable and requires_human_review:
        raise ValueError(
            "semantic grounding repairable cannot be true when human review is required."
        )
    if not passed and not blocking_claim_ids and not requires_human_review:
        raise ValueError(
            "semantic grounding pass=false requires blocking claims or human review."
        )
    if repairable and not blocking_claim_ids:
        raise ValueError(
            "semantic grounding repairable failures require blocking claims."
        )
    if repairable and not passed and not requires_human_review and not repair_instructions:
        raise ValueError(
            "semantic grounding repairable failures require repair_instructions."
        )
    if repairable:
        _validate_claim_addressed_repair(
            claim_by_id=claim_by_id,
            blocking_claim_ids=blocking_claim_ids,
            repair_instructions=repair_instructions,
        )

    return SemanticGroundingReviewResult(
        passed=passed,
        claim_reviews=claim_reviews,
        blocking_claim_ids=blocking_claim_ids,
        automatic_fail_reason=automatic_fail_reason,
        requires_human_review=requires_human_review,
        human_review_reason=human_review_reason,
        repairable=repairable,
        repair_instructions=repair_instructions,
    )


def _normalize_claim_reviews(
    claims: Any,
    selected_evidence_ids: set[str],
) -> tuple[SemanticGroundingClaimReview, ...]:
    if not isinstance(claims, list):
        raise ValueError("semantic grounding claims must be a list.")
    reviews = []
    seen_ids = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError("semantic grounding claim entries must be objects.")
        claim_id = _required_string(claim, "claim_id")
        if claim_id in seen_ids:
            raise ValueError(f"semantic grounding duplicate claim_id: {claim_id}")
        seen_ids.add(claim_id)
        evidence_ids = _string_tuple(
            claim.get("supported_evidence_ids", claim.get("evidence_ids", ())),
            "supported_evidence_ids",
        )
        for evidence_id in evidence_ids:
            if evidence_id not in selected_evidence_ids:
                raise ValueError(
                    f"semantic grounding claim {claim_id} references unselected evidence ID: {evidence_id}"
                )
        support_status = _required_enum(
            claim,
            "support_status",
            SEMANTIC_GROUNDING_SUPPORT_STATUSES,
        )
        severity = _required_enum(
            claim,
            "severity",
            SEMANTIC_GROUNDING_SEVERITIES,
        )
        if not is_valid_status_severity_pair(support_status, severity):
            raise ValueError(
                "semantic grounding support_status/severity pair is not allowed."
            )
        reviews.append(
            SemanticGroundingClaimReview(
                claim_id=claim_id,
                field_name=_required_string(claim, "field_name"),
                value_index=_optional_index(claim.get("value_index")),
                claim_text=_required_string(claim, "claim_text"),
                claim_type=_required_enum(
                    claim,
                    "claim_type",
                    SEMANTIC_GROUNDING_CLAIM_TYPES,
                ),
                support_status=support_status,
                severity=severity,
                supported_evidence_ids=evidence_ids,
                required_qualifications=_string_tuple(
                    claim.get("required_qualifications", ()),
                    "required_qualifications",
                ),
                missing_qualifications=_string_tuple(
                    claim.get("missing_qualifications", ()),
                    "missing_qualifications",
                ),
                rationale=_optional_string(claim, "rationale"),
                repair_hint=_optional_string(claim, "repair_hint"),
            )
        )
    return tuple(reviews)


def _claim_blocks_grounding(claim: SemanticGroundingClaimReview) -> bool:
    return is_blocking_claim_state(claim.support_status, claim.severity)


def _validate_claim_addressed_repair(
    *,
    claim_by_id: dict[str, SemanticGroundingClaimReview],
    blocking_claim_ids: tuple[str, ...],
    repair_instructions: tuple[str, ...],
) -> None:
    if len(repair_instructions) != len(blocking_claim_ids):
        raise ValueError(
            "semantic grounding repair_instructions must address each blocking claim."
        )

    addressed_claim_ids = []
    for claim_id, instruction in zip(blocking_claim_ids, repair_instructions):
        if not claim_by_id[claim_id].repair_hint.strip():
            raise ValueError(
                "semantic grounding repairable blocking claims require repair_hint."
            )
        stripped = instruction.strip()
        if not stripped.startswith(f"{claim_id}:") or not stripped[len(claim_id) + 1 :].strip():
            raise ValueError(
                "semantic grounding repair_instructions must start with claim_id."
            )
        addressed_claim_ids.append(claim_id)

    if tuple(addressed_claim_ids) != blocking_claim_ids:
        raise ValueError(
            "semantic grounding repair_instructions must match failed_claim_ids."
        )


def _required_bool(payload: dict[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"semantic grounding {field_name} must be a boolean.")
    return value


def _optional_bool(payload: dict[str, Any], field_name: str, default: bool) -> bool:
    value = payload.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(f"semantic grounding {field_name} must be a boolean.")
    return value


def _required_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"semantic grounding {field_name} must be a non-empty string.")
    return value


def _optional_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"semantic grounding {field_name} must be a string.")
    return value


def _required_enum(
    payload: dict[str, Any],
    field_name: str,
    allowed: tuple[str, ...],
) -> str:
    value = _required_string(payload, field_name)
    if value not in allowed:
        raise ValueError(f"unsupported semantic grounding {field_name}: {value}")
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"semantic grounding {field_name} must be a list.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"semantic grounding {field_name} entries must be non-empty strings."
            )
        result.append(item)
    return tuple(result)


def _optional_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("semantic grounding value_index must be a non-negative integer or null.")
    return value
