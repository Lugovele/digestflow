"""Post-acceptance publication assembly for LinkedIn core posts.

This module starts after deterministic acceptance. It does not call providers,
rewrite posts, judge quality, repair text, or own Candidate Writer output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_post_candidate_post_contract import (
    CandidatePost,
    candidate_post_from_dict,
    validate_candidate_post,
)


@dataclass(frozen=True)
class AcceptedPost:
    """Accepted core LinkedIn post content."""

    post_text: str

    def __post_init__(self) -> None:
        validate_candidate_post(CandidatePost(post_text=self.post_text))

    def to_dict(self) -> dict[str, str]:
        return {"post_text": self.post_text}


@dataclass(frozen=True)
class LinkedInPublicationPackage:
    """Publication-facing projection assembled after core-post acceptance."""

    post_text: str
    hook_variants: tuple[str, ...]
    cta_variants: tuple[str, ...]
    hashtags: tuple[str, ...]
    carousel_outline: tuple[dict[str, Any], ...]
    quality_checks: dict[str, bool]
    validation_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_text": self.post_text,
            "hook_variants": list(self.hook_variants),
            "cta_variants": list(self.cta_variants),
            "hashtags": list(self.hashtags),
            "carousel_outline": [dict(item) for item in self.carousel_outline],
            "quality_checks": dict(self.quality_checks),
            "validation_report": _serialize_json_value(self.validation_report),
        }


def accepted_post_from_payload(payload: dict[str, Any]) -> AcceptedPost:
    """Build an accepted core post from a CandidatePost-shaped payload."""

    candidate_post = candidate_post_from_dict(payload)
    return AcceptedPost(post_text=candidate_post.post_text)


def assemble_linkedin_publication_package(
    accepted_post: AcceptedPost,
    *,
    deterministic_gate_passed: bool,
    semantic_grounding_passed: bool,
    quality_passed: bool,
    final_accepted: bool,
    hook_variants: tuple[str, ...] = (),
    cta_variants: tuple[str, ...] = (),
    hashtags: tuple[str, ...] = (),
    carousel_outline: tuple[dict[str, Any], ...] = (),
) -> LinkedInPublicationPackage:
    """Assemble optional publication packaging from an accepted core post."""

    _require_bool(deterministic_gate_passed, "deterministic_gate_passed")
    _require_bool(semantic_grounding_passed, "semantic_grounding_passed")
    _require_bool(quality_passed, "quality_passed")
    _require_bool(final_accepted, "final_accepted")
    if not final_accepted:
        raise ValueError("publication package can only be assembled after acceptance.")

    quality_checks = _derived_quality_checks(
        deterministic_gate_passed=deterministic_gate_passed,
        semantic_grounding_passed=semantic_grounding_passed,
        quality_passed=quality_passed,
        final_accepted=final_accepted,
    )
    validation_report = {
        "status": "valid" if all(quality_checks.values()) else "invalid",
        "derived_from": {
            "deterministic_gate": deterministic_gate_passed,
            "semantic_grounding": semantic_grounding_passed,
            "quality_evaluator": quality_passed,
            "adjudication": final_accepted,
        },
    }
    return LinkedInPublicationPackage(
        post_text=accepted_post.post_text,
        hook_variants=_normalize_optional_string_tuple(
            hook_variants,
            "hook_variants",
        ),
        cta_variants=_normalize_optional_string_tuple(cta_variants, "cta_variants"),
        hashtags=_normalize_optional_string_tuple(hashtags, "hashtags"),
        carousel_outline=_normalize_carousel_outline(carousel_outline),
        quality_checks=quality_checks,
        validation_report=validation_report,
    )


def _derived_quality_checks(
    *,
    deterministic_gate_passed: bool,
    semantic_grounding_passed: bool,
    quality_passed: bool,
    final_accepted: bool,
) -> dict[str, bool]:
    return {
        "linkedin_ready": (
            deterministic_gate_passed
            and semantic_grounding_passed
            and quality_passed
            and final_accepted
        ),
        "uses_only_provided_facts": semantic_grounding_passed,
        "has_clear_point_of_view": quality_passed,
    }


def _normalize_optional_string_tuple(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple.")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain only non-empty strings.")
    return tuple(value)


def _normalize_carousel_outline(
    value: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, tuple):
        raise TypeError("carousel_outline must be a tuple.")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("carousel_outline must contain dictionaries.")
        normalized.append(dict(item))
    return tuple(normalized)


def _require_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")


def _serialize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_value(item) for item in value]
    return value
