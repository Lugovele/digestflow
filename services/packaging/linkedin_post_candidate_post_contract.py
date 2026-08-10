"""Core Candidate Writer post contract for first-attempt PostFlow validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)


CANDIDATE_POST_TEXT_MAX_CHARS = FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS
CANDIDATE_POST_FIELDS = ("post_text",)
CANDIDATE_POST_REQUIRED_FIELDS = ("post_text",)


class CandidatePostContractError(ValueError):
    """Raised when a core CandidatePost violates deterministic structure."""


@dataclass(frozen=True)
class CandidatePost:
    """Canonical core post artifact produced by Candidate Writer."""

    post_text: str

    def to_dict(self) -> dict[str, str]:
        return candidate_post_to_dict(self)


def validate_candidate_post(candidate_post: CandidatePost) -> None:
    if not isinstance(candidate_post, CandidatePost):
        raise CandidatePostContractError("candidate_post must be a CandidatePost.")
    if not isinstance(candidate_post.post_text, str):
        raise CandidatePostContractError("post_text must be a string.")
    if not candidate_post.post_text.strip():
        raise CandidatePostContractError("post_text must be a non-empty string.")
    if len(candidate_post.post_text) > CANDIDATE_POST_TEXT_MAX_CHARS:
        raise CandidatePostContractError(
            "post_text must be at most "
            f"{CANDIDATE_POST_TEXT_MAX_CHARS} characters."
        )


def candidate_post_to_dict(candidate_post: CandidatePost) -> dict[str, str]:
    validate_candidate_post(candidate_post)
    return {"post_text": candidate_post.post_text}


def candidate_post_from_dict(value: dict[str, Any]) -> CandidatePost:
    if not isinstance(value, dict):
        raise CandidatePostContractError("candidate post payload must be a dictionary.")
    unexpected_fields = sorted(set(value) - set(CANDIDATE_POST_FIELDS))
    if unexpected_fields:
        raise CandidatePostContractError(
            f"candidate post payload contains unexpected fields: {unexpected_fields}."
        )
    missing_fields = [field for field in CANDIDATE_POST_REQUIRED_FIELDS if field not in value]
    if missing_fields:
        raise CandidatePostContractError(
            f"candidate post payload is missing required fields: {missing_fields}."
        )
    candidate_post = CandidatePost(post_text=value["post_text"])
    validate_candidate_post(candidate_post)
    return candidate_post


def build_candidate_post_constraints() -> dict[str, Any]:
    return {
        "post_text": {
            "type": "string",
            "required": True,
            "min_chars": 1,
            "max_chars": CANDIDATE_POST_TEXT_MAX_CHARS,
        },
        "allowed_fields": list(CANDIDATE_POST_FIELDS),
        "forbidden_fields": [
            "hook_variants",
            "cta_variants",
            "hashtags",
            "quality_checks",
            "carousel_outline",
        ],
    }
