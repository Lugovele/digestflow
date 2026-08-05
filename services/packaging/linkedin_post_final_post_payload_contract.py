"""Canonical structural constraints for Candidate Writer FinalPostPayload output."""
from __future__ import annotations

from typing import Any


FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS = 1300
FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MIN_CHARS = 1100
FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MAX_CHARS = 1250
FINAL_POST_PAYLOAD_HOOK_VARIANTS_MIN_COUNT = 3
FINAL_POST_PAYLOAD_CTA_VARIANTS_MIN_COUNT = 3
FINAL_POST_PAYLOAD_HASHTAGS_MIN_COUNT = 1

REQUIRED_QUALITY_CHECKS = (
    "linkedin_ready",
    "uses_only_provided_facts",
    "has_clear_point_of_view",
)


def build_final_post_payload_constraints() -> dict[str, Any]:
    """Return the deterministic Candidate Writer payload constraints."""

    return {
        "schema_version": "1.0",
        "post_text": {
            "required": True,
            "type": "string",
            "hard_max_chars": FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
            "prompt_target_min_chars": (
                FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MIN_CHARS
            ),
            "prompt_target_max_chars": (
                FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MAX_CHARS
            ),
        },
        "hook_variants": {
            "required": True,
            "type": "list[string]",
            "min_count": FINAL_POST_PAYLOAD_HOOK_VARIANTS_MIN_COUNT,
        },
        "cta_variants": {
            "required": True,
            "type": "list[string]",
            "min_count": FINAL_POST_PAYLOAD_CTA_VARIANTS_MIN_COUNT,
        },
        "hashtags": {
            "required": True,
            "type": "list[string]",
            "min_count": FINAL_POST_PAYLOAD_HASHTAGS_MIN_COUNT,
        },
        "quality_checks": {
            "required": True,
            "type": "object",
            "required_boolean_keys": list(REQUIRED_QUALITY_CHECKS),
        },
        "carousel_outline": {
            "required": False,
            "type": "list",
            "default": [],
        },
    }


__all__ = (
    "FINAL_POST_PAYLOAD_CTA_VARIANTS_MIN_COUNT",
    "FINAL_POST_PAYLOAD_HASHTAGS_MIN_COUNT",
    "FINAL_POST_PAYLOAD_HOOK_VARIANTS_MIN_COUNT",
    "FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS",
    "FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MAX_CHARS",
    "FINAL_POST_PAYLOAD_POST_TEXT_PROMPT_TARGET_MIN_CHARS",
    "REQUIRED_QUALITY_CHECKS",
    "build_final_post_payload_constraints",
)
