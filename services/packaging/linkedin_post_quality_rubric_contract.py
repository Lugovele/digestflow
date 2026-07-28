"""Versioned rubric payload for future LinkedIn post quality evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RUBRIC_VERSION = "1.0"
RUBRIC_SOURCE_DOCUMENT = "docs/linkedin-post-quality-target.md"

SCORE_MIN = 1
SCORE_MAX = 5
TOTAL_MIN = 9
TOTAL_MAX = 45
PASS_THRESHOLD = 36

QUALITY_CRITERIA = (
    "hook",
    "controlling_angle",
    "reader_problem",
    "pattern_interrupt",
    "evidence",
    "author_point_of_view",
    "human_voice",
    "practical_value",
    "cta",
)

_CRITERION_DEFINITIONS = (
    (
        "hook",
        "Evaluate whether the opening creates immediate relevance, tension, "
        "or curiosity before the post starts explaining.",
    ),
    (
        "controlling_angle",
        "Evaluate whether one clear main claim is visible throughout the post "
        "instead of a broad article summary or generic commentary.",
    ),
    (
        "reader_problem",
        "Evaluate whether the post names a concrete, recognizable reader "
        "problem tied to a real workflow or business situation.",
    ),
    (
        "pattern_interrupt",
        "Evaluate whether the post contains an early meaningful shift that "
        "reframes the expected topic.",
    ),
    (
        "evidence",
        "Evaluate whether selected evidence supports the angle without taking "
        "over the post or turning it into a source summary.",
    ),
    (
        "author_point_of_view",
        "Evaluate whether the author interprets the evidence with a clear "
        "judgment or thesis instead of neutrally reporting facts.",
    ),
    (
        "human_voice",
        "Evaluate whether the post sounds natural, specific, and human rather "
        "than corporate, template-like, or generic AI-generated text.",
    ),
    (
        "practical_value",
        "Evaluate whether the post gives one useful takeaway the reader can "
        "apply.",
    ),
    (
        "cta",
        "Evaluate whether the post has one open question or call to action "
        "that naturally follows from the post.",
    ),
)

_REQUIRED_MINIMUMS = (
    ("hook", 4),
    ("controlling_angle", 4),
    ("author_point_of_view", 4),
    ("human_voice", 4),
    ("evidence", 3),
)

_AUTOMATIC_FAIL_CONDITIONS = (
    "invents facts",
    "invents personal experience",
    "uses external links in the body",
    "has no clear angle",
    "reads like a summary of articles",
    "sounds like generic AI-generated content",
    "reads like a corporate memo instead of a human LinkedIn post",
    "has no human author voice",
    "makes source terminology the main angle by accident",
    "exceeds 1300 characters",
    "relies on generic phrases as the main argument",
    "has more than one CTA",
)


@dataclass(frozen=True)
class QualityEvaluatorRubricPayload:
    rubric_version: str
    source_document: str
    criteria: dict[str, str]
    score_min: int
    score_max: int
    total_min: int
    total_max: int
    pass_threshold: int
    required_minimums: dict[str, int]
    automatic_fail_conditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_version": self.rubric_version,
            "source_document": self.source_document,
            "criteria": dict(self.criteria),
            "score_min": self.score_min,
            "score_max": self.score_max,
            "total_min": self.total_min,
            "total_max": self.total_max,
            "pass_threshold": self.pass_threshold,
            "required_minimums": dict(self.required_minimums),
            "automatic_fail_conditions": list(self.automatic_fail_conditions),
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "rubric_version": self.rubric_version,
            "criteria": dict(self.criteria),
            "score_min": self.score_min,
            "score_max": self.score_max,
            "total_min": self.total_min,
            "total_max": self.total_max,
            "pass_threshold": self.pass_threshold,
            "required_minimums": dict(self.required_minimums),
            "automatic_fail_conditions": list(self.automatic_fail_conditions),
        }


def get_quality_evaluator_rubric_payload() -> QualityEvaluatorRubricPayload:
    return QualityEvaluatorRubricPayload(
        rubric_version=RUBRIC_VERSION,
        source_document=RUBRIC_SOURCE_DOCUMENT,
        criteria=dict(_CRITERION_DEFINITIONS),
        score_min=SCORE_MIN,
        score_max=SCORE_MAX,
        total_min=TOTAL_MIN,
        total_max=TOTAL_MAX,
        pass_threshold=PASS_THRESHOLD,
        required_minimums=dict(_REQUIRED_MINIMUMS),
        automatic_fail_conditions=tuple(_AUTOMATIC_FAIL_CONDITIONS),
    )
