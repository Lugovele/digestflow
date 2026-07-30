"""Versioned rubric payload for future LinkedIn post quality evaluation."""
from __future__ import annotations

import copy
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


def normalize_quality_evaluator_rubric_payload(
    rubric: QualityEvaluatorRubricPayload | dict[str, Any],
) -> QualityEvaluatorRubricPayload:
    """Return the canonical rubric payload required by prompt rendering."""

    if isinstance(rubric, QualityEvaluatorRubricPayload):
        return rubric
    if not isinstance(rubric, dict):
        raise ValueError("quality evaluator rubric must be a rubric payload or dictionary.")

    required_fields = {
        "rubric_version",
        "source_document",
        "criteria",
        "score_min",
        "score_max",
        "total_min",
        "total_max",
        "pass_threshold",
        "required_minimums",
        "automatic_fail_conditions",
    }
    missing_fields = sorted(required_fields - rubric.keys())
    if missing_fields:
        raise ValueError(
            f"quality evaluator rubric is missing required fields: {missing_fields}."
        )

    automatic_fail_conditions = rubric["automatic_fail_conditions"]
    if not isinstance(automatic_fail_conditions, (list, tuple)):
        raise ValueError(
            "quality evaluator rubric automatic_fail_conditions must be a list or tuple."
        )

    criteria = _require_string_dict(rubric["criteria"], "criteria")
    missing_criteria = sorted(set(QUALITY_CRITERIA) - criteria.keys())
    extra_criteria = sorted(criteria.keys() - set(QUALITY_CRITERIA))
    if missing_criteria or extra_criteria:
        raise ValueError(
            "quality evaluator rubric criteria must match the canonical full shape."
        )

    required_minimums = _require_int_dict(
        rubric["required_minimums"],
        "required_minimums",
    )
    canonical_required_minimums = dict(_REQUIRED_MINIMUMS)
    missing_minimums = sorted(canonical_required_minimums.keys() - required_minimums.keys())
    extra_minimums = sorted(required_minimums.keys() - canonical_required_minimums.keys())
    if missing_minimums or extra_minimums:
        raise ValueError(
            "quality evaluator rubric required_minimums must match the canonical full shape."
        )

    return QualityEvaluatorRubricPayload(
        rubric_version=_require_string(rubric["rubric_version"], "rubric_version"),
        source_document=_require_string(rubric["source_document"], "source_document"),
        criteria=criteria,
        score_min=_require_int(rubric["score_min"], "score_min"),
        score_max=_require_int(rubric["score_max"], "score_max"),
        total_min=_require_int(rubric["total_min"], "total_min"),
        total_max=_require_int(rubric["total_max"], "total_max"),
        pass_threshold=_require_int(rubric["pass_threshold"], "pass_threshold"),
        required_minimums=required_minimums,
        automatic_fail_conditions=tuple(
            _require_string(item, "automatic_fail_conditions")
            for item in automatic_fail_conditions
        ),
    )


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"quality evaluator rubric {field_name} must be a string.")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"quality evaluator rubric {field_name} must be an integer.")
    return value


def _require_string_dict(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"quality evaluator rubric {field_name} must be a dictionary.")
    result: dict[str, str] = {}
    for key, item in copy.deepcopy(value).items():
        result[_require_string(key, field_name)] = _require_string(item, field_name)
    return result


def _require_int_dict(value: Any, field_name: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"quality evaluator rubric {field_name} must be a dictionary.")
    result: dict[str, int] = {}
    for key, item in copy.deepcopy(value).items():
        result[_require_string(key, field_name)] = _require_int(item, field_name)
    return result
