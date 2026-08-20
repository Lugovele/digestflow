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
        "over the post, turning it into a source summary, or adding causal "
        "claims not supported by selected evidence.",
    ),
    (
        "author_point_of_view",
        "Evaluate whether the post text contains a visible author-owned "
        "judgment about what the author notices, rejects, or thinks matters "
        "instead of only a clear thesis, rhetorical question, or neutral "
        "source relationship. Score 5 requires exactly one explicit "
        "author-owned interpretive statement tied to supplied evidence. "
        "Do not count ordinary thesis, synthesis, or rhetorical consequence "
        "as another explicit ownership statement unless it includes explicit "
        "author self-attribution.",
    ),
    (
        "human_voice",
        "Evaluate whether the post sounds natural, readable, specific, and "
        "human rather than corporate, template-like, summary-like, or generic "
        "AI-generated text. Do not require first person for human_voice.",
    ),
    (
        "practical_value",
        "Evaluate whether the post gives one useful takeaway the reader can "
        "apply instead of ending with broad market interpretation or generic "
        "advice.",
    ),
    (
        "cta",
        "Evaluate whether the post has one open question or call to action "
        "inside post_text that naturally follows from the post.",
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
    "invents cases",
    "invents metrics",
    "invents personal experience",
    "uses external links in the body",
    "has no clear angle",
    "reads like a summary of articles",
    "uses unsupported causal strengthening",
    "sounds like generic AI-generated content",
    "reads like a corporate memo instead of a human LinkedIn post",
    "has no human author voice",
    "makes source terminology the main angle by accident",
    "exceeds 1300 characters",
    "relies on generic phrases as the main argument",
    "has no required CTA in post_text",
    "has more than one CTA",
    "leaks internal process language into reader-facing text",
)

_SCORING_INVARIANTS = (
    "CTA scoring must be based on the actual post_text, not cta_variants.",
    "When a required CTA is absent from post_text, cta must score 1 and the review must fail.",
    "Summary-like source recap cannot score 4 or 5 for both author_point_of_view and human_voice.",
    "A post without a concrete reader takeaway cannot pass practical_value.",
    "Unsupported factual or causal drift must fail evidence or trigger automatic failure.",
    "Source coverage is not the same as synthesis.",
    "Clean grammar and coherent structure are not sufficient for human_voice.",
    "Readable human voice is not sufficient for author_point_of_view.",
    "Score 5 for author_point_of_view requires exactly one explicit author-owned interpretive statement, a substantive choice between competing readings, evidence-tied judgment, and no fabricated experience or authority.",
    "An explicit author-owned interpretive statement must visibly assign the interpretation to the author through a clear ownership signal such as I reject, I do not think, my reading, or I would treat this as.",
    "Ordinary thesis, synthesis, practical takeaway, causal interpretation, or rhetorical consequence statements do not count as additional explicit author-owned statements unless they independently include explicit author self-attribution or invented author context.",
    "When personal_presence_requirement is explicit_author_owned_statement_required, zero qualifying explicit author-owned interpretive statements must score no higher than 3 for author_point_of_view and pass must be false.",
    "When personal_presence_requirement is explicit_author_owned_statement_required, multiple qualifying explicit author-owned interpretive statements must score no higher than 4 for author_point_of_view and pass must be false.",
    "Strong article-like editorial ownership with limited explicit personal presence may qualify for author_point_of_view 4 but not 5 only when explicit personal presence is not required.",
    "A strong thesis, rhetorical question, short sentences, or editorial confidence alone is not sufficient for author_point_of_view 5.",
    "First-person wording alone is not sufficient for author_point_of_view.",
    "Personal presence must not automatically increase human_voice.",
    "Declared brief or angle metadata must not inflate scores when post_text does not deliver it.",
    "Any automatic failure forces pass to false regardless of total_score.",
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
    scoring_invariants: tuple[str, ...]

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
            "scoring_invariants": list(self.scoring_invariants),
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
            "scoring_invariants": list(self.scoring_invariants),
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
        scoring_invariants=tuple(_SCORING_INVARIANTS),
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
        "scoring_invariants",
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
    scoring_invariants = rubric["scoring_invariants"]
    if not isinstance(scoring_invariants, (list, tuple)):
        raise ValueError(
            "quality evaluator rubric scoring_invariants must be a list or tuple."
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
        scoring_invariants=tuple(
            _require_string(item, "scoring_invariants")
            for item in scoring_invariants
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
