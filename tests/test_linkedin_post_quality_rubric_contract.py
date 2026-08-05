from __future__ import annotations

import inspect
import json
from pathlib import Path
import re

from django.test import SimpleTestCase

from services.packaging import linkedin_post_quality_rubric_contract
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_ACCEPT,
    ACTION_NEEDS_HUMAN_REVIEW,
    ACTION_NOT_READY,
    ACTION_REPAIR_EDITORIAL,
    ACTION_REPAIR_MECHANICAL,
    ACTION_TRY_ALTERNATIVE_MODEL,
)
from services.packaging.linkedin_post_flow_decision import (
    QUALITY_PASS_THRESHOLD,
    REQUIRED_QUALITY_MINIMUMS,
)
from services.packaging.linkedin_post_prompt_registry import (
    PROMPT_FINAL_POST_QUALITY_EVALUATOR,
    get_prompt_contract,
)
from services.packaging.linkedin_post_quality_review_contract import (
    CANONICAL_QUALITY_SCORE_KEYS,
    MAX_CRITERION_SCORE,
    MAX_TOTAL_SCORE,
    MIN_CRITERION_SCORE,
    MIN_TOTAL_SCORE,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    QualityEvaluatorRubricPayload,
    get_quality_evaluator_rubric_payload,
    normalize_quality_evaluator_rubric_payload,
)


EXPECTED_CRITERIA = (
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

EXPECTED_REQUIRED_MINIMUMS = {
    "hook": 4,
    "controlling_angle": 4,
    "author_point_of_view": 4,
    "human_voice": 4,
    "evidence": 3,
}

EXPECTED_AUTOMATIC_FAIL_CONDITIONS = (
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

EXPECTED_SCORING_INVARIANTS = (
    "CTA scoring must be based on the actual post_text, not cta_variants.",
    "When a required CTA is absent from post_text, cta must score 1 and the review must fail.",
    "Summary-like source recap cannot score 4 or 5 for both author_point_of_view and human_voice.",
    "A post without a concrete reader takeaway cannot pass practical_value.",
    "Unsupported factual or causal drift must fail evidence or trigger automatic failure.",
    "Source coverage is not the same as synthesis.",
    "Clean grammar and coherent structure are not sufficient for human_voice.",
    "Readable human voice is not sufficient for author_point_of_view.",
    "Score 5 for author_point_of_view requires exactly one explicit author-owned interpretive statement, a substantive choice between competing readings, evidence-tied judgment, and no fabricated experience or authority.",
    "When personal_presence_requirement is explicit_author_owned_statement_required, zero qualifying explicit author-owned interpretive statements must score no higher than 3 for author_point_of_view and pass must be false.",
    "When personal_presence_requirement is explicit_author_owned_statement_required, multiple qualifying explicit author-owned interpretive statements must score no higher than 4 for author_point_of_view and pass must be false.",
    "Strong article-like editorial ownership with limited explicit personal presence may qualify for author_point_of_view 4 but not 5 only when explicit personal presence is not required.",
    "A strong thesis, rhetorical question, short sentences, or editorial confidence alone is not sufficient for author_point_of_view 5.",
    "First-person wording alone is not sufficient for author_point_of_view.",
    "Personal presence must not automatically increase human_voice.",
    "Declared brief or angle metadata must not inflate scores when post_text does not deliver it.",
    "Any automatic failure forces pass to false regardless of total_score.",
)

DECISION_ACTIONS = (
    ACTION_ACCEPT,
    ACTION_REPAIR_MECHANICAL,
    ACTION_REPAIR_EDITORIAL,
    ACTION_TRY_ALTERNATIVE_MODEL,
    ACTION_NOT_READY,
    ACTION_NEEDS_HUMAN_REVIEW,
)


class LinkedInPostQualityRubricContractTests(SimpleTestCase):
    def test_factory_returns_quality_evaluator_rubric_payload(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertIsInstance(payload, QualityEvaluatorRubricPayload)

    def test_rubric_version_is_stable(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(payload.rubric_version, "1.0")

    def test_source_document_is_recorded(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(payload.source_document, "docs/linkedin-post-quality-target.md")

    def test_exactly_nine_criteria_exist(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(len(payload.criteria), 9)

    def test_criteria_order_is_canonical(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(tuple(payload.criteria), EXPECTED_CRITERIA)

    def test_human_voice_is_present(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertIn("human_voice", payload.criteria)

    def test_no_unexpected_criterion_exists(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(set(payload.criteria), set(EXPECTED_CRITERIA))
        self.assertNotIn("structure", payload.criteria)
        self.assertNotIn("specificity", payload.criteria)
        self.assertNotIn("reader_relevance", payload.criteria)

    def test_every_criterion_has_a_non_empty_definition(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        for criterion, definition in payload.criteria.items():
            with self.subTest(criterion=criterion):
                self.assertIsInstance(definition, str)
                self.assertTrue(definition.strip())

    def test_definitions_are_evaluation_only(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        for criterion, definition in payload.criteria.items():
            with self.subTest(criterion=criterion):
                normalized = definition.lower()
                self.assertIn("evaluate", normalized)
                self.assertNotIn("rewrite", normalized)
                self.assertNotIn("repair", normalized)
                self.assertNotIn("route", normalized)

    def test_score_minimum_is_one(self) -> None:
        self.assertEqual(get_quality_evaluator_rubric_payload().score_min, 1)

    def test_score_maximum_is_five(self) -> None:
        self.assertEqual(get_quality_evaluator_rubric_payload().score_max, 5)

    def test_total_minimum_is_nine(self) -> None:
        self.assertEqual(get_quality_evaluator_rubric_payload().total_min, 9)

    def test_total_maximum_is_forty_five(self) -> None:
        self.assertEqual(get_quality_evaluator_rubric_payload().total_max, 45)

    def test_pass_threshold_is_thirty_six(self) -> None:
        self.assertEqual(get_quality_evaluator_rubric_payload().pass_threshold, 36)

    def test_required_minimums_match_documented_five_criteria(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(payload.required_minimums, EXPECTED_REQUIRED_MINIMUMS)

    def test_no_extra_required_minimum_exists(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(set(payload.required_minimums), set(EXPECTED_REQUIRED_MINIMUMS))

    def test_automatic_failure_list_matches_documentation_exactly(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(
            payload.automatic_fail_conditions,
            _automatic_fail_conditions_from_docs(),
        )

    def test_automatic_failure_list_is_non_empty(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertTrue(payload.automatic_fail_conditions)

    def test_scoring_invariants_are_present(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(payload.scoring_invariants, EXPECTED_SCORING_INVARIANTS)

    def test_scoring_invariants_capture_false_positive_regressions(self) -> None:
        invariants = " ".join(get_quality_evaluator_rubric_payload().scoring_invariants)

        for phrase in (
            "actual post_text",
            "not cta_variants",
            "source recap",
            "author_point_of_view and human_voice",
            "concrete reader takeaway",
            "causal drift",
            "Source coverage is not the same as synthesis",
            "Clean grammar and coherent structure",
            "Readable human voice is not sufficient",
            "Score 5 for author_point_of_view requires exactly one",
            "zero qualifying explicit author-owned",
            "multiple qualifying explicit author-owned",
            "First-person wording alone",
            "Personal presence must not automatically increase human_voice",
            "metadata must not inflate scores",
            "automatic failure forces pass to false",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, invariants)

    def test_automatic_failures_contain_no_routing_action_names(self) -> None:
        payload = get_quality_evaluator_rubric_payload()
        text = " ".join(payload.automatic_fail_conditions)

        for action in DECISION_ACTIONS:
            with self.subTest(action=action):
                self.assertNotIn(action, text)

    def test_payload_is_json_serializable(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        serialized = json.dumps(payload.to_dict(), ensure_ascii=False, sort_keys=True)

        self.assertIn("human_voice", serialized)

    def test_to_dict_preserves_order_and_values(self) -> None:
        payload = get_quality_evaluator_rubric_payload()
        payload_dict = payload.to_dict()

        self.assertEqual(tuple(payload_dict["criteria"]), EXPECTED_CRITERIA)
        self.assertEqual(
            tuple(payload_dict["required_minimums"]),
            tuple(EXPECTED_REQUIRED_MINIMUMS),
        )
        self.assertEqual(
            payload_dict["automatic_fail_conditions"],
            list(EXPECTED_AUTOMATIC_FAIL_CONDITIONS),
        )
        self.assertEqual(
            payload_dict["scoring_invariants"],
            list(EXPECTED_SCORING_INVARIANTS),
        )
        self.assertEqual(payload_dict["pass_threshold"], 36)

    def test_to_prompt_dict_returns_dictionary(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertIsInstance(payload.to_prompt_dict(), dict)

    def test_to_prompt_dict_preserves_model_facing_rubric_fields(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(
            tuple(payload.to_prompt_dict()),
            (
                "rubric_version",
                "criteria",
                "score_min",
                "score_max",
                "total_min",
                "total_max",
                "pass_threshold",
                "required_minimums",
                "automatic_fail_conditions",
                "scoring_invariants",
            ),
        )

    def test_to_prompt_dict_excludes_source_document(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertNotIn("source_document", payload.to_prompt_dict())

    def test_to_prompt_dict_preserves_order_and_values(self) -> None:
        payload = get_quality_evaluator_rubric_payload()
        prompt_dict = payload.to_prompt_dict()

        self.assertEqual(prompt_dict["rubric_version"], "1.0")
        self.assertEqual(tuple(prompt_dict["criteria"]), EXPECTED_CRITERIA)
        self.assertEqual(prompt_dict["score_min"], 1)
        self.assertEqual(prompt_dict["score_max"], 5)
        self.assertEqual(prompt_dict["total_min"], 9)
        self.assertEqual(prompt_dict["total_max"], 45)
        self.assertEqual(prompt_dict["pass_threshold"], 36)
        self.assertEqual(
            tuple(prompt_dict["required_minimums"]),
            tuple(EXPECTED_REQUIRED_MINIMUMS),
        )
        self.assertEqual(
            prompt_dict["automatic_fail_conditions"],
            list(EXPECTED_AUTOMATIC_FAIL_CONDITIONS),
        )
        self.assertEqual(
            prompt_dict["scoring_invariants"],
            list(EXPECTED_SCORING_INVARIANTS),
        )

    def test_to_prompt_dict_is_json_serializable(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        serialized = json.dumps(
            payload.to_prompt_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        self.assertIn("human_voice", serialized)

    def test_to_prompt_dict_mutation_does_not_mutate_source_payload(self) -> None:
        payload = get_quality_evaluator_rubric_payload()
        prompt_dict = payload.to_prompt_dict()

        prompt_dict["criteria"]["hook"] = "changed"
        prompt_dict["required_minimums"]["hook"] = 1
        prompt_dict["automatic_fail_conditions"].append("changed")
        prompt_dict["scoring_invariants"].append("changed")

        self.assertNotEqual(payload.criteria["hook"], "changed")
        self.assertEqual(payload.required_minimums["hook"], 4)
        self.assertNotIn("changed", payload.automatic_fail_conditions)
        self.assertNotIn("changed", payload.scoring_invariants)

    def test_repeated_to_prompt_dict_calls_return_independent_nested_values(self) -> None:
        payload = get_quality_evaluator_rubric_payload()
        first = payload.to_prompt_dict()
        second = payload.to_prompt_dict()

        first["criteria"]["hook"] = "changed"
        first["required_minimums"]["hook"] = 1
        first["automatic_fail_conditions"].append("changed")
        first["scoring_invariants"].append("changed")

        self.assertNotEqual(second["criteria"]["hook"], "changed")
        self.assertEqual(second["required_minimums"]["hook"], 4)
        self.assertNotIn("changed", second["automatic_fail_conditions"])
        self.assertNotIn("changed", second["scoring_invariants"])
        self.assertIsNot(first["criteria"], second["criteria"])
        self.assertIsNot(first["required_minimums"], second["required_minimums"])
        self.assertIsNot(
            first["automatic_fail_conditions"],
            second["automatic_fail_conditions"],
        )
        self.assertIsNot(first["scoring_invariants"], second["scoring_invariants"])

    def test_to_dict_still_includes_provenance(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertIn("source_document", payload.to_dict())

    def test_normalize_rubric_payload_accepts_existing_payload_unchanged(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        normalized = normalize_quality_evaluator_rubric_payload(payload)

        self.assertIs(normalized, payload)

    def test_normalize_rubric_payload_accepts_full_serialized_payload(self) -> None:
        payload_dict = get_quality_evaluator_rubric_payload().to_dict()

        normalized = normalize_quality_evaluator_rubric_payload(payload_dict)

        self.assertIsInstance(normalized, QualityEvaluatorRubricPayload)
        self.assertEqual(normalized.to_dict(), payload_dict)

    def test_normalize_rubric_payload_rejects_prompt_only_dict(self) -> None:
        prompt_only = get_quality_evaluator_rubric_payload().to_prompt_dict()

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            normalize_quality_evaluator_rubric_payload(prompt_only)

    def test_normalize_rubric_payload_rejects_placeholder_dict(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            normalize_quality_evaluator_rubric_payload({})

    def test_normalize_rubric_payload_rejects_partial_nested_criteria(self) -> None:
        payload_dict = get_quality_evaluator_rubric_payload().to_dict()
        payload_dict["criteria"] = {"hook": payload_dict["criteria"]["hook"]}

        with self.assertRaisesRegex(ValueError, "criteria must match"):
            normalize_quality_evaluator_rubric_payload(payload_dict)

    def test_normalize_rubric_payload_rejects_partial_nested_minimums(self) -> None:
        payload_dict = get_quality_evaluator_rubric_payload().to_dict()
        payload_dict["required_minimums"] = {
            "hook": payload_dict["required_minimums"]["hook"]
        }

        with self.assertRaisesRegex(ValueError, "required_minimums must match"):
            normalize_quality_evaluator_rubric_payload(payload_dict)

    def test_normalize_rubric_payload_defensively_copies_nested_values(self) -> None:
        payload_dict = get_quality_evaluator_rubric_payload().to_dict()

        normalized = normalize_quality_evaluator_rubric_payload(payload_dict)
        payload_dict["criteria"]["hook"] = "changed"
        payload_dict["required_minimums"]["hook"] = 1
        payload_dict["automatic_fail_conditions"].append("changed")
        payload_dict["scoring_invariants"].append("changed")

        self.assertNotEqual(normalized.criteria["hook"], "changed")
        self.assertEqual(normalized.required_minimums["hook"], 4)
        self.assertNotIn("changed", normalized.automatic_fail_conditions)
        self.assertNotIn("changed", normalized.scoring_invariants)

    def test_normalize_rubric_payload_accepts_tuple_automatic_fail_conditions(
        self,
    ) -> None:
        payload_dict = get_quality_evaluator_rubric_payload().to_dict()
        payload_dict["automatic_fail_conditions"] = tuple(
            payload_dict["automatic_fail_conditions"]
        )

        normalized = normalize_quality_evaluator_rubric_payload(payload_dict)

        self.assertIsInstance(normalized.automatic_fail_conditions, tuple)

    def test_serialized_mutation_does_not_mutate_source_payload(self) -> None:
        payload = get_quality_evaluator_rubric_payload()
        payload_dict = payload.to_dict()

        payload_dict["criteria"]["hook"] = "changed"
        payload_dict["required_minimums"]["hook"] = 1
        payload_dict["automatic_fail_conditions"].append("changed")
        payload_dict["scoring_invariants"].append("changed")

        self.assertNotEqual(payload.criteria["hook"], "changed")
        self.assertEqual(payload.required_minimums["hook"], 4)
        self.assertNotIn("changed", payload.automatic_fail_conditions)
        self.assertNotIn("changed", payload.scoring_invariants)

    def test_repeated_factory_results_are_independent(self) -> None:
        first = get_quality_evaluator_rubric_payload()
        second = get_quality_evaluator_rubric_payload()

        first.criteria["hook"] = "changed"
        first.required_minimums["hook"] = 1

        self.assertNotEqual(second.criteria["hook"], "changed")
        self.assertEqual(second.required_minimums["hook"], 4)
        self.assertIsNot(first.criteria, second.criteria)
        self.assertIsNot(first.required_minimums, second.required_minimums)

    def test_no_runtime_markdown_parsing_exists(self) -> None:
        source = inspect.getsource(linkedin_post_quality_rubric_contract)

        self.assertNotIn("markdown", source.lower())
        self.assertNotIn("read_text", source)

    def test_no_filesystem_access_exists_in_rubric_module(self) -> None:
        source = inspect.getsource(linkedin_post_quality_rubric_contract)

        self.assertNotIn("Path", source)
        self.assertNotIn("open(", source)
        self.assertNotIn("os.", source)

    def test_no_prompt_loading_exists(self) -> None:
        source = inspect.getsource(linkedin_post_quality_rubric_contract)

        self.assertNotIn("prompt_registry", source)
        self.assertNotIn("prompt_renderers", source)
        self.assertNotIn("prompt_path", source)

    def test_no_api_provider_or_model_imports_exist(self) -> None:
        source = inspect.getsource(linkedin_post_quality_rubric_contract)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("provider", source.lower())
        self.assertNotIn("model", source.lower())

    def test_no_runtime_django_repair_attempt_or_persistence_imports_exist(self) -> None:
        source = inspect.getsource(linkedin_post_quality_rubric_contract)

        self.assertNotIn("django", source.lower())
        self.assertNotIn("repair", source.lower())
        self.assertNotIn("FinalPostAttempt", source)
        self.assertNotIn("ContentPackage", source)
        self.assertNotIn("generator", source)

    def test_prompt_contract_criteria_match_payload(self) -> None:
        prompt = _prompt_text()
        payload = get_quality_evaluator_rubric_payload()

        for criterion in payload.criteria:
            with self.subTest(criterion=criterion):
                self.assertIn(criterion, prompt)

    def test_prompt_score_ranges_match_payload(self) -> None:
        prompt = _normalized_prompt_text()
        payload = get_quality_evaluator_rubric_payload()

        self.assertIn(
            f"integer score from {payload.score_min} to {payload.score_max}",
            prompt,
        )
        self.assertIn(
            f"each score must be an integer from {payload.score_min} to {payload.score_max}",
            prompt,
        )

    def test_prompt_total_range_matches_payload(self) -> None:
        prompt = _normalized_prompt_text()
        payload = get_quality_evaluator_rubric_payload()

        self.assertIn(
            f"total_score must be an integer from {payload.total_min} to {payload.total_max}",
            prompt,
        )

    def test_prompt_threshold_matches_payload(self) -> None:
        prompt = _normalized_prompt_text()
        payload = get_quality_evaluator_rubric_payload()

        self.assertIn(f"total_score is at least {payload.pass_threshold}", prompt)
        self.assertIn(f"total_score is below {payload.pass_threshold}", prompt)

    def test_prompt_minimums_match_payload(self) -> None:
        prompt = _normalized_prompt_text()
        payload = get_quality_evaluator_rubric_payload()

        for criterion, minimum in payload.required_minimums.items():
            with self.subTest(criterion=criterion):
                self.assertIn(f"{criterion} is at least {minimum}", prompt)

    def test_prompt_automatic_failure_semantics_do_not_contradict_payload(self) -> None:
        prompt = _normalized_prompt_text()

        self.assertIn("use only automatic-failure rules defined", prompt)
        self.assertIn("any documented automatic failure applies", prompt)
        self.assertIn('"pass" must be false when automatic_fail_reason is not an empty string', prompt)

    def test_prompt_and_payload_versions_are_consistent(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(contract.prompt_version, payload.rubric_version)

    def test_decision_controller_threshold_matches_payload(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(QUALITY_PASS_THRESHOLD, payload.pass_threshold)

    def test_decision_controller_required_minimums_match_payload(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(REQUIRED_QUALITY_MINIMUMS, payload.required_minimums)

    def test_review_contract_score_ranges_match_payload(self) -> None:
        payload = get_quality_evaluator_rubric_payload()

        self.assertEqual(CANONICAL_QUALITY_SCORE_KEYS, tuple(payload.criteria))
        self.assertEqual(MIN_CRITERION_SCORE, payload.score_min)
        self.assertEqual(MAX_CRITERION_SCORE, payload.score_max)
        self.assertEqual(MIN_TOTAL_SCORE, payload.total_min)
        self.assertEqual(MAX_TOTAL_SCORE, payload.total_max)

    def test_rubric_contains_no_decision_actions(self) -> None:
        serialized = json.dumps(
            get_quality_evaluator_rubric_payload().to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )

        for action in DECISION_ACTIONS:
            with self.subTest(action=action):
                self.assertNotIn(action, serialized)


def _prompt_path() -> Path:
    contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)
    return Path(__file__).resolve().parents[1] / contract.prompt_path


def _prompt_text() -> str:
    return _prompt_path().read_text(encoding="utf-8")


def _normalized_prompt_text() -> str:
    return re.sub(r"\s+", " ", _prompt_text().lower()).strip()


def _quality_target_path() -> Path:
    payload = get_quality_evaluator_rubric_payload()
    return Path(__file__).resolve().parents[1] / payload.source_document


def _automatic_fail_conditions_from_docs() -> tuple[str, ...]:
    text = _quality_target_path().read_text(encoding="utf-8")
    match = re.search(
        r"## Automatic Fail Conditions\s+"
        r"A post fails automatically if it:\s+"
        r"(?P<body>(?:- .+\n?)+)",
        text,
    )
    if not match:
        raise AssertionError("Automatic Fail Conditions section was not found.")
    return tuple(
        line.removeprefix("- ").removesuffix(";").removesuffix(".")
        for line in match.group("body").splitlines()
        if line.startswith("- ")
    )
