from __future__ import annotations

import copy
import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_quality_review_contract
from services.packaging.linkedin_final_post_diagnostics import diagnose_final_post_payload
from services.packaging.linkedin_post_flow_contracts import ACTION_ACCEPT
from services.packaging.linkedin_post_flow_contracts import FinalPostAttemptHistory
from services.packaging.linkedin_post_flow_decision import FinalPostDecisionController
from services.packaging.linkedin_post_flow_handoffs import (
    QualityReviewResult as HandoffQualityReviewResult,
)
from services.packaging.linkedin_post_pipeline import (
    QualityReviewResult as PipelineQualityReviewResult,
)
from services.packaging.linkedin_post_quality_review_contract import (
    CANONICAL_QUALITY_SCORE_KEYS,
    normalize_quality_review_result,
)


class LinkedInPostQualityReviewContractTests(SimpleTestCase):
    def test_handoff_quality_review_result_normalizes_correctly(self) -> None:
        review = _handoff_review()

        normalized = normalize_quality_review_result(review)

        self.assertEqual(normalized["scores"], _scores())
        self.assertEqual(normalized["total_score"], 37)
        self.assertIs(normalized["pass"], True)
        self.assertEqual(normalized["failed_criteria"], [])
        self.assertEqual(normalized["automatic_fail_reason"], "")
        self.assertEqual(normalized["notes"], ["Ready."])

    def test_canonical_dict_normalizes_correctly(self) -> None:
        review = _canonical_review(notes=["Readable."])

        normalized = normalize_quality_review_result(review)

        self.assertEqual(normalized, {**review, "scores": _scores()})
        self.assertIsNot(normalized, review)

    def test_legacy_pipeline_pass_result_converts_to_pass(self) -> None:
        review = PipelineQualityReviewResult(
            scores=_scores(),
            total_score=35,
            pass_result=False,
            failed_criteria=["human_voice"],
            automatic_fail_reason="",
        )

        normalized = normalize_quality_review_result(review)

        self.assertIs(normalized["pass"], False)
        self.assertNotIn("pass_result", normalized)

    def test_output_contains_pass_field(self) -> None:
        normalized = normalize_quality_review_result(_canonical_review())

        self.assertIn("pass", normalized)

    def test_output_does_not_contain_pass_result(self) -> None:
        normalized = normalize_quality_review_result(_legacy_review_dict())

        self.assertNotIn("pass_result", normalized)

    def test_output_does_not_contain_passed(self) -> None:
        normalized = normalize_quality_review_result(_canonical_review())

        self.assertNotIn("passed", normalized)

    def test_exact_required_score_keys_are_preserved(self) -> None:
        normalized = normalize_quality_review_result(_canonical_review())

        self.assertEqual(tuple(normalized["scores"]), CANONICAL_QUALITY_SCORE_KEYS)
        self.assertEqual(set(normalized["scores"]), set(CANONICAL_QUALITY_SCORE_KEYS))

    def test_missing_human_voice_fails(self) -> None:
        review = _canonical_review(scores={key: value for key, value in _scores().items() if key != "human_voice"})

        with self.assertRaisesRegex(ValueError, "missing criteria: \\['human_voice'\\]"):
            normalize_quality_review_result(review)

    def test_missing_required_score_key_fails(self) -> None:
        review = _canonical_review(scores={key: value for key, value in _scores().items() if key != "hook"})

        with self.assertRaisesRegex(ValueError, "missing criteria: \\['hook'\\]"):
            normalize_quality_review_result(review)

    def test_unexpected_score_key_fails(self) -> None:
        review = _canonical_review(scores={**_scores(), "reader_relevance": 4})

        with self.assertRaisesRegex(ValueError, "unexpected criteria: \\['reader_relevance'\\]"):
            normalize_quality_review_result(review)

    def test_integer_score_is_accepted(self) -> None:
        normalized = normalize_quality_review_result(
            _canonical_review(scores={**_scores(), "hook": 5})
        )

        self.assertEqual(normalized["scores"]["hook"], 5)

    def test_boolean_score_fails_and_identifies_criterion(self) -> None:
        with self.assertRaisesRegex(ValueError, "score 'hook' must be an integer"):
            normalize_quality_review_result(
                _canonical_review(scores={**_scores(), "hook": True})
            )

    def test_string_score_fails_and_identifies_criterion(self) -> None:
        with self.assertRaisesRegex(ValueError, "score 'hook' must be an integer"):
            normalize_quality_review_result(
                _canonical_review(scores={**_scores(), "hook": "4"})
            )

    def test_none_score_fails_and_identifies_criterion(self) -> None:
        with self.assertRaisesRegex(ValueError, "score 'hook' must be an integer"):
            normalize_quality_review_result(
                _canonical_review(scores={**_scores(), "hook": None})
            )

    def test_float_score_fails_and_identifies_criterion(self) -> None:
        with self.assertRaisesRegex(ValueError, "score 'hook' must be an integer"):
            normalize_quality_review_result(
                _canonical_review(scores={**_scores(), "hook": 4.0})
            )

    def test_container_score_fails_and_identifies_criterion(self) -> None:
        with self.assertRaisesRegex(ValueError, "score 'hook' must be an integer"):
            normalize_quality_review_result(
                _canonical_review(scores={**_scores(), "hook": [4]})
            )

    def test_score_below_documented_range_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "score 'hook' must be between 1 and 5"):
            normalize_quality_review_result(
                _canonical_review(scores={**_scores(), "hook": 0})
            )

    def test_score_above_documented_range_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "score 'hook' must be between 1 and 5"):
            normalize_quality_review_result(
                _canonical_review(scores={**_scores(), "hook": 6})
            )

    def test_invalid_score_in_legacy_object_fails(self) -> None:
        review = PipelineQualityReviewResult(
            scores={**_scores(), "hook": "4"},
            total_score=37,
            pass_result=True,
            failed_criteria=[],
            automatic_fail_reason="",
        )

        with self.assertRaisesRegex(ValueError, "score 'hook' must be an integer"):
            normalize_quality_review_result(review)

    def test_invalid_score_in_canonical_dict_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "score 'human_voice' must be an integer"):
            normalize_quality_review_result(
                _canonical_review(scores={**_scores(), "human_voice": {}})
            )

    def test_total_score_is_preserved_exactly(self) -> None:
        normalized = normalize_quality_review_result(_canonical_review(total_score=36))

        self.assertEqual(normalized["total_score"], 36)

    def test_total_score_is_not_recomputed(self) -> None:
        normalized = normalize_quality_review_result(
            _canonical_review(scores={key: 5 for key in CANONICAL_QUALITY_SCORE_KEYS}, total_score=36)
        )

        self.assertEqual(normalized["total_score"], 36)

    def test_boolean_total_score_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_score must be an integer"):
            normalize_quality_review_result(_canonical_review(total_score=True))

    def test_string_total_score_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_score must be an integer"):
            normalize_quality_review_result(_canonical_review(total_score="37"))

    def test_none_total_score_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_score must be an integer"):
            normalize_quality_review_result(_canonical_review(total_score=None))

    def test_float_total_score_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_score must be an integer"):
            normalize_quality_review_result(_canonical_review(total_score=36.5))

    def test_container_total_score_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_score must be an integer"):
            normalize_quality_review_result(_canonical_review(total_score={"score": 37}))

    def test_total_score_below_documented_range_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_score must be between 9 and 45"):
            normalize_quality_review_result(_canonical_review(total_score=8))

    def test_total_score_above_documented_range_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_score must be between 9 and 45"):
            normalize_quality_review_result(_canonical_review(total_score=46))

    def test_pass_flag_is_preserved_exactly(self) -> None:
        normalized = normalize_quality_review_result(_canonical_review(passed=False))

        self.assertIs(normalized["pass"], False)

    def test_failed_criteria_are_preserved(self) -> None:
        normalized = normalize_quality_review_result(
            _canonical_review(failed_criteria=("hook", "human_voice"))
        )

        self.assertEqual(normalized["failed_criteria"], ["hook", "human_voice"])

    def test_automatic_fail_reason_is_preserved(self) -> None:
        normalized = normalize_quality_review_result(
            _canonical_review(automatic_fail_reason="unsupported claim")
        )

        self.assertEqual(normalized["automatic_fail_reason"], "unsupported claim")

    def test_notes_are_preserved_when_supplied(self) -> None:
        normalized = normalize_quality_review_result(
            _canonical_review(notes=("Human voice is specific.",))
        )

        self.assertEqual(normalized["notes"], ["Human voice is specific."])

    def test_human_review_flags_are_preserved_when_supplied(self) -> None:
        normalized = normalize_quality_review_result(
            _canonical_review(
                requires_human_review=True,
                human_review_reason="Sensitive factual ambiguity.",
                blocking_factuality_ambiguity=True,
                unsupported_claims_cannot_be_safely_repaired=False,
                sensitive_topic_risk=True,
            )
        )

        self.assertTrue(normalized["requires_human_review"])
        self.assertEqual(normalized["human_review_reason"], "Sensitive factual ambiguity.")
        self.assertTrue(normalized["blocking_factuality_ambiguity"])
        self.assertFalse(normalized["unsupported_claims_cannot_be_safely_repaired"])
        self.assertTrue(normalized["sensitive_topic_risk"])

    def test_missing_pass_field_fails(self) -> None:
        review = _canonical_review()
        review.pop("pass")

        with self.assertRaisesRegex(ValueError, "missing pass result field"):
            normalize_quality_review_result(review)

    def test_conflicting_pass_aliases_fail(self) -> None:
        review = {**_canonical_review(), "pass_result": True}

        with self.assertRaisesRegex(ValueError, "conflicting pass fields: pass, pass_result"):
            normalize_quality_review_result(review)

    def test_conflicting_pass_and_passed_aliases_fail(self) -> None:
        review = {**_canonical_review(), "passed": True}

        with self.assertRaisesRegex(ValueError, "conflicting pass fields: pass, passed"):
            normalize_quality_review_result(review)

    def test_conflicting_pass_result_and_passed_aliases_fail(self) -> None:
        review = _legacy_review_dict()
        review["passed"] = False

        with self.assertRaisesRegex(ValueError, "conflicting pass fields: pass_result, passed"):
            normalize_quality_review_result(review)

    def test_conflicting_all_pass_aliases_fail(self) -> None:
        review = {**_canonical_review(), "pass_result": True, "passed": True}

        with self.assertRaisesRegex(ValueError, "conflicting pass fields: pass, pass_result, passed"):
            normalize_quality_review_result(review)

    def test_passed_only_dict_fails(self) -> None:
        review = _canonical_review()
        review["passed"] = review.pop("pass")

        with self.assertRaisesRegex(ValueError, "field 'passed' is not a supported input shape"):
            normalize_quality_review_result(review)

    def test_unsupported_input_type_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported QualityReviewResult or dictionary"):
            normalize_quality_review_result(object())

    def test_malformed_scores_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "scores must be a dictionary"):
            normalize_quality_review_result(_canonical_review(scores=["hook"]))

    def test_malformed_failed_criteria_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "failed_criteria must be a list or tuple"):
            normalize_quality_review_result(_canonical_review(failed_criteria="hook"))

    def test_malformed_automatic_fail_reason_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "automatic_fail_reason must be a string"):
            normalize_quality_review_result(_canonical_review(automatic_fail_reason=None))

    def test_invalid_optional_human_review_fields_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires_human_review must be a boolean"):
            normalize_quality_review_result(_canonical_review(requires_human_review="yes"))

    def test_input_object_is_not_mutated(self) -> None:
        review = _handoff_review()
        before = copy.deepcopy(review)

        normalize_quality_review_result(review)

        self.assertEqual(review, before)

    def test_input_dict_and_nested_values_are_not_mutated(self) -> None:
        review = _canonical_review(
            scores=_scores(),
            failed_criteria=["human_voice"],
            notes=["Original note."],
        )
        before = copy.deepcopy(review)

        normalized = normalize_quality_review_result(review)
        normalized["scores"]["human_voice"] = 1
        normalized["failed_criteria"].append("hook")
        normalized["notes"].append("Changed note.")

        self.assertEqual(review, before)

    def test_failed_validation_does_not_mutate_input_dict_or_nested_values(self) -> None:
        review = _canonical_review(
            scores={**_scores(), "hook": "4"},
            failed_criteria=["hook"],
            notes=["Original note."],
        )
        before = copy.deepcopy(review)

        with self.assertRaises(ValueError):
            normalize_quality_review_result(review)

        self.assertEqual(review, before)

    def test_output_is_json_serializable(self) -> None:
        normalized = normalize_quality_review_result(
            _canonical_review(notes=("Ready.",), requires_human_review=False)
        )

        serialized = json.dumps(normalized, sort_keys=True)

        self.assertIn("human_voice", serialized)

    def test_output_is_accepted_by_existing_decision_controller(self) -> None:
        normalized = normalize_quality_review_result(_canonical_review())

        decision = FinalPostDecisionController().decide(
            validation_passed=True,
            validation_error="",
            diagnostics=diagnose_final_post_payload(
                _payload(),
                selected_evidence_ids=["a0-summary"],
                schema_validation_passed=True,
            ),
            quality_review=normalized,
            attempt_history=FinalPostAttemptHistory(attempts=[]),
        )

        self.assertEqual(decision.action, ACTION_ACCEPT)

    def test_contract_module_does_not_import_execution_runtime_or_repair_layers(self) -> None:
        source = inspect.getsource(linkedin_post_quality_review_contract)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)
        self.assertNotIn("ContentPackage", source)
        self.assertNotIn("django.db", source)
        self.assertNotIn("TargetedRepairPlan", source)
        self.assertNotIn("RepairAgent", source)
        self.assertNotIn("run_final_post_deterministic_gate", source)
        self.assertNotIn("prepare_final_post_decision_ready_result", source)


def _handoff_review() -> HandoffQualityReviewResult:
    return HandoffQualityReviewResult(
        scores=_scores(),
        total_score=37,
        passed=True,
        failed_criteria=(),
        automatic_fail_reason="",
        notes=("Ready.",),
    )


def _legacy_review_dict() -> dict:
    return {
        "scores": _scores(),
        "total_score": 35,
        "pass_result": False,
        "failed_criteria": ["human_voice"],
        "automatic_fail_reason": "",
    }


def _canonical_review(**overrides) -> dict:
    review = {
        "scores": _scores(),
        "total_score": 37,
        "pass": True,
        "failed_criteria": [],
        "automatic_fail_reason": "",
    }
    review.update(overrides)
    if "passed" in review:
        review["pass"] = review.pop("passed")
    return review


def _scores() -> dict[str, int]:
    return {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 3,
        "author_point_of_view": 4,
        "human_voice": 4,
        "practical_value": 4,
        "cta": 4,
    }


def _payload() -> dict:
    return {
        "post_text": "Final post text.",
        "hook_variants": ["Hook one", "Hook two", "Hook three"],
        "cta_variants": ["CTA one", "CTA two", "CTA three"],
        "hashtags": ["#FutureOfWork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }
