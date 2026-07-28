from __future__ import annotations

import copy
import inspect
import json
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging import linkedin_post_quality_evaluator_parser
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_quality_evaluator_execution import (
    QualityEvaluatorRawResponse,
)
from services.packaging.linkedin_post_quality_evaluator_parser import (
    ERROR_EMPTY_RAW_RESPONSE,
    ERROR_EXECUTION_FAILED,
    ERROR_MALFORMED_FENCE,
    ERROR_MALFORMED_JSON,
    ERROR_NON_OBJECT_JSON,
    ERROR_NORMALIZATION_FAILED,
    QualityEvaluatorResponseParseError,
    parse_and_normalize_quality_evaluator_response,
    parse_quality_evaluator_raw_response,
)


class LinkedInQualityEvaluatorParserTests(SimpleTestCase):
    def test_plain_json_object_parses(self) -> None:
        parsed = parse_quality_evaluator_raw_response(
            _raw_response('{"scores": {"hook": 4}, "notes": ["ok"]}')
        )

        self.assertEqual(parsed["scores"], {"hook": 4})
        self.assertEqual(parsed["notes"], ["ok"])

    def test_json_labelled_fence_parses(self) -> None:
        parsed = parse_quality_evaluator_raw_response(
            _raw_response('```json\n{"scores": {"hook": 4}}\n```')
        )

        self.assertEqual(parsed, {"scores": {"hook": 4}})

    def test_generic_fence_parses(self) -> None:
        parsed = parse_quality_evaluator_raw_response(
            _raw_response('```\n{"scores": {"hook": 4}}\n```')
        )

        self.assertEqual(parsed, {"scores": {"hook": 4}})

    def test_surrounding_response_whitespace_parses(self) -> None:
        parsed = parse_quality_evaluator_raw_response(
            _raw_response('\n\t  {"scores": {"hook": 4}}  \r\n')
        )

        self.assertEqual(parsed["scores"]["hook"], 4)

    def test_whitespace_inside_fence_parses(self) -> None:
        parsed = parse_quality_evaluator_raw_response(
            _raw_response('```json\n\n  {"scores": {"hook": 4}}  \n\n```')
        )

        self.assertEqual(parsed["scores"]["hook"], 4)

    def test_unicode_strings_are_preserved(self) -> None:
        parsed = parse_quality_evaluator_raw_response(
            _raw_response('{"notes": ["\\u0447\\u0435\\u043b\\u043e\\u0432\\u0435\\u0447\\u043d\\u043e"]}')
        )

        self.assertEqual(parsed["notes"], ["человечно"])

    def test_nested_dictionaries_and_lists_parse(self) -> None:
        raw = _raw_response(
            json.dumps(
                {
                    "scores": {"hook": 4},
                    "notes": [{"items": ["a", "b"]}],
                }
            )
        )

        parsed = parse_quality_evaluator_raw_response(raw)

        self.assertEqual(parsed["notes"][0]["items"], ["a", "b"])

    def test_canonical_quality_review_payload_parses(self) -> None:
        parsed = parse_quality_evaluator_raw_response(
            _raw_response(json.dumps(_canonical_review()))
        )

        self.assertEqual(parsed["scores"], _scores())
        self.assertIs(parsed["pass"], True)

    def test_raw_response_remains_unchanged(self) -> None:
        raw = _raw_response(json.dumps(_canonical_review()))
        before = copy.deepcopy(raw)

        parse_quality_evaluator_raw_response(raw)

        self.assertEqual(raw, before)

    def test_parsed_result_is_independent_between_calls(self) -> None:
        raw = _raw_response(json.dumps({"nested": {"items": ["a"]}}))

        first = parse_quality_evaluator_raw_response(raw)
        second = parse_quality_evaluator_raw_response(raw)
        first["nested"]["items"].append("changed")

        self.assertEqual(second["nested"]["items"], ["a"])

    def test_execution_failure_wins_and_skips_json_decode(self) -> None:
        raw = _raw_response('{"scores": {"hook": 4}}', execution_error="provider failed")

        with patch(
            "services.packaging.linkedin_post_quality_evaluator_parser.json.loads"
        ) as mock_loads:
            with self.assertRaisesRegex(
                QualityEvaluatorResponseParseError,
                "execution failed",
            ) as error:
                parse_quality_evaluator_raw_response(raw)

        self.assertEqual(error.exception.code, ERROR_EXECUTION_FAILED)
        mock_loads.assert_not_called()

    def test_execution_failure_message_does_not_include_raw_text_or_metadata(self) -> None:
        raw = _raw_response(
            '{"secret": "candidate text"}',
            execution_error="provider failed",
            provider="openai-sentinel",
            model="model-sentinel",
            usage={"request_id": "usage-sentinel"},
        )

        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(raw)

        message = str(error.exception)
        self.assertNotIn("candidate text", message)
        self.assertNotIn("openai-sentinel", message)
        self.assertNotIn("model-sentinel", message)
        self.assertNotIn("usage-sentinel", message)

    def test_empty_raw_response_fails(self) -> None:
        for raw_text in (None, "", " ", "\t", "\r\n"):
            with self.subTest(raw_text=raw_text):
                with self.assertRaises(QualityEvaluatorResponseParseError) as error:
                    parse_quality_evaluator_raw_response(_raw_response(raw_text))

                self.assertEqual(error.exception.code, ERROR_EMPTY_RAW_RESPONSE)

    def test_unsupported_fence_label_fails(self) -> None:
        for label in ("python", "javascript", "text", "yaml", "markdown"):
            with self.subTest(label=label):
                with self.assertRaises(QualityEvaluatorResponseParseError) as error:
                    parse_quality_evaluator_raw_response(
                        _raw_response(f"```{label}\n{{}}\n```")
                    )

                self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_prose_before_fence_fails(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(_raw_response('Intro\n```json\n{}\n```'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_prose_after_fence_fails(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(
                _raw_response('```json\n{}\n```\nTrailing prose')
            )

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_multiple_fences_fail(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(
                _raw_response('```json\n{}\n```\n```json\n{}\n```')
            )

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_unclosed_fence_fails(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(_raw_response('```json\n{}'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_closing_fence_without_opening_fails(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(_raw_response('{}\n```'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_nested_fence_fails(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(
                _raw_response('```json\n{"text": "ok"}\n```\n```')
            )

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_four_backtick_fence_fails(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(_raw_response('````json\n{}\n````'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_content_after_closing_fence_fails(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(_raw_response('```json\n{}\n``` {}'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_malformed_opening_fence_label_fails(self) -> None:
        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_quality_evaluator_raw_response(_raw_response('``` json\n{}\n```'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_backticks_inside_plain_json_string_do_not_trigger_fence_parsing(self) -> None:
        parsed = parse_quality_evaluator_raw_response(
            _raw_response('{"note": "contains ``` inside string"}')
        )

        self.assertEqual(parsed["note"], "contains ``` inside string")

    def test_malformed_json_fails(self) -> None:
        cases = (
            '{"scores": ',
            '{"scores": {"hook": 4,}}',
            '{"a": 1} {"b": 2}',
            '{"a": 1} trailing',
            'leading {"a": 1}',
        )
        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                with self.assertRaises(QualityEvaluatorResponseParseError) as error:
                    parse_quality_evaluator_raw_response(_raw_response(raw_text))

                self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_non_standard_json_numbers_fail(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with self.assertRaises(QualityEvaluatorResponseParseError) as error:
                    parse_quality_evaluator_raw_response(
                        _raw_response(f'{{"total_score": {constant}}}')
                    )

                self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_non_object_json_fails_before_normalization(self) -> None:
        cases = ("[]", '"review"', "42", "4.2", "true", "null")
        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                with patch(
                    "services.packaging.linkedin_post_quality_evaluator_parser."
                    "normalize_quality_review_result"
                ) as mock_normalize:
                    with self.assertRaises(QualityEvaluatorResponseParseError) as error:
                        parse_quality_evaluator_raw_response(_raw_response(raw_text))

                self.assertEqual(error.exception.code, ERROR_NON_OBJECT_JSON)
                mock_normalize.assert_not_called()

    def test_parse_and_normalize_valid_quality_pass_response(self) -> None:
        normalized = parse_and_normalize_quality_evaluator_response(
            _raw_response(json.dumps(_canonical_review(passed=True, total_score=37)))
        )

        self.assertIs(normalized["pass"], True)
        self.assertEqual(normalized["total_score"], 37)

    def test_parse_and_normalize_valid_quality_fail_response(self) -> None:
        normalized = parse_and_normalize_quality_evaluator_response(
            _raw_response(
                json.dumps(
                    _canonical_review(
                        passed=False,
                        total_score=35,
                        failed_criteria=["human_voice"],
                    )
                )
            )
        )

        self.assertIs(normalized["pass"], False)
        self.assertEqual(normalized["failed_criteria"], ["human_voice"])

    def test_normalization_error_missing_rubric_criterion_is_distinct(self) -> None:
        review = _canonical_review()
        review["scores"].pop("human_voice")

        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_and_normalize_quality_evaluator_response(
                _raw_response(json.dumps(review))
            )

        self.assertEqual(error.exception.code, ERROR_NORMALIZATION_FAILED)
        self.assertIsInstance(error.exception.__cause__, ValueError)

    def test_normalization_error_invalid_score_range_is_distinct(self) -> None:
        review = _canonical_review(scores={**_scores(), "hook": 6})

        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_and_normalize_quality_evaluator_response(
                _raw_response(json.dumps(review))
            )

        self.assertEqual(error.exception.code, ERROR_NORMALIZATION_FAILED)

    def test_normalization_error_invalid_pass_field_is_distinct(self) -> None:
        review = _canonical_review(passed="yes")

        with self.assertRaises(QualityEvaluatorResponseParseError) as error:
            parse_and_normalize_quality_evaluator_response(
                _raw_response(json.dumps(review))
            )

        self.assertEqual(error.exception.code, ERROR_NORMALIZATION_FAILED)

    def test_unknown_top_level_fields_follow_normalizer_policy(self) -> None:
        review = {**_canonical_review(), "audit": "sentinel"}

        normalized = parse_and_normalize_quality_evaluator_response(
            _raw_response(json.dumps(review))
        )

        self.assertNotIn("audit", normalized)

    def test_existing_pass_result_alias_handling_is_reused(self) -> None:
        review = _canonical_review()
        review["pass_result"] = review.pop("pass")

        normalized = parse_and_normalize_quality_evaluator_response(
            _raw_response(json.dumps(review))
        )

        self.assertIs(normalized["pass"], True)
        self.assertNotIn("pass_result", normalized)

    def test_normalizer_is_called_by_adapter_not_reimplemented(self) -> None:
        raw = _raw_response('{"pass": true}')

        with patch(
            "services.packaging.linkedin_post_quality_evaluator_parser."
            "normalize_quality_review_result",
            return_value={"pass": True},
        ) as mock_normalize:
            normalized = parse_and_normalize_quality_evaluator_response(raw)

        self.assertEqual(normalized, {"pass": True})
        mock_normalize.assert_called_once_with({"pass": True})

    def test_audit_metadata_is_not_added_to_parse_or_normalized_result(self) -> None:
        raw = _raw_response(
            json.dumps(_canonical_review()),
            provider="provider-sentinel",
            model="model-sentinel",
            usage={"usage_sentinel": 1},
            raw_provider_response={"raw_sentinel": True},
            prompt_metadata=PromptMetadata(
                prompt_name="prompt-sentinel",
                prompt_version="version-sentinel",
                prompt_path="path-sentinel",
            ),
        )

        parsed = parse_quality_evaluator_raw_response(raw)
        normalized = parse_and_normalize_quality_evaluator_response(raw)

        for result in (parsed, normalized):
            serialized = json.dumps(result, sort_keys=True)
            self.assertNotIn("provider-sentinel", serialized)
            self.assertNotIn("model-sentinel", serialized)
            self.assertNotIn("usage_sentinel", serialized)
            self.assertNotIn("raw_sentinel", serialized)
            self.assertNotIn("prompt-sentinel", serialized)

    def test_mutating_parsed_nested_values_does_not_mutate_raw_response(self) -> None:
        raw = _raw_response(json.dumps({"nested": {"items": ["original"]}}))
        before = copy.deepcopy(raw)

        parsed = parse_quality_evaluator_raw_response(raw)
        parsed["nested"]["items"].append("changed")

        self.assertEqual(raw, before)

    def test_mutating_normalized_output_does_not_mutate_parsed_input(self) -> None:
        raw = _raw_response(json.dumps(_canonical_review(notes=["Original note."])))
        parsed = parse_quality_evaluator_raw_response(raw)

        normalized = parse_and_normalize_quality_evaluator_response(raw)
        normalized["scores"]["hook"] = 1
        normalized["notes"].append("Changed note.")

        self.assertEqual(parsed["scores"]["hook"], 4)
        self.assertEqual(parsed["notes"], ["Original note."])

    def test_parser_module_has_no_provider_decision_repair_or_runtime_dependencies(self) -> None:
        source = inspect.getsource(linkedin_post_quality_evaluator_parser)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("django.conf", source)
        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("ContentPackage", source)
        self.assertNotIn("apps.packaging.models", source)
        self.assertNotIn("FinalPostDecisionController", source)
        self.assertNotIn("TargetedRepairPlan", source)
        self.assertNotIn("run_final_post_deterministic_gate", source)
        self.assertNotIn("linkedin_post_prompt_registry", source)
        self.assertNotIn("render_quality_evaluator_prompt_input", source)

    def test_parser_does_not_duplicate_rubric_or_decision_rules(self) -> None:
        source = inspect.getsource(linkedin_post_quality_evaluator_parser)

        self.assertNotIn("CANONICAL_QUALITY_SCORE_KEYS", source)
        self.assertNotIn("QUALITY_PASS_THRESHOLD", source)
        self.assertNotIn("REQUIRED_QUALITY_MINIMUMS", source)


def _raw_response(
    raw_text,
    *,
    execution_error: str | None = None,
    provider: str = "openai",
    model: str = "quality-model",
    prompt_metadata: PromptMetadata | None = None,
    usage: dict | None = None,
    raw_provider_response: dict | None = None,
) -> QualityEvaluatorRawResponse:
    return QualityEvaluatorRawResponse(
        raw_text=raw_text,
        provider=provider,
        model=model,
        prompt_metadata=prompt_metadata,
        usage=usage,
        raw_provider_response=raw_provider_response,
        execution_error=execution_error,
    )


def _canonical_review(
    *,
    scores: dict | None = None,
    total_score: int = 37,
    passed=True,
    failed_criteria: list[str] | None = None,
    automatic_fail_reason: str = "",
    notes: list[str] | None = None,
) -> dict:
    return {
        "scores": scores or _scores(),
        "total_score": total_score,
        "pass": passed,
        "failed_criteria": failed_criteria or [],
        "automatic_fail_reason": automatic_fail_reason,
        "notes": notes or ["Ready."],
    }


def _scores() -> dict[str, int]:
    return {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 4,
        "author_point_of_view": 4,
        "human_voice": 5,
        "practical_value": 4,
        "cta": 4,
    }
