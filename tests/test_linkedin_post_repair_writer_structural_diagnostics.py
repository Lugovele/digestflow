from __future__ import annotations

import json
from types import SimpleNamespace

from django.test import SimpleTestCase

from services.packaging.linkedin_post_candidate_writer_output_adapter import (
    CandidateWriterOutputAdaptationError,
    adapt_candidate_writer_payload,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_repair_writer_structural_diagnostics import (
    CANDIDATE_POST_OTHER_VALIDATION_FAILURE,
    COMPLETE_FENCED_JSON,
    COMPLETE_PLAIN_JSON,
    MALFORMED_FENCE_NOT_TRUNCATED,
    NON_JSON_RESPONSE,
    POST_TEXT_BLANK,
    POST_TEXT_EMPTY,
    POST_TEXT_MISSING,
    POST_TEXT_TOO_LONG,
    POST_TEXT_WRONG_TYPE,
    TRUNCATED_AFTER_JSON_BEFORE_FENCE,
    TRUNCATED_INSIDE_JSON,
    UNKNOWN,
    build_repair_writer_response_structure_diagnostics,
    build_repair_writer_structural_diagnostics,
)


class RepairWriterStructuralDiagnosticsTests(SimpleTestCase):
    def test_valid_candidate_post_shape_has_unknown_failure_category(self) -> None:
        diagnostics = build_repair_writer_structural_diagnostics(
            parsed_repair_candidate={"post_text": "valid text"},
            repair_raw_response=_raw_response('{"post_text":"valid text"}'),
        )

        self.assertEqual(diagnostics.structural_failure_category, UNKNOWN)
        self.assertEqual(diagnostics.parsed_top_level_type, "dict")
        self.assertEqual(diagnostics.parsed_top_level_keys, ("post_text",))
        self.assertTrue(diagnostics.post_text_present)
        self.assertEqual(diagnostics.post_text_type, "str")
        self.assertEqual(diagnostics.post_text_character_count, len("valid text"))
        self.assertFalse(diagnostics.post_text_is_empty)
        self.assertFalse(diagnostics.post_text_is_blank)
        self.assertTrue(diagnostics.post_text_within_candidate_max_length)
        self.assertEqual(
            diagnostics.candidate_post_max_length,
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
        )
        self.assertTrue(diagnostics.starts_with_json_object)
        self.assertTrue(diagnostics.ends_with_json_object)
        self.assertFalse(diagnostics.starts_with_code_fence)
        self.assertFalse(diagnostics.ends_with_code_fence)

    def test_missing_post_text_is_classified(self) -> None:
        diagnostics = _diagnostics_for_invalid({})

        self.assertEqual(diagnostics.structural_failure_category, POST_TEXT_MISSING)
        self.assertFalse(diagnostics.post_text_present)
        self.assertIsNone(diagnostics.post_text_type)
        self.assertEqual(diagnostics.candidate_validation_error_field, "post_text")

    def test_wrong_type_post_text_is_classified(self) -> None:
        diagnostics = _diagnostics_for_invalid({"post_text": 123})

        self.assertEqual(diagnostics.structural_failure_category, POST_TEXT_WRONG_TYPE)
        self.assertEqual(diagnostics.post_text_type, "int")
        self.assertFalse(diagnostics.post_text_is_string)

    def test_empty_post_text_is_classified(self) -> None:
        diagnostics = _diagnostics_for_invalid({"post_text": ""})

        self.assertEqual(diagnostics.structural_failure_category, POST_TEXT_EMPTY)
        self.assertEqual(diagnostics.post_text_character_count, 0)
        self.assertTrue(diagnostics.post_text_is_empty)
        self.assertFalse(diagnostics.post_text_is_blank)

    def test_blank_post_text_is_classified(self) -> None:
        diagnostics = _diagnostics_for_invalid({"post_text": "   "})

        self.assertEqual(diagnostics.structural_failure_category, POST_TEXT_BLANK)
        self.assertEqual(diagnostics.post_text_character_count, 3)
        self.assertFalse(diagnostics.post_text_is_empty)
        self.assertTrue(diagnostics.post_text_is_blank)

    def test_overlength_post_text_is_classified_without_value_leakage(self) -> None:
        overlength = "x" * (FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1)

        diagnostics = _diagnostics_for_invalid({"post_text": overlength})
        serialized = json.dumps(diagnostics.to_dict(), sort_keys=True)

        self.assertEqual(diagnostics.structural_failure_category, POST_TEXT_TOO_LONG)
        self.assertEqual(
            diagnostics.post_text_character_count,
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1,
        )
        self.assertFalse(diagnostics.post_text_within_candidate_max_length)
        self.assertNotIn(overlength, serialized)

    def test_extra_field_is_other_validation_failure(self) -> None:
        diagnostics = _diagnostics_for_invalid(
            {"post_text": "valid text", "extra": "x"}
        )

        self.assertEqual(
            diagnostics.structural_failure_category,
            CANDIDATE_POST_OTHER_VALIDATION_FAILURE,
        )
        self.assertEqual(diagnostics.parsed_top_level_keys, ("extra", "post_text"))

    def test_non_dict_parsed_value_is_other_validation_failure(self) -> None:
        diagnostics = _diagnostics_for_invalid(["not", "a", "dict"])

        self.assertEqual(
            diagnostics.structural_failure_category,
            CANDIDATE_POST_OTHER_VALIDATION_FAILURE,
        )
        self.assertEqual(diagnostics.parsed_top_level_type, "list")
        self.assertEqual(diagnostics.parsed_top_level_keys, ())

    def test_raw_response_fence_shape_is_bounded(self) -> None:
        diagnostics = build_repair_writer_structural_diagnostics(
            parsed_repair_candidate={"post_text": "valid text"},
            repair_raw_response=_raw_response('```json\n{"post_text":"valid text"}\n```'),
        )

        self.assertEqual(
            diagnostics.raw_response_character_count,
            len('```json\n{"post_text":"valid text"}\n```'),
        )
        self.assertTrue(diagnostics.starts_with_code_fence)
        self.assertTrue(diagnostics.ends_with_code_fence)
        self.assertFalse(diagnostics.starts_with_json_object)
        self.assertFalse(diagnostics.ends_with_json_object)
        self.assertEqual(diagnostics.opening_fence_language, "json")
        self.assertEqual(diagnostics.markdown_fence_count, 2)
        self.assertTrue(diagnostics.contains_json_object_start_after_fence)
        self.assertTrue(diagnostics.contains_json_object_end)
        self.assertEqual(diagnostics.json_brace_balance, 0)
        self.assertFalse(diagnostics.json_string_appears_unterminated)
        self.assertEqual(
            diagnostics.response_structure_classification,
            COMPLETE_FENCED_JSON,
        )

    def test_response_shape_classifies_complete_plain_json(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '{"post_text":"ok"}'
        )

        self.assertEqual(
            diagnostics["response_structure_classification"],
            COMPLETE_PLAIN_JSON,
        )
        self.assertTrue(diagnostics["starts_with_json_object"])
        self.assertTrue(diagnostics["ends_with_json_object"])

    def test_response_shape_classifies_complete_json_fence(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '```json\n{"post_text":"ok"}\n```'
        )

        self.assertEqual(
            diagnostics["response_structure_classification"],
            COMPLETE_FENCED_JSON,
        )
        self.assertEqual(diagnostics["opening_fence_language"], "json")

    def test_response_shape_classifies_complete_generic_fence(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '```\n{"post_text":"ok"}\n```'
        )

        self.assertEqual(
            diagnostics["response_structure_classification"],
            COMPLETE_FENCED_JSON,
        )
        self.assertEqual(diagnostics["opening_fence_language"], "")

    def test_response_shape_classifies_truncated_inside_json(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '```json\n{"post_text":"unfinished',
            provider_output_limit_reached=True,
        )

        self.assertEqual(
            diagnostics["response_structure_classification"],
            TRUNCATED_INSIDE_JSON,
        )
        self.assertTrue(diagnostics["json_string_appears_unterminated"])

    def test_response_shape_classifies_truncated_after_json_before_fence(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '```json\n{"post_text":"ok"}',
            provider_output_limit_reached=True,
        )

        self.assertEqual(
            diagnostics["response_structure_classification"],
            TRUNCATED_AFTER_JSON_BEFORE_FENCE,
        )
        self.assertTrue(diagnostics["contains_json_object_end"])

    def test_response_shape_rejects_complete_json_with_missing_fence_without_limit(
        self,
    ) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '```json\n{"post_text":"ok"}',
            provider_output_limit_reached=False,
        )

        self.assertEqual(
            diagnostics["response_structure_classification"],
            MALFORMED_FENCE_NOT_TRUNCATED,
        )

    def test_response_shape_classifies_malformed_opening_fence(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '``` json\n{"post_text":"ok"}\n```'
        )

        self.assertEqual(diagnostics["opening_fence_language"], "other")
        self.assertEqual(
            diagnostics["response_structure_classification"],
            MALFORMED_FENCE_NOT_TRUNCATED,
        )

    def test_response_shape_does_not_expose_malformed_fence_language(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '```Here is secret provider prose\n{"post_text":"ok"}\n```'
        )
        serialized = json.dumps(diagnostics, sort_keys=True)

        self.assertEqual(diagnostics["opening_fence_language"], "other")
        self.assertNotIn("Here is secret provider prose", serialized)
        self.assertNotIn("secret provider", serialized)

    def test_response_shape_classifies_non_json_prose(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            "Here is the repaired post."
        )

        self.assertEqual(
            diagnostics["response_structure_classification"],
            NON_JSON_RESPONSE,
        )

    def test_response_shape_rejects_multiple_fenced_blocks(self) -> None:
        diagnostics = build_repair_writer_response_structure_diagnostics(
            '```json\n{"post_text":"one"}\n```\n```json\n{"post_text":"two"}\n```'
        )

        self.assertEqual(diagnostics["markdown_fence_count"], 4)
        self.assertEqual(
            diagnostics["response_structure_classification"],
            MALFORMED_FENCE_NOT_TRUNCATED,
        )

    def test_response_shape_excerpts_are_bounded_and_masked(self) -> None:
        secret = "secret repaired post " * 20
        diagnostics = build_repair_writer_response_structure_diagnostics(
            json.dumps({"post_text": secret})
        )
        serialized = json.dumps(diagnostics, sort_keys=True)

        self.assertLessEqual(len(diagnostics["raw_response_prefix_excerpt"]), 80)
        self.assertLessEqual(len(diagnostics["raw_response_suffix_excerpt"]), 120)
        self.assertNotIn(secret, serialized)
        self.assertNotIn("secret repaired post", serialized)

    def test_serialized_diagnostics_do_not_include_raw_values_or_provider_payloads(
        self,
    ) -> None:
        secret_text = "secret repaired post"
        diagnostics = build_repair_writer_structural_diagnostics(
            parsed_repair_candidate={
                "post_text": secret_text,
                "api_key": "secret-key",
            },
            adaptation_error=_adaptation_error_for(
                {"post_text": secret_text, "api_key": "secret-key"}
            ),
            repair_raw_response=_raw_response(
                '{"post_text":"secret repaired post","api_key":"secret-key"}'
            ),
        )

        serialized = json.dumps(diagnostics.to_dict(), sort_keys=True)

        self.assertNotIn(secret_text, serialized)
        self.assertNotIn("secret-key", serialized)
        self.assertNotIn("api_key", serialized)


def _diagnostics_for_invalid(value):
    return build_repair_writer_structural_diagnostics(
        parsed_repair_candidate=value,
        adaptation_error=_adaptation_error_for(value),
        repair_raw_response=_raw_response(json.dumps(value)),
    )


def _adaptation_error_for(value) -> CandidateWriterOutputAdaptationError:
    try:
        adapt_candidate_writer_payload(value)
    except CandidateWriterOutputAdaptationError as exc:
        return exc
    raise AssertionError("expected adaptation error")


def _raw_response(text: str):
    return SimpleNamespace(raw_text=text)
