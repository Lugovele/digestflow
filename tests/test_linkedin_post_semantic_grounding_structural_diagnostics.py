from __future__ import annotations

import json
from types import SimpleNamespace

from django.test import SimpleTestCase

from services.packaging.linkedin_post_semantic_grounding_structural_diagnostics import (
    COMPLETE_FENCED_JSON,
    COMPLETE_PLAIN_JSON,
    EXTRA_PROSE_AROUND_JSON,
    MALFORMED_FENCE_NOT_TRUNCATED,
    MULTIPLE_JSON_BLOCKS,
    NON_JSON_RESPONSE,
    TRUNCATED_AFTER_JSON_BEFORE_FENCE,
    TRUNCATED_INSIDE_JSON,
    build_semantic_grounding_raw_response_diagnostics,
    build_semantic_grounding_response_structure_diagnostics,
)


class SemanticGroundingStructuralDiagnosticsTests(SimpleTestCase):
    def test_complete_plain_json_is_classified_without_parsing_semantics(self) -> None:
        diagnostics = build_semantic_grounding_response_structure_diagnostics(
            '{"pass":true}'
        )

        self.assertEqual(
            diagnostics.response_structure_classification,
            COMPLETE_PLAIN_JSON,
        )
        self.assertTrue(diagnostics.starts_with_json_object)
        self.assertTrue(diagnostics.ends_with_json_object)

    def test_complete_json_fence_is_classified(self) -> None:
        diagnostics = build_semantic_grounding_response_structure_diagnostics(
            '```json\n{"pass":true}\n```'
        )

        self.assertEqual(
            diagnostics.response_structure_classification,
            COMPLETE_FENCED_JSON,
        )
        self.assertTrue(diagnostics.starts_with_code_fence)
        self.assertEqual(diagnostics.opening_fence_language, "json")

    def test_complete_generic_fence_is_classified(self) -> None:
        diagnostics = build_semantic_grounding_response_structure_diagnostics(
            '```\n{"pass":true}\n```'
        )

        self.assertEqual(
            diagnostics.response_structure_classification,
            COMPLETE_FENCED_JSON,
        )
        self.assertEqual(diagnostics.opening_fence_language, "")

    def test_malformed_opening_fence_is_not_treated_as_complete_json(self) -> None:
        diagnostics = build_semantic_grounding_response_structure_diagnostics(
            '```javascript\n{"pass":true}\n```'
        )

        self.assertEqual(
            diagnostics.response_structure_classification,
            MALFORMED_FENCE_NOT_TRUNCATED,
        )
        self.assertEqual(diagnostics.opening_fence_language, "other")

    def test_missing_closing_fence_without_output_limit_is_malformed(self) -> None:
        diagnostics = build_semantic_grounding_response_structure_diagnostics(
            '```json\n{"pass":true}'
        )

        self.assertEqual(
            diagnostics.response_structure_classification,
            MALFORMED_FENCE_NOT_TRUNCATED,
        )
        self.assertFalse(diagnostics.ends_with_code_fence)

    def test_complete_json_missing_closing_fence_with_output_limit_is_truncated_after_json(self) -> None:
        diagnostics = build_semantic_grounding_response_structure_diagnostics(
            '```json\n{"pass":true}\n',
            provider_output_limit_reached=True,
        )

        self.assertEqual(
            diagnostics.response_structure_classification,
            TRUNCATED_AFTER_JSON_BEFORE_FENCE,
        )
        self.assertTrue(diagnostics.contains_json_object_end)

    def test_incomplete_json_inside_fence_with_output_limit_is_truncated_inside_json(self) -> None:
        diagnostics = build_semantic_grounding_response_structure_diagnostics(
            '```json\n{"pass": tru',
            provider_output_limit_reached=True,
        )

        self.assertEqual(
            diagnostics.response_structure_classification,
            TRUNCATED_INSIDE_JSON,
        )
        self.assertGreater(diagnostics.json_brace_balance, 0)

    def test_extra_prose_around_json_is_classified(self) -> None:
        for raw_text in ('Here is JSON {"pass":true}', '{"pass":true} trailing'):
            with self.subTest(raw_text=raw_text):
                diagnostics = build_semantic_grounding_response_structure_diagnostics(
                    raw_text
                )
                self.assertEqual(
                    diagnostics.response_structure_classification,
                    EXTRA_PROSE_AROUND_JSON,
                )

    def test_multiple_json_blocks_are_classified(self) -> None:
        diagnostics = build_semantic_grounding_response_structure_diagnostics(
            '{"pass":true}{"pass":false}'
        )

        self.assertEqual(
            diagnostics.response_structure_classification,
            MULTIPLE_JSON_BLOCKS,
        )

    def test_empty_and_whitespace_response_are_non_json(self) -> None:
        for raw_text in ("", "   \n"):
            with self.subTest(raw_text=raw_text):
                diagnostics = build_semantic_grounding_response_structure_diagnostics(
                    raw_text
                )
                self.assertEqual(
                    diagnostics.response_structure_classification,
                    NON_JSON_RESPONSE,
                )

    def test_raw_response_diagnostics_are_bounded_and_do_not_include_secret_text(self) -> None:
        raw_text = json.dumps(
            {
                "claim_text": "secret live claim content should never appear",
                "rationale": "another secret phrase",
            }
        )
        diagnostics = build_semantic_grounding_raw_response_diagnostics(
            SimpleNamespace(
                raw_text=raw_text,
                provider_response_metadata={
                    "provider_finish_reason": "length",
                    "provider_stop_reason": None,
                    "provider_max_output_tokens": 4800,
                    "provider_output_limit_reached": True,
                    "provider_prompt_tokens": 2000,
                    "provider_visible_output_tokens": 100,
                    "provider_hidden_output_tokens": 4600,
                    "provider_combined_output_tokens": 4700,
                    "provider_output_budget_utilization_percent": 97.916,
                    "raw_provider_response": {"secret": "not allowed"},
                },
                usage={
                    "prompt_tokens": 2000,
                    "completion_tokens": 100,
                    "total_tokens": 2100,
                    "raw": "not allowed",
                },
                execution_diagnostics={
                    "provider_error_category": "rate_limit",
                    "provider_http_status": 429,
                    "raw_prompt": "secret prompt text",
                    "raw_provider_message": "secret provider message",
                    "provider_body": {"raw": "secret body"},
                    "response_text": "secret raw response text",
                    "provider_error_message_safe": "safe provider failure",
                },
            )
        )

        serialized = json.dumps(diagnostics, sort_keys=True)
        self.assertNotIn("secret live claim content", serialized)
        self.assertNotIn("another secret phrase", serialized)
        self.assertNotIn("secret prompt text", serialized)
        self.assertNotIn("raw_provider_response", serialized)
        self.assertLessEqual(len(diagnostics["raw_response_prefix_excerpt"]), 80)
        self.assertLessEqual(len(diagnostics["raw_response_suffix_excerpt"]), 120)
        self.assertEqual(diagnostics["provider_finish_reason"], "length")
        self.assertEqual(diagnostics["provider_max_output_tokens"], 4800)
        self.assertTrue(diagnostics["provider_output_limit_reached"])
        self.assertEqual(diagnostics["provider_visible_output_tokens"], 100)
        self.assertEqual(diagnostics["provider_hidden_output_tokens"], 4600)
        self.assertEqual(diagnostics["provider_combined_output_tokens"], 4700)
        self.assertEqual(
            diagnostics["provider_output_budget_utilization_percent"],
            97.92,
        )
        self.assertEqual(
            diagnostics["usage"],
            {
                "prompt_tokens": 2000,
                "completion_tokens": 100,
                "total_tokens": 2100,
            },
        )
        self.assertEqual(
            diagnostics["provider_error_diagnostics"]["provider_error_message_safe"],
            "safe provider failure",
        )
        self.assertNotIn("raw_prompt", diagnostics["provider_error_diagnostics"])
        self.assertNotIn("raw_provider_message", diagnostics["provider_error_diagnostics"])
        self.assertNotIn("provider_body", diagnostics["provider_error_diagnostics"])
        self.assertNotIn("response_text", diagnostics["provider_error_diagnostics"])
