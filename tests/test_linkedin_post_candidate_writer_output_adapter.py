from __future__ import annotations

import ast
import inspect
import json
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging import linkedin_post_candidate_writer_output_adapter
from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterRawResponse,
)
from services.packaging.linkedin_post_candidate_writer_output_adapter import (
    CANONICAL_CANDIDATE_POST_FIELDS,
    ERROR_INVALID_CANDIDATE_POST,
    ERROR_INVALID_PARSED_CANDIDATE,
    ERROR_MISSING_REQUIRED_FIELD,
    ERROR_RAW_RESPONSE_EXECUTION_ERROR,
    CandidateWriterOutputAdaptationError,
    adapt_candidate_writer_payload,
    build_candidate_writer_output_from_parsed_response,
)
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    ADAPTER_ERROR_INVALID_FIELD_TYPES,
    ADAPTER_ERROR_INVALID_FIELD_VALUES,
    ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
    ADAPTER_ERROR_PAYLOAD_CONTRACT_VIOLATION,
    ADAPTER_ERROR_TOP_LEVEL_NOT_OBJECT,
    FIELD_VIOLATION_ABOVE_MAX_LENGTH,
    FIELD_VIOLATION_EMPTY_STRING,
    FIELD_VIOLATION_WRONG_TYPE,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput


class CandidateWriterOutputAdapterTests(SimpleTestCase):
    def test_adapt_candidate_writer_payload_returns_core_candidate_post_dict(self) -> None:
        payload = adapt_candidate_writer_payload(_parsed_candidate())

        self.assertEqual(tuple(payload), CANONICAL_CANDIDATE_POST_FIELDS)
        self.assertEqual(payload, {"post_text": "Clear final post text."})

    def test_required_post_text_is_enforced_by_adapter_error(self) -> None:
        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload({})

        self.assertEqual(error.exception.code, ERROR_MISSING_REQUIRED_FIELD)
        self.assertEqual(error.exception.diagnostics.missing_required_fields, ("post_text",))

    def test_packaging_fields_are_rejected_as_unexpected_fields(self) -> None:
        parsed = _parsed_candidate(hook_variants=["not writer owned"])

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(parsed)

        self.assertEqual(error.exception.code, ERROR_INVALID_CANDIDATE_POST)
        self.assertEqual(
            error.exception.diagnostics.adapter_error_code,
            ADAPTER_ERROR_PAYLOAD_CONTRACT_VIOLATION,
        )
        self.assertEqual(error.exception.diagnostics.unexpected_fields, ("hook_variants",))
        self.assertIn("hook_variants", error.exception.safe_details["unexpected_fields"])

    def test_non_dict_parsed_candidate_fails(self) -> None:
        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(["not", "a", "dict"])

        self.assertEqual(error.exception.code, ERROR_INVALID_PARSED_CANDIDATE)
        self.assertEqual(
            error.exception.diagnostics.adapter_error_code,
            ADAPTER_ERROR_TOP_LEVEL_NOT_OBJECT,
        )
        self.assertEqual(error.exception.diagnostics.top_level_json_type, "list")

    def test_post_text_wrong_type_has_bounded_field_violation(self) -> None:
        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(_parsed_candidate(post_text=["not text"]))

        self.assertEqual(error.exception.code, ERROR_INVALID_CANDIDATE_POST)
        self.assertEqual(
            error.exception.diagnostics.adapter_error_code,
            ADAPTER_ERROR_INVALID_FIELD_TYPES,
        )
        violation = error.exception.diagnostics.field_violations[0]
        self.assertEqual(violation.field_name, "post_text")
        self.assertEqual(violation.reason_code, FIELD_VIOLATION_WRONG_TYPE)
        self.assertEqual(violation.actual_type, "list")

    def test_empty_and_whitespace_post_text_share_empty_string_violation(self) -> None:
        for value in ("", "   "):
            with self.subTest(value=repr(value)):
                with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
                    adapt_candidate_writer_payload(_parsed_candidate(post_text=value))

                self.assertEqual(
                    error.exception.diagnostics.adapter_error_code,
                    ADAPTER_ERROR_INVALID_FIELD_VALUES,
                )
                violation = error.exception.diagnostics.field_violations[0]
                self.assertEqual(violation.reason_code, FIELD_VIOLATION_EMPTY_STRING)
                self.assertEqual(violation.actual_length, len(value))

    def test_post_text_over_hard_max_fails_with_safe_details(self) -> None:
        rejected_content = "x" * (FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1)

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(_parsed_candidate(post_text=rejected_content))

        self.assertEqual(error.exception.code, ERROR_INVALID_CANDIDATE_POST)
        self.assertEqual(
            error.exception.safe_details["post_text"]["input_length"],
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1,
        )
        violation = error.exception.diagnostics.field_violations[0]
        self.assertEqual(violation.reason_code, FIELD_VIOLATION_ABOVE_MAX_LENGTH)
        self.assertEqual(violation.maximum_allowed, FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS)
        self.assertNotIn(rejected_content, json.dumps(error.exception.safe_details))

    def test_post_text_at_hard_max_succeeds(self) -> None:
        payload = adapt_candidate_writer_payload(
            _parsed_candidate(post_text="x" * FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS)
        )

        self.assertEqual(len(payload["post_text"]), FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS)

    def test_build_candidate_writer_output_maps_core_payload_and_metadata(self) -> None:
        raw_response = _raw_response()

        output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=_parsed_candidate(),
            raw_response=raw_response,
        )

        self.assertIsInstance(output, CandidateWriterOutput)
        self.assertEqual(output.payload, {"post_text": "Clear final post text."})
        self.assertEqual(output.raw_output, raw_response.raw_text)
        self.assertEqual(output.provider, "openai")
        self.assertEqual(output.model, "gpt-4.1-2025-04-14")
        self.assertEqual(output.prompt_name, "final_post_candidate_from_brief")
        self.assertEqual(output.prompt_version, "1.0")
        self.assertEqual(output.token_usage, {"input_tokens": 10})

    def test_raw_response_execution_error_fails_before_payload_adaptation(self) -> None:
        raw_response = _raw_response(execution_error="provider failed")
        with patch.object(
            linkedin_post_candidate_writer_output_adapter,
            "adapt_candidate_writer_payload",
        ) as adapt_payload:
            with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
                build_candidate_writer_output_from_parsed_response(
                    parsed_candidate=_parsed_candidate(),
                    raw_response=raw_response,
                )

        adapt_payload.assert_not_called()
        self.assertEqual(error.exception.code, ERROR_RAW_RESPONSE_EXECUTION_ERROR)

    def test_output_payload_is_independent_from_parsed_source(self) -> None:
        parsed = _parsed_candidate()
        payload = adapt_candidate_writer_payload(parsed)
        parsed["post_text"] = "mutated"
        payload["post_text"] = "also mutated"

        self.assertEqual(
            adapt_candidate_writer_payload(_parsed_candidate())["post_text"],
            "Clear final post text.",
        )

    def test_candidate_writer_output_to_dict_is_json_serializable(self) -> None:
        output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=_parsed_candidate(),
            raw_response=_raw_response(),
        )

        serialized = json.dumps(output.to_dict(), allow_nan=False, sort_keys=True)

        self.assertIn("Clear final post text", serialized)
        self.assertNotIn("hook_variants", serialized)
        self.assertNotIn("quality_checks", serialized)

    def test_error_message_does_not_include_raw_candidate_content(self) -> None:
        secret = "secret candidate sentence"
        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(_parsed_candidate(post_text=secret, hashtags=[]))

        self.assertNotIn(secret, str(error.exception))
        self.assertNotIn(secret, json.dumps(error.exception.safe_details))

    def test_adapter_module_has_no_gate_quality_decision_repair_runtime_or_provider_imports(self) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_candidate_writer_output_adapter))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )

        forbidden_fragments = (
            "openai",
            "gemini",
            "anthropic",
            "deterministic_gate",
            "quality_evaluator",
            "flow_decision",
            "controlled_repair",
            "generator",
        )
        for module_name in imported_modules:
            self.assertFalse(
                any(fragment in module_name for fragment in forbidden_fragments),
                module_name,
            )

    def test_adapter_module_does_not_reference_raw_articles_or_final_payload_validation(self) -> None:
        source = inspect.getsource(linkedin_post_candidate_writer_output_adapter)

        self.assertNotIn("raw_articles", source)
        self.assertNotIn("validate_final_post_payload", source)
        self.assertNotIn("FinalPostPayload", source)


def _parsed_candidate(**overrides) -> dict:
    parsed = {"post_text": "Clear final post text."}
    parsed.update(overrides)
    return parsed


def _raw_response(*, execution_error: str = "") -> CandidateWriterRawResponse:
    return CandidateWriterRawResponse(
        raw_text=json.dumps(_parsed_candidate()),
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_metadata=PromptMetadata(
            prompt_name="final_post_candidate_from_brief",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        ),
        usage={"input_tokens": 10},
        execution_error=execution_error,
        provider_response_metadata=None,
        empty_text_classification=None,
    )
