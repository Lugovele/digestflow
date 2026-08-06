from __future__ import annotations

import ast
import copy
import inspect
import json
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging import linkedin_post_candidate_writer_parser
from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterRawResponse,
)
from services.packaging.linkedin_post_candidate_writer_parser import (
    ERROR_EMPTY_RAW_RESPONSE,
    ERROR_EXECUTION_FAILED,
    ERROR_MALFORMED_FENCE,
    ERROR_MALFORMED_JSON,
    ERROR_NON_OBJECT_JSON,
    CandidateWriterResponseParseError,
    parse_candidate_writer_raw_response,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata


class LinkedInCandidateWriterParserTests(SimpleTestCase):
    def test_plain_json_object_parses(self) -> None:
        parsed = parse_candidate_writer_raw_response(
            _raw_response('{"post_text": "Post", "hashtags": ["#AI"]}')
        )

        self.assertEqual(parsed["post_text"], "Post")
        self.assertEqual(parsed["hashtags"], ["#AI"])

    def test_json_labelled_fence_parses(self) -> None:
        parsed = parse_candidate_writer_raw_response(
            _raw_response('```json\n{"post_text": "Post"}\n```')
        )

        self.assertEqual(parsed, {"post_text": "Post"})

    def test_generic_fence_parses(self) -> None:
        parsed = parse_candidate_writer_raw_response(
            _raw_response('```\n{"post_text": "Post"}\n```')
        )

        self.assertEqual(parsed, {"post_text": "Post"})

    def test_surrounding_response_whitespace_parses(self) -> None:
        parsed = parse_candidate_writer_raw_response(
            _raw_response('\n\t  {"post_text": "Post"}  \r\n')
        )

        self.assertEqual(parsed["post_text"], "Post")

    def test_whitespace_inside_fence_parses(self) -> None:
        parsed = parse_candidate_writer_raw_response(
            _raw_response('```json\n\n  {"post_text": "Post"}  \n\n```')
        )

        self.assertEqual(parsed["post_text"], "Post")

    def test_unicode_strings_are_preserved(self) -> None:
        parsed = parse_candidate_writer_raw_response(
            _raw_response(
                '{"post_text": "\\u0447\\u0435\\u043b\\u043e\\u0432\\u0435\\u0447\\u043d\\u043e"}'
            )
        )

        self.assertEqual(parsed["post_text"], "человечно")

    def test_nested_canonical_values_parse(self) -> None:
        raw = _raw_response(
            json.dumps(
                {
                    "post_text": "Post",
                    "hook_variants": ["Hook 1", "Hook 2", "Hook 3"],
                    "cta_variants": ["CTA 1", "CTA 2", "CTA 3"],
                    "hashtags": ["#AI"],
                    "quality_checks": {
                        "linkedin_ready": True,
                        "uses_only_provided_facts": True,
                        "has_clear_point_of_view": True,
                    },
                    "carousel_outline": [{"slide": 1, "title": "Point"}],
                }
            )
        )

        parsed = parse_candidate_writer_raw_response(raw)

        self.assertEqual(parsed["quality_checks"]["linkedin_ready"], True)
        self.assertEqual(parsed["carousel_outline"][0]["title"], "Point")

    def test_valid_multiline_post_text_with_escaped_paragraph_breaks_parses(self) -> None:
        raw = _raw_response(
            json.dumps(
                {
                    "post_text": "First paragraph.\n\nSecond paragraph.",
                    "hook_variants": ["Hook 1", "Hook 2", "Hook 3"],
                    "cta_variants": ["CTA 1", "CTA 2", "CTA 3"],
                    "hashtags": ["#AI"],
                    "quality_checks": {
                        "linkedin_ready": True,
                        "uses_only_provided_facts": True,
                        "has_clear_point_of_view": True,
                    },
                    "carousel_outline": [],
                }
            )
        )

        parsed = parse_candidate_writer_raw_response(raw)

        self.assertEqual(
            parsed["post_text"],
            "First paragraph.\n\nSecond paragraph.",
        )

    def test_raw_response_remains_unchanged(self) -> None:
        raw = _raw_response(json.dumps({"post_text": "Post"}))
        before = copy.deepcopy(raw)

        parse_candidate_writer_raw_response(raw)

        self.assertEqual(raw, before)

    def test_parsed_result_is_independent_between_calls(self) -> None:
        raw = _raw_response(json.dumps({"nested": {"items": ["a"]}}))

        first = parse_candidate_writer_raw_response(raw)
        second = parse_candidate_writer_raw_response(raw)
        first["nested"]["items"].append("changed")

        self.assertEqual(second["nested"]["items"], ["a"])

    def test_execution_failure_wins_and_skips_json_decode(self) -> None:
        raw = _raw_response('{"post_text": "Post"}', execution_error="provider failed")

        with patch(
            "services.packaging.linkedin_post_candidate_writer_parser.json.loads"
        ) as mock_loads:
            with self.assertRaisesRegex(
                CandidateWriterResponseParseError,
                "execution failed",
            ) as error:
                parse_candidate_writer_raw_response(raw)

        self.assertEqual(error.exception.code, ERROR_EXECUTION_FAILED)
        self.assertEqual(error.exception.diagnostics.parser_error_code, ERROR_EXECUTION_FAILED)
        mock_loads.assert_not_called()

    def test_execution_failure_message_does_not_include_raw_text_or_metadata(
        self,
    ) -> None:
        raw = _raw_response(
            '{"secret": "candidate text"}',
            execution_error="provider failed",
            provider="openai-sentinel",
            model="model-sentinel",
            usage={"request_id": "usage-sentinel"},
            raw_provider_response={"raw_sentinel": True},
        )

        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(raw)

        message = str(error.exception)
        self.assertNotIn("candidate text", message)
        self.assertNotIn("openai-sentinel", message)
        self.assertNotIn("model-sentinel", message)
        self.assertNotIn("usage-sentinel", message)
        self.assertNotIn("raw_sentinel", message)

    def test_empty_raw_response_fails(self) -> None:
        for raw_text in (None, "", " ", "\t", "\r\n"):
            with self.subTest(raw_text=raw_text):
                with self.assertRaises(CandidateWriterResponseParseError) as error:
                    parse_candidate_writer_raw_response(_raw_response(raw_text))

                self.assertEqual(error.exception.code, ERROR_EMPTY_RAW_RESPONSE)
                self.assertEqual(
                    error.exception.diagnostics.parser_error_code,
                    ERROR_EMPTY_RAW_RESPONSE,
                )

    def test_unsupported_fence_label_fails(self) -> None:
        for label in ("python", "javascript", "text", "yaml", "markdown"):
            with self.subTest(label=label):
                with self.assertRaises(CandidateWriterResponseParseError) as error:
                    parse_candidate_writer_raw_response(
                        _raw_response(f"```{label}\n{{}}\n```")
                    )

                self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_prose_before_fence_fails(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response('Intro\n```json\n{}\n```'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)
        self.assertEqual(error.exception.diagnostics.parser_error_code, ERROR_MALFORMED_FENCE)

    def test_prose_after_fence_fails(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(
                _raw_response('```json\n{}\n```\nTrailing prose')
            )

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_multiple_fences_fail(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(
                _raw_response('```json\n{}\n```\n```json\n{}\n```')
            )

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_unclosed_fence_fails(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response('```json\n{}'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_closing_fence_without_opening_fails(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response('{}\n```'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_nested_fence_fails(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(
                _raw_response('```json\n{"text": "ok"}\n```\n```')
            )

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_four_backtick_fence_fails(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response('````json\n{}\n````'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_content_after_closing_fence_fails(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response('```json\n{}\n``` {}'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_malformed_opening_fence_label_fails(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response('``` json\n{}\n```'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_backticks_inside_plain_json_string_do_not_trigger_fence_parsing(
        self,
    ) -> None:
        parsed = parse_candidate_writer_raw_response(
            _raw_response('{"post_text": "contains ``` inside string"}')
        )

        self.assertEqual(parsed["post_text"], "contains ``` inside string")

    def test_malformed_json_fails(self) -> None:
        cases = (
            '{"post_text": ',
            '{"post_text": "Post",}',
            '{"a": 1} {"b": 2}',
            '{"a": 1} trailing',
            'leading {"a": 1}',
        )
        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                with self.assertRaises(CandidateWriterResponseParseError) as error:
                    parse_candidate_writer_raw_response(_raw_response(raw_text))

                self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)
                self.assertEqual(
                    error.exception.diagnostics.parser_error_code,
                    ERROR_MALFORMED_JSON,
                )

    def test_literal_physical_newline_inside_quoted_json_string_fails(self) -> None:
        malformed = '{"post_text": "First paragraph.\nSecond paragraph."}'

        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response(malformed))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)
        self.assertEqual(error.exception.diagnostics.candidate_text_length, len(malformed))

    def test_trailing_comma_fails_without_repair(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response('{"post_text": "Post",}'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_commented_json_fails_without_repair(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(
                _raw_response('{"post_text": "Post" // comment\n}')
            )

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_single_quoted_pseudo_json_fails_without_repair(self) -> None:
        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response("{'post_text': 'Post'}"))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_non_standard_json_numbers_fail(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with self.assertRaises(CandidateWriterResponseParseError) as error:
                    parse_candidate_writer_raw_response(
                        _raw_response(f'{{"score": {constant}}}')
                    )

                self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_non_object_json_fails(self) -> None:
        cases = ("[]", '"post"', "42", "4.2", "true", "null")
        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                with self.assertRaises(CandidateWriterResponseParseError) as error:
                    parse_candidate_writer_raw_response(_raw_response(raw_text))

                self.assertEqual(error.exception.code, ERROR_NON_OBJECT_JSON)
                self.assertEqual(
                    error.exception.diagnostics.parser_error_code,
                    ERROR_NON_OBJECT_JSON,
                )
                self.assertIsNotNone(error.exception.diagnostics.top_level_json_type)

    def test_parse_diagnostics_record_length_without_response_content(self) -> None:
        raw_text = '{"post_text": "secret post value", "api_key": "secret"} trailing'

        with self.assertRaises(CandidateWriterResponseParseError) as error:
            parse_candidate_writer_raw_response(_raw_response(raw_text))

        diagnostics = error.exception.diagnostics.to_dict()
        serialized = json.dumps(diagnostics, sort_keys=True)
        self.assertEqual(diagnostics["candidate_text_length"], len(raw_text))
        self.assertNotIn("secret post value", serialized)
        self.assertNotIn("api_key", serialized)

    def test_unknown_top_level_fields_are_preserved_for_later_adapter_policy(
        self,
    ) -> None:
        parsed = parse_candidate_writer_raw_response(
            _raw_response('{"post_text": "Post", "audit_sentinel": "keep-for-adapter"}')
        )

        self.assertEqual(parsed["audit_sentinel"], "keep-for-adapter")

    def test_missing_canonical_fields_are_left_to_later_adapter_policy(self) -> None:
        parsed = parse_candidate_writer_raw_response(_raw_response('{"post_text": "Post"}'))

        self.assertEqual(parsed, {"post_text": "Post"})
        self.assertNotIn("carousel_outline", parsed)

    def test_audit_metadata_is_not_added_to_parsed_result(self) -> None:
        raw = _raw_response(
            '{"post_text": "Post"}',
            provider="provider-sentinel",
            model="model-sentinel",
            usage={"usage_sentinel": 1},
            raw_provider_response={"raw_sentinel": True},
            execution_metadata={"execution_sentinel": True},
            prompt_metadata=PromptMetadata(
                prompt_name="prompt-sentinel",
                prompt_version="version-sentinel",
                prompt_path="path-sentinel",
            ),
        )

        parsed = parse_candidate_writer_raw_response(raw)
        serialized = json.dumps(parsed, sort_keys=True)

        self.assertNotIn("provider-sentinel", serialized)
        self.assertNotIn("model-sentinel", serialized)
        self.assertNotIn("usage_sentinel", serialized)
        self.assertNotIn("raw_sentinel", serialized)
        self.assertNotIn("execution_sentinel", serialized)
        self.assertNotIn("prompt-sentinel", serialized)

    def test_mutating_parsed_nested_values_does_not_mutate_raw_response(self) -> None:
        raw = _raw_response(json.dumps({"nested": {"items": ["original"]}}))
        before = copy.deepcopy(raw)

        parsed = parse_candidate_writer_raw_response(raw)
        parsed["nested"]["items"].append("changed")

        self.assertEqual(raw, before)

    def test_parser_module_has_no_adapter_gate_decision_repair_or_runtime_dependencies(
        self,
    ) -> None:
        source = inspect.getsource(linkedin_post_candidate_writer_parser)
        imports = _imported_names_and_modules(source)

        self.assertNotIn("OpenAIClient", imports)
        self.assertNotIn("django.conf", imports)
        self.assertNotIn(
            "services.packaging.linkedin_post_candidate_writer_execution",
            imports,
        )
        self.assertNotIn("services.packaging.generator", imports)
        self.assertNotIn("ContentPackage", imports)
        self.assertNotIn("apps.packaging.models", imports)
        self.assertNotIn("CandidateWriterOutput", imports)
        self.assertNotIn("FinalPostPayload", imports)
        self.assertNotIn("final_post_payload_to_dict", imports)
        self.assertNotIn("validate_final_post_payload", imports)
        self.assertNotIn("FinalPostDecisionController", imports)
        self.assertNotIn("TargetedRepairPlan", imports)
        self.assertNotIn("run_final_post_deterministic_gate", imports)
        self.assertNotIn("normalize_quality_review_result", imports)
        self.assertNotIn("linkedin_post_prompt_registry", imports)
        self.assertNotIn("render_candidate_writer_prompt_input", imports)

    def test_parser_does_not_duplicate_final_post_payload_or_gate_rules(self) -> None:
        source = inspect.getsource(linkedin_post_candidate_writer_parser)

        self.assertNotIn("REQUIRED_QUALITY_CHECKS", source)
        self.assertNotIn("post_text must not exceed", source)
        self.assertNotIn("linkedin_ready", source)
        self.assertNotIn("selected evidence", source)
        self.assertNotIn("scaffold", source)


def _raw_response(
    raw_text,
    *,
    execution_error: str | None = None,
    provider: str = "openai",
    model: str = "candidate-model",
    prompt_metadata: PromptMetadata | None = None,
    usage: dict | None = None,
    raw_provider_response: dict | None = None,
    execution_metadata: dict | None = None,
) -> CandidateWriterRawResponse:
    return CandidateWriterRawResponse(
        raw_text=raw_text,
        provider=provider,
        model=model,
        prompt_metadata=prompt_metadata,
        usage=usage,
        raw_provider_response=raw_provider_response,
        execution_error=execution_error,
        execution_metadata=execution_metadata,
    )


def _imported_names_and_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    imported: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "TYPE_CHECKING":
                return
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
                imported.add(alias.asname or alias.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            for alias in node.names:
                imported.add(alias.name)
                imported.add(alias.asname or alias.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return imported
