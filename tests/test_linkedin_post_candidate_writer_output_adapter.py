from __future__ import annotations

import ast
import copy
import inspect
import json
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging import linkedin_post_candidate_writer_output_adapter
from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterRawResponse,
)
from services.packaging.linkedin_post_candidate_writer_output_adapter import (
    CANONICAL_FINAL_POST_PAYLOAD_FIELDS,
    ERROR_INVALID_FINAL_POST_PAYLOAD,
    ERROR_INVALID_PARSED_CANDIDATE,
    ERROR_MISSING_REQUIRED_FIELD,
    ERROR_RAW_RESPONSE_EXECUTION_ERROR,
    CandidateWriterOutputAdaptationError,
    adapt_candidate_writer_payload,
    build_candidate_writer_output_from_parsed_response,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_CTA_VARIANTS_MIN_COUNT,
    FINAL_POST_PAYLOAD_HASHTAGS_MIN_COUNT,
    FINAL_POST_PAYLOAD_HOOK_VARIANTS_MIN_COUNT,
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput


class CandidateWriterOutputAdapterTests(SimpleTestCase):
    def test_adapt_candidate_writer_payload_returns_canonical_payload_dict(self) -> None:
        payload = adapt_candidate_writer_payload(_parsed_candidate())

        self.assertEqual(tuple(payload), CANONICAL_FINAL_POST_PAYLOAD_FIELDS)
        self.assertEqual(payload["post_text"], "Clear final post text.")
        self.assertEqual(payload["hook_variants"], ["Hook one", "Hook two", "Hook three"])
        self.assertEqual(payload["carousel_outline"], [])

    def test_required_fields_are_enforced_by_adapter_error(self) -> None:
        parsed = _parsed_candidate()
        parsed.pop("post_text")

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(parsed)

        self.assertEqual(error.exception.code, ERROR_MISSING_REQUIRED_FIELD)
        self.assertIn("post_text", str(error.exception))

    def test_missing_carousel_outline_gets_adapter_owned_empty_list_default(self) -> None:
        parsed = _parsed_candidate()
        parsed.pop("carousel_outline")

        payload = adapt_candidate_writer_payload(parsed)

        self.assertEqual(payload["carousel_outline"], [])

    def test_invalid_optional_carousel_outline_fails_structure_validation(self) -> None:
        parsed = _parsed_candidate(carousel_outline=None)

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(parsed)

        self.assertEqual(error.exception.code, ERROR_INVALID_FINAL_POST_PAYLOAD)

    def test_non_dict_parsed_candidate_fails(self) -> None:
        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(["not", "a", "dict"])

        self.assertEqual(error.exception.code, ERROR_INVALID_PARSED_CANDIDATE)

    def test_invalid_basic_types_fail_through_final_post_payload_validation(self) -> None:
        invalid_cases = (
            ("post_text", ""),
            ("hook_variants", ["Only one hook"]),
            ("cta_variants", "not a list"),
            ("hashtags", []),
            ("quality_checks", {"linkedin_ready": True}),
        )

        for field_name, invalid_value in invalid_cases:
            with self.subTest(field_name=field_name):
                with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
                    adapt_candidate_writer_payload(
                        _parsed_candidate(**{field_name: invalid_value})
                    )

                self.assertEqual(
                    error.exception.code,
                    ERROR_INVALID_FINAL_POST_PAYLOAD,
                )

    def test_non_boolean_quality_check_values_fail(self) -> None:
        parsed = _parsed_candidate(
            quality_checks={
                "linkedin_ready": "yes",
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
            }
        )

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(parsed)

        self.assertEqual(error.exception.code, ERROR_INVALID_FINAL_POST_PAYLOAD)
        self.assertEqual(
            error.exception.safe_details["quality_checks"]["provided_keys"],
            [
                "has_clear_point_of_view",
                "linkedin_ready",
                "uses_only_provided_facts",
            ],
        )

    def test_missing_quality_check_key_fails_with_required_key_details(self) -> None:
        parsed = _parsed_candidate(
            quality_checks={
                "linkedin_ready": True,
                "has_clear_point_of_view": True,
            }
        )

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(parsed)

        self.assertEqual(error.exception.code, ERROR_INVALID_FINAL_POST_PAYLOAD)
        self.assertIn(
            "uses_only_provided_facts",
            error.exception.safe_details["quality_checks"]["required_boolean_keys"],
        )

    def test_post_text_over_hard_max_fails_with_safe_details(self) -> None:
        rejected_content = "x" * (FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1)

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(_parsed_candidate(post_text=rejected_content))

        self.assertEqual(error.exception.code, ERROR_INVALID_FINAL_POST_PAYLOAD)
        self.assertEqual(
            error.exception.safe_details["post_text"]["input_length"],
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1,
        )
        self.assertEqual(
            error.exception.safe_details["post_text"]["max_chars"],
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
        )
        self.assertNotIn(rejected_content, json.dumps(error.exception.safe_details))

    def test_post_text_at_hard_max_succeeds(self) -> None:
        payload = adapt_candidate_writer_payload(
            _parsed_candidate(post_text="x" * FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS)
        )

        self.assertEqual(
            len(payload["post_text"]),
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
        )

    def test_too_few_hook_variants_fail_with_safe_count(self) -> None:
        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(_parsed_candidate(hook_variants=["Only one"]))

        self.assertEqual(error.exception.safe_details["hook_variants"]["input_count"], 1)
        self.assertEqual(
            error.exception.safe_details["hook_variants"]["min_count"],
            FINAL_POST_PAYLOAD_HOOK_VARIANTS_MIN_COUNT,
        )

    def test_too_few_cta_variants_fail_with_safe_count(self) -> None:
        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(_parsed_candidate(cta_variants=["Only one"]))

        self.assertEqual(error.exception.safe_details["cta_variants"]["input_count"], 1)
        self.assertEqual(
            error.exception.safe_details["cta_variants"]["min_count"],
            FINAL_POST_PAYLOAD_CTA_VARIANTS_MIN_COUNT,
        )

    def test_zero_hashtags_fail_with_safe_count(self) -> None:
        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(_parsed_candidate(hashtags=[]))

        self.assertEqual(error.exception.safe_details["hashtags"]["input_count"], 0)
        self.assertEqual(
            error.exception.safe_details["hashtags"]["min_count"],
            FINAL_POST_PAYLOAD_HASHTAGS_MIN_COUNT,
        )

    def test_adaptation_safe_details_do_not_include_rejected_content(self) -> None:
        parsed = _parsed_candidate(
            post_text="secret rejected post text",
            hook_variants=[],
        )

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(parsed)

        serialized = json.dumps(error.exception.safe_details, sort_keys=True)
        self.assertIn("hook_variants", serialized)
        self.assertNotIn("secret rejected post text", serialized)

    def test_adapter_reuses_final_post_payload_validation(self) -> None:
        with patch(
            "services.packaging.linkedin_post_candidate_writer_output_adapter."
            "validate_final_post_payload"
        ) as validate:
            adapt_candidate_writer_payload(_parsed_candidate())

        validate.assert_called_once()

    def test_extra_top_level_fields_are_dropped_through_positive_allowlist(self) -> None:
        parsed = _parsed_candidate(
            provider="provider-sentinel",
            model="model-sentinel",
            usage={"total_tokens": 10},
            raw_response={"id": "raw-sentinel"},
            debug="debug-sentinel",
            runtime="runtime-sentinel",
            audit="audit-sentinel",
            source_document="source-document-sentinel",
            internal_notes="internal-notes-sentinel",
            package={"id": "package-sentinel"},
            persistence={"id": "persistence-sentinel"},
        )

        payload = adapt_candidate_writer_payload(parsed)
        serialized = json.dumps(payload, sort_keys=True)

        self.assertEqual(set(payload), set(CANONICAL_FINAL_POST_PAYLOAD_FIELDS))
        for sentinel in (
            "provider-sentinel",
            "model-sentinel",
            "raw-sentinel",
            "debug-sentinel",
            "runtime-sentinel",
            "audit-sentinel",
            "source-document-sentinel",
            "internal-notes-sentinel",
            "package-sentinel",
            "persistence-sentinel",
        ):
            self.assertNotIn(sentinel, serialized)

    def test_quality_checks_are_allowlisted_to_required_canonical_keys(self) -> None:
        payload = adapt_candidate_writer_payload(
            _parsed_candidate(
                quality_checks={
                    "linkedin_ready": True,
                    "uses_only_provided_facts": True,
                    "has_clear_point_of_view": True,
                    "debug": "debug-sentinel",
                    "provider": "provider-sentinel",
                }
            )
        )

        self.assertEqual(
            set(payload["quality_checks"]),
            {
                "linkedin_ready",
                "uses_only_provided_facts",
                "has_clear_point_of_view",
            },
        )
        self.assertNotIn("debug-sentinel", json.dumps(payload, sort_keys=True))

    def test_source_parsed_dictionary_is_not_mutated(self) -> None:
        parsed = _parsed_candidate(carousel_outline=[{"title": "Slide one"}])
        before = copy.deepcopy(parsed)

        adapt_candidate_writer_payload(parsed)

        self.assertEqual(parsed, before)

    def test_nested_canonical_values_are_defensively_copied(self) -> None:
        parsed = _parsed_candidate(carousel_outline=[{"title": "Slide one"}])

        payload = adapt_candidate_writer_payload(parsed)
        parsed["hook_variants"].append("Mutated hook")
        parsed["quality_checks"]["linkedin_ready"] = False
        parsed["carousel_outline"][0]["title"] = "Mutated slide"

        self.assertEqual(payload["hook_variants"], ["Hook one", "Hook two", "Hook three"])
        self.assertTrue(payload["quality_checks"]["linkedin_ready"])
        self.assertEqual(payload["carousel_outline"], [{"title": "Slide one"}])

    def test_mutating_adapted_payload_does_not_mutate_repeated_outputs(self) -> None:
        parsed = _parsed_candidate(carousel_outline=[{"title": "Slide one"}])

        first = adapt_candidate_writer_payload(parsed)
        second = adapt_candidate_writer_payload(parsed)
        first["hook_variants"].append("Changed")
        first["quality_checks"]["linkedin_ready"] = False
        first["carousel_outline"][0]["title"] = "Changed"

        self.assertEqual(second["hook_variants"], ["Hook one", "Hook two", "Hook three"])
        self.assertTrue(second["quality_checks"]["linkedin_ready"])
        self.assertEqual(second["carousel_outline"], [{"title": "Slide one"}])

    def test_error_message_does_not_include_raw_candidate_content(self) -> None:
        parsed = _parsed_candidate(post_text="secret raw candidate content")
        parsed["hook_variants"] = []

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            adapt_candidate_writer_payload(parsed)

        self.assertNotIn("secret raw candidate content", str(error.exception))

    def test_build_candidate_writer_output_maps_payload_and_raw_response_metadata(
        self,
    ) -> None:
        raw_response = _raw_response()

        output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=_parsed_candidate(),
            raw_response=raw_response,
        )

        self.assertIsInstance(output, CandidateWriterOutput)
        self.assertEqual(output.payload, adapt_candidate_writer_payload(_parsed_candidate()))
        self.assertEqual(output.raw_output, raw_response.raw_text)
        self.assertEqual(output.provider, "openai")
        self.assertEqual(output.model, "gpt-4.1-2025-04-14")
        self.assertEqual(output.prompt_name, "final_post_candidate_from_brief")
        self.assertEqual(output.prompt_version, "1.0")
        self.assertEqual(output.token_usage, {"total_tokens": 42})
        self.assertIsNone(output.cost_metadata)

    def test_raw_response_execution_error_fails_before_payload_adaptation(self) -> None:
        raw_response = _raw_response(execution_error="provider failed")

        with patch(
            "services.packaging.linkedin_post_candidate_writer_output_adapter."
            "adapt_candidate_writer_payload"
        ) as adapt:
            with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
                build_candidate_writer_output_from_parsed_response(
                    parsed_candidate=_parsed_candidate(),
                    raw_response=raw_response,
                )

        self.assertEqual(error.exception.code, ERROR_RAW_RESPONSE_EXECUTION_ERROR)
        adapt.assert_not_called()

    def test_output_payload_excludes_raw_response_audit_metadata(self) -> None:
        output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=_parsed_candidate(),
            raw_response=_raw_response(
                usage={"usage_sentinel": 1},
                raw_provider_response={"raw_provider_sentinel": True},
                execution_metadata={"execution_sentinel": True},
            ),
        )
        serialized_payload = json.dumps(output.payload, sort_keys=True)

        self.assertNotIn("usage_sentinel", serialized_payload)
        self.assertNotIn("raw_provider_sentinel", serialized_payload)
        self.assertNotIn("execution_sentinel", serialized_payload)

    def test_candidate_writer_output_to_dict_is_json_serializable(self) -> None:
        output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=_parsed_candidate(),
            raw_response=_raw_response(),
        )

        serialized = json.dumps(output.to_dict(), sort_keys=True)

        self.assertIn("Clear final post text.", serialized)
        self.assertIn("gpt-4.1-2025-04-14", serialized)

    def test_output_metadata_is_defensively_copied_from_raw_response(self) -> None:
        raw_response = _raw_response(usage={"total_tokens": {"count": 42}})

        output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=_parsed_candidate(),
            raw_response=raw_response,
        )
        output.token_usage["total_tokens"]["count"] = 99

        self.assertEqual(raw_response.usage, {"total_tokens": {"count": 42}})

    def test_missing_raw_response_attribute_fails_contract_shape_check(self) -> None:
        class DriftedRawResponse:
            execution_error = None
            prompt_metadata = None
            raw_text = '{"post_text": "Clear final post text."}'
            provider = "openai"
            model = "gpt-4.1-2025-04-14"

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            build_candidate_writer_output_from_parsed_response(
                parsed_candidate=_parsed_candidate(),
                raw_response=DriftedRawResponse(),
            )

        self.assertEqual(error.exception.code, "invalid_raw_response")
        self.assertIn("usage", str(error.exception))

    def test_missing_prompt_metadata_attribute_fails_contract_shape_check(self) -> None:
        class DriftedPromptMetadata:
            prompt_name = "final_post_candidate_from_brief"

        raw_response = _raw_response()
        drifted_response = CandidateWriterRawResponse(
            raw_text=raw_response.raw_text,
            provider=raw_response.provider,
            model=raw_response.model,
            prompt_metadata=DriftedPromptMetadata(),
            usage=raw_response.usage,
        )

        with self.assertRaises(CandidateWriterOutputAdaptationError) as error:
            build_candidate_writer_output_from_parsed_response(
                parsed_candidate=_parsed_candidate(),
                raw_response=drifted_response,
            )

        self.assertEqual(error.exception.code, "invalid_raw_response")
        self.assertIn("prompt_version", str(error.exception))

    def test_output_payload_is_independent_from_parsed_source(self) -> None:
        parsed = _parsed_candidate()

        output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=parsed,
            raw_response=_raw_response(),
        )
        parsed["post_text"] = "Changed source text"
        parsed["quality_checks"]["linkedin_ready"] = False

        self.assertEqual(output.payload["post_text"], "Clear final post text.")
        self.assertTrue(output.payload["quality_checks"]["linkedin_ready"])

    def test_adapter_module_has_no_gate_quality_decision_repair_runtime_or_provider_imports(
        self,
    ) -> None:
        source = inspect.getsource(linkedin_post_candidate_writer_output_adapter)
        imports = _imported_names_and_modules(source)

        forbidden_imports = {
            "OpenAIClient",
            "django.conf",
            "django.db",
            "apps.packaging.models",
            "ContentPackage",
            "services.packaging.generator",
            "services.packaging.linkedin_post_deterministic_gate",
            "run_final_post_deterministic_gate",
            "services.packaging.linkedin_post_deterministic_orchestration",
            "services.packaging.linkedin_post_quality_evaluator_parser",
            "services.packaging.linkedin_post_quality_review_contract",
            "services.packaging.linkedin_post_flow_decision",
            "FinalPostDecisionController",
            "TargetedRepairPlan",
            "RepairAgentInput",
            "execute_candidate_writer_prompt",
        }

        self.assertTrue(forbidden_imports.isdisjoint(imports))

    def test_adapter_module_does_not_reference_raw_articles(self) -> None:
        source = inspect.getsource(linkedin_post_candidate_writer_output_adapter).lower()

        self.assertNotIn("raw_article", source)
        self.assertNotIn("articles", source)

    def test_adapter_module_declares_approved_public_api(self) -> None:
        self.assertEqual(
            linkedin_post_candidate_writer_output_adapter.__all__,
            (
                "CandidateWriterOutputAdaptationError",
                "adapt_candidate_writer_payload",
                "build_candidate_writer_output_from_parsed_response",
            ),
        )


def _parsed_candidate(**overrides) -> dict:
    payload = {
        "post_text": "Clear final post text.",
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
    payload.update(overrides)
    return payload


def _raw_response(
    *,
    usage: dict | None = None,
    raw_provider_response: dict | None = None,
    execution_metadata: dict | None = None,
    execution_error: str | None = None,
) -> CandidateWriterRawResponse:
    return CandidateWriterRawResponse(
        raw_text='{"post_text": "Clear final post text."}',
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_metadata=PromptMetadata(
            prompt_name="final_post_candidate_from_brief",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        ),
        usage=usage if usage is not None else {"total_tokens": 42},
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
