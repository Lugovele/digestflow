from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging import linkedin_post_repair_writer_benchmark
from services.packaging.linkedin_post_repair_writer_benchmark import (
    BENCHMARK_STATUS_COMPLETED,
    BENCHMARK_STATUS_DRY_RUN,
    DEFAULT_EXPERIMENT_ID,
    FIXED_QUALITY_EVALUATOR_MODEL,
    FIXED_QUALITY_EVALUATOR_PROVIDER,
    FIXED_SEMANTIC_GROUNDING_MODEL,
    FIXED_SEMANTIC_GROUNDING_PROVIDER,
    PLAN_CLAUDE_REPAIR,
    PLAN_GEMINI_REPAIR,
    PLAN_GPT_REPAIR,
    REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING,
    REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_MINIMAL_REASONING,
    REPAIR_WRITER_EXECUTION_PROFILE_PROVIDER_DEFAULT,
    REPAIR_WRITER_JSON_MODE,
    REPAIR_WRITER_SELECTION_BLOCKER_GENERICIZATION,
    REPAIR_WRITER_MAX_OUTPUT_TOKENS,
    RepairWriterBenchmarkRequest,
    build_repair_writer_benchmark_prompt_render,
    default_repair_writer_benchmark_cases,
    default_repair_writer_benchmark_plans,
    load_repair_writer_benchmark_case,
    run_repair_writer_benchmark,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    QualityEvaluatorRawResponse,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_repair_writer_execution import (
    RepairWriterRawResponse,
)
from services.packaging.linkedin_post_provider_diagnostics import (
    PROVIDER_ERROR_RATE_LIMIT,
)
from services.packaging.linkedin_post_repair_writer_structural_diagnostics import (
    COMPLETE_PLAIN_JSON,
    NON_JSON_RESPONSE,
    POST_TEXT_TOO_LONG,
    UNKNOWN,
)
from services.packaging.linkedin_post_semantic_grounding_execution import (
    SemanticGroundingRawResponse,
)
from services.packaging.linkedin_post_semantic_grounding_structural_diagnostics import (
    TRUNCATED_INSIDE_JSON as SEMANTIC_TRUNCATED_INSIDE_JSON,
)


class RepairWriterBenchmarkTests(SimpleTestCase):
    def test_default_cases_are_the_approved_three_case_set(self) -> None:
        cases = default_repair_writer_benchmark_cases()

        self.assertEqual(
            tuple(case.case_id for case in cases),
            (
                "topic_140_digest_126",
                "topic_214_digest_128__claude_v3",
                "topic_200_digest_134__gpt_v2",
            ),
        )
        self.assertEqual(cases[0].frozen_input_reconstruction, "DIRECT")
        self.assertEqual(cases[1].frozen_input_reconstruction, "DETERMINISTIC_RECONSTRUCTION")
        self.assertEqual(cases[2].frozen_input_reconstruction, "DETERMINISTIC_RECONSTRUCTION")

    def test_default_plans_compare_gpt_claude_and_gemini_repair_writers(self) -> None:
        plans = default_repair_writer_benchmark_plans()

        self.assertEqual(
            tuple(plan.plan_id for plan in plans),
            (PLAN_GPT_REPAIR, PLAN_CLAUDE_REPAIR, PLAN_GEMINI_REPAIR),
        )
        self.assertEqual(tuple(plan.provider for plan in plans), ("openai", "anthropic", "gemini"))
        self.assertEqual(
            tuple(plan.model for plan in plans),
            ("gpt-4.1-2025-04-14", "claude-sonnet-5", "gemini-3.6-flash"),
        )
        self.assertEqual(
            tuple(plan.max_output_tokens for plan in plans),
            (1800, 1800, 1800),
        )
        self.assertEqual(tuple(plan.json_mode for plan in plans), (False, False, False))
        self.assertEqual(
            tuple(plan.execution_profile for plan in plans),
            (
                REPAIR_WRITER_EXECUTION_PROFILE_PROVIDER_DEFAULT,
                REPAIR_WRITER_EXECUTION_PROFILE_PROVIDER_DEFAULT,
                REPAIR_WRITER_EXECUTION_PROFILE_PROVIDER_DEFAULT,
            ),
        )

    def test_plan_to_dict_includes_execution_profile(self) -> None:
        plan = default_repair_writer_benchmark_plans()[2].__class__(
            "gemini_low",
            "gemini",
            "gemini-3.6-flash",
            execution_profile=REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING,
        )

        self.assertEqual(
            plan.to_dict()["execution_profile"],
            REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING,
        )

    def test_default_case_excludes_gpt_v5_diagnostic_case(self) -> None:
        self.assertNotIn(
            "topic_214_digest_128__gpt_v5",
            {case.case_id for case in default_repair_writer_benchmark_cases()},
        )

    def test_loaded_case_uses_source_fixture_candidate_and_repair_metadata(self) -> None:
        case = load_repair_writer_benchmark_case(
            "tests/fixtures/linkedin_post_repair_writer_benchmark/topic_200_digest_134__gpt_v2.json"
        )

        self.assertEqual(case.case_id, "topic_200_digest_134__gpt_v2")
        self.assertEqual(case.repair_instruction["failed_criterion"], "author_point_of_view")
        self.assertEqual(case.known_quality_result["pass"], False)
        self.assertEqual(case.selected_evidence, tuple(case.post_brief["evidence_to_use"]))

    def test_prompt_render_preserves_candidate_context_and_instruction(self) -> None:
        case = default_repair_writer_benchmark_cases()[1]

        render = build_repair_writer_benchmark_prompt_render(case)

        self.assertEqual(render.prompt_name, "fixture_repair_writer_prompt")
        self.assertIn("original_candidate_payload_json", render.variables)
        self.assertIn("repair_instruction_json", render.variables)
        self.assertIn(case.candidate_payload["post_text"][:80], render.input_text)
        self.assertIn(case.repair_instruction["repair_instruction"][:80], render.input_text)

    def test_repair_prompt_output_contract_rejects_fences_and_trailing_prose(self) -> None:
        self.assertIn("Return exactly one plain JSON object", linkedin_post_repair_writer_benchmark.REPAIR_PROMPT_TEXT)
        self.assertIn('{"post_text":"..."}', linkedin_post_repair_writer_benchmark.REPAIR_PROMPT_TEXT)
        self.assertIn("Do not wrap the JSON in markdown or code fences", linkedin_post_repair_writer_benchmark.REPAIR_PROMPT_TEXT)
        self.assertIn("Do not include prose before or after the JSON", linkedin_post_repair_writer_benchmark.REPAIR_PROMPT_TEXT)

    def test_dry_run_writes_artifacts_and_makes_zero_provider_calls(self) -> None:
        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=default_repair_writer_benchmark_cases(),
                    plans=default_repair_writer_benchmark_plans(),
                    allow_api=False,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )

        self.assertEqual(result.status, BENCHMARK_STATUS_DRY_RUN)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.run_count, 9)
        self.assertEqual(result.provider_call_count, 0)
        for record in result.run_records:
            self.assertEqual(record["provider_invocation_counts"]["candidate_writer_provider_api_calls"], 0)
            self.assertEqual(record["provider_invocation_counts"]["repair_writer_provider_api_calls"], 0)
            self.assertEqual(record["provider_invocation_counts"]["semantic_grounding_provider_api_calls"], 0)
            self.assertEqual(record["provider_invocation_counts"]["quality_evaluator_provider_api_calls"], 0)
            self.assertIn("repair_adapter_diagnostics", record)
            self.assertIsNone(record["repair_adapter_diagnostics"])

    def test_manifest_accounts_for_logical_and_max_live_calls(self) -> None:
        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=default_repair_writer_benchmark_cases(),
                    plans=default_repair_writer_benchmark_plans(),
                    allow_api=False,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )
            manifest = json.loads(Path(result.artifacts.manifest_json).read_text(encoding="utf-8"))

        self.assertEqual(manifest["planned_live_accounting"]["logical_repair_runs"], 9)
        self.assertEqual(manifest["planned_live_accounting"]["planned_repair_writer_calls"], 9)
        self.assertEqual(manifest["planned_live_accounting"]["planned_semantic_grounding_calls"], 9)
        self.assertEqual(manifest["planned_live_accounting"]["planned_quality_evaluator_calls"], 9)
        self.assertEqual(manifest["planned_live_accounting"]["planned_max_provider_calls"], 27)


    def test_benchmark_prompt_render_uses_case_specific_length_guidance(self) -> None:
        cases = default_repair_writer_benchmark_cases()
        expected_lengths = {
            "topic_140_digest_126": 1274,
            "topic_214_digest_128__claude_v3": 1299,
            "topic_200_digest_134__gpt_v2": 1043,
        }

        for case in cases:
            with self.subTest(case=case.case_id):
                guidance = build_repair_writer_benchmark_prompt_render(case).variables[
                    "repair_writer_length_guidance"
                ]

                self.assertIn(
                    f"The original post_text is {expected_lengths[case.case_id]} characters",
                    guidance,
                )
                self.assertIn("Use 1150-1200 characters", guidance)
                if case.candidate_post_character_length >= 1200:
                    self.assertIn("NEAR-LIMIT ORIGINAL:", guidance)
                    self.assertIn("Prefer a net-negative character delta", guidance)
                else:
                    self.assertNotIn("NEAR-LIMIT ORIGINAL:", guidance)
                    self.assertIn("never pad the post to reach the safe target", guidance)


    def test_benchmark_cta_target_metadata_matches_controlled_repair_contract(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        render = build_repair_writer_benchmark_prompt_render(case)
        repair_instruction = json.loads(render.variables["repair_instruction_json"])

        self.assertEqual(repair_instruction["failed_criterion"], "cta")
        self.assertEqual(repair_instruction["target_locality"], "ending_local")
        self.assertEqual(
            repair_instruction["replacement_preference"],
            "replace_or_sharpen_existing_ending_do_not_append",
        )

    def test_manifest_records_genericization_as_selection_blocker_without_selection_change(self) -> None:
        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=default_repair_writer_benchmark_cases(),
                    plans=default_repair_writer_benchmark_plans(),
                    allow_api=False,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )
            manifest = json.loads(Path(result.artifacts.manifest_json).read_text(encoding="utf-8"))

        selection_rule = manifest["repair_writer_selection_rule"]
        self.assertEqual(
            selection_rule["genericization"],
            REPAIR_WRITER_SELECTION_BLOCKER_GENERICIZATION,
        )
        self.assertIn("anti_genericness", selection_rule["criteria"])
        self.assertIn("distinctive_voice_preservation", selection_rule["criteria"])
        self.assertFalse(selection_rule["production_selection_changed"])

    def test_default_plans_have_identical_prompt_input_hash_for_each_case(self) -> None:
        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=default_repair_writer_benchmark_cases(),
                    plans=default_repair_writer_benchmark_plans(),
                    allow_api=False,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )

        by_case: dict[str, set[str]] = {}
        for record in result.run_records:
            by_case.setdefault(record["case_id"], set()).add(
                record["input_parity"]["repair_prompt_input_sha256"]
            )

        self.assertEqual(
            set(by_case),
            {
                "topic_140_digest_126",
                "topic_214_digest_128__claude_v3",
                "topic_200_digest_134__gpt_v2",
            },
        )
        self.assertTrue(all(len(hashes) == 1 for hashes in by_case.values()))

    def test_live_path_uses_only_repair_plan_provider_and_fixed_downstream_roles(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[1]
        calls = {"repair": [], "grounding": [], "quality": []}

        def repair_executor(request):
            calls["repair"].append(request)
            local_repair = case.candidate_payload["post_text"] + " Act?"
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": local_repair}),
                provider=request.provider,
                model=request.model,
                prompt_metadata=PromptMetadata(
                    prompt_name=request.rendered_prompt_input.prompt_name,
                    prompt_version=request.rendered_prompt_input.prompt_version,
                    prompt_path=request.rendered_prompt_input.prompt_path,
                ),
                usage={"total_tokens": 10},
                raw_provider_response={"id": "repair"},
                provider_response_metadata={
                    "provider": request.provider,
                    "model": request.model,
                    "provider_finish_reason": "stop",
                    "provider_stop_reason": None,
                    "provider_max_output_tokens": REPAIR_WRITER_MAX_OUTPUT_TOKENS,
                    "provider_reported_output_tokens": 10,
                    "provider_output_limit_reached": False,
                },
            )

        def grounding_executor(request):
            calls["grounding"].append(request)
            return SemanticGroundingRawResponse(
                raw_text=json.dumps(_passing_grounding_payload()),
                provider=request.provider,
                model=request.model,
                prompt_metadata=None,
                usage={"total_tokens": 11},
                raw_provider_response={"id": "grounding"},
            )

        def quality_executor(request):
            calls["quality"].append(request)
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(_quality_review_payload(passed=True)),
                provider=request.provider,
                model=request.model,
                prompt_metadata=None,
                usage={"total_tokens": 12},
                raw_provider_response={"id": "quality"},
            )

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
                semantic_grounding_executor=grounding_executor,
                quality_evaluator_executor=quality_executor,
            )
            comparison_text = Path(result.artifacts.repair_comparison_md).read_text(
                encoding="utf-8"
            )

        self.assertEqual(result.status, BENCHMARK_STATUS_COMPLETED)
        self.assertEqual(result.provider_call_count, 3)
        self.assertEqual(len(calls["repair"]), 1)
        self.assertEqual(calls["repair"][0].provider, "anthropic")
        self.assertEqual(calls["repair"][0].model, "claude-sonnet-5")
        self.assertEqual(calls["repair"][0].max_output_tokens, REPAIR_WRITER_MAX_OUTPUT_TOKENS)
        self.assertEqual(calls["repair"][0].rendered_prompt_input.to_dict()["variables"]["repair_attempt_json"].count("editorial"), 1)
        self.assertEqual(len(calls["grounding"]), 1)
        self.assertEqual(calls["grounding"][0].provider, FIXED_SEMANTIC_GROUNDING_PROVIDER)
        self.assertEqual(calls["grounding"][0].model, FIXED_SEMANTIC_GROUNDING_MODEL)
        self.assertEqual(len(calls["quality"]), 1)
        self.assertEqual(calls["quality"][0].provider, FIXED_QUALITY_EVALUATOR_PROVIDER)
        self.assertEqual(calls["quality"][0].model, FIXED_QUALITY_EVALUATOR_MODEL)
        record = result.run_records[0]
        diagnostics = record["repair_adapter_diagnostics"]
        self.assertEqual(diagnostics["structural_failure_category"], UNKNOWN)
        self.assertEqual(diagnostics["parsed_top_level_keys"], ["post_text"])
        self.assertTrue(diagnostics["post_text_within_candidate_max_length"])
        self.assertIn("raw_text_sha256", record["response_diagnostics"])
        self.assertIn("raw_text_length", record["response_diagnostics"])
        self.assertEqual(record["response_diagnostics"]["provider_finish_reason"], "stop")
        self.assertEqual(
            record["response_diagnostics"]["provider_max_output_tokens"],
            REPAIR_WRITER_MAX_OUTPUT_TOKENS,
        )
        self.assertFalse(
            record["response_diagnostics"]["provider_output_limit_reached"]
        )
        response_structure = record["response_diagnostics"][
            "response_structure_diagnostics"
        ]
        self.assertEqual(
            response_structure["response_structure_classification"],
            COMPLETE_PLAIN_JSON,
        )
        self.assertIn("raw_response_character_count", response_structure)
        preservation = record["payload_preservation"]
        self.assertEqual(preservation["original_character_count"], case.candidate_post_character_length)
        self.assertEqual(
            preservation["repaired_character_count"],
            len(case.candidate_payload["post_text"] + " Act?"),
        )
        self.assertGreater(preservation["character_delta"], 0)
        self.assertTrue(preservation["within_1300"])
        self.assertFalse(preservation["within_safe_target"])
        self.assertTrue(preservation["near_limit_original"])
        self.assertFalse(preservation["net_shortened"])
        self.assertIn(
            preservation["repair_scope_locality"],
            {"paragraph_local", "sentence_local", "broad_rewrite"},
        )
        self.assertIn("distinctive_fragment_count", preservation)
        self.assertIn("distinctive_fragments_preserved", preservation)
        self.assertIn("distinctive_preservation_rate", preservation)
        self.assertIn("lost_distinctive_fragments", preservation)
        self.assertEqual(preservation["added_generic_marker_count"], 0)
        self.assertEqual(preservation["added_generic_markers"], [])
        self.assertNotIn(case.candidate_payload["post_text"], comparison_text)
        self.assertNotIn(case.candidate_payload["post_text"] + " Act?", comparison_text)
        target = record["target_repair_diagnostics"]
        self.assertEqual(target["failed_criterion"], "cta")
        self.assertTrue(target["target_repair_attempted"])
        self.assertIsNone(target["target_repair_success"])
        self.assertIn("quality score for cta unavailable", target["target_repair_success_reason"])
        self.assertEqual(
            record["downstream_failure_classification"],
            "quality_evaluator_provider_or_parser_infrastructure_failure",
        )

    def test_repair_writer_failure_blocks_downstream_evaluators(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[0]
        calls = {"grounding": 0, "quality": 0}

        def repair_executor(request):
            return RepairWriterRawResponse(
                raw_text="",
                provider=request.provider,
                model=request.model,
                execution_error="empty provider response",
                execution_diagnostics={
                    "provider_error_type": "RateLimitError",
                    "provider_error_code": "rate_limit_exceeded",
                    "provider_http_status": 429,
                    "provider_error_category": PROVIDER_ERROR_RATE_LIMIT,
                    "provider_error_retryable": True,
                    "provider_endpoint_family": "responses",
                    "provider_model": request.model,
                    "provider_error_message_safe": (
                        "provider execution failed; raw exception message omitted"
                    ),
                    "raw_prompt": "secret prompt text",
                    "raw_provider_message": "secret prompt text",
                },
            )

        def grounding_executor(request):
            calls["grounding"] += 1
            raise AssertionError("semantic grounding should not run")

        def quality_executor(request):
            calls["quality"] += 1
            raise AssertionError("quality evaluator should not run")

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
                semantic_grounding_executor=grounding_executor,
                quality_evaluator_executor=quality_executor,
            )

        record = result.run_records[0]
        self.assertEqual(record["failure_code"], "repair_writer_empty_response")
        provider_error = record["response_diagnostics"]["provider_error_diagnostics"]
        self.assertEqual(
            provider_error["provider_error_category"],
            PROVIDER_ERROR_RATE_LIMIT,
        )
        self.assertEqual(provider_error["provider_http_status"], 429)
        self.assertNotIn("prompt", json.dumps(provider_error).lower())
        self.assertNotIn("raw_prompt", provider_error)
        self.assertNotIn("raw_provider_message", provider_error)
        self.assertEqual(calls, {"grounding": 0, "quality": 0})

    def test_downstream_semantic_grounding_parse_failure_records_response_diagnostics(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[2]
        quality_calls = 0

        def repair_executor(request):
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": case.candidate_payload["post_text"] + " Act?"}),
                provider=request.provider,
                model=request.model,
                prompt_metadata=PromptMetadata(
                    prompt_name=request.rendered_prompt_input.prompt_name,
                    prompt_version=request.rendered_prompt_input.prompt_version,
                    prompt_path=request.rendered_prompt_input.prompt_path,
                ),
                usage={"total_tokens": 10},
                raw_provider_response={"id": "repair"},
            )

        def grounding_executor(request):
            return SemanticGroundingRawResponse(
                raw_text='```json\n{"pass": tru',
                provider=request.provider,
                model=request.model,
                usage={
                    "prompt_tokens": 2067,
                    "completion_tokens": 70,
                    "total_tokens": 3863,
                },
                raw_provider_response={"id": "grounding"},
                provider_response_metadata={
                    "provider_finish_reason": "length",
                    "provider_stop_reason": None,
                    "provider_max_output_tokens": 4800,
                    "provider_output_limit_reached": True,
                    "provider_prompt_tokens": 2067,
                    "provider_visible_output_tokens": 70,
                    "provider_hidden_output_tokens": 1726,
                    "provider_combined_output_tokens": 1796,
                    "provider_output_budget_utilization_percent": 37.42,
                },
            )

        def quality_executor(request):
            nonlocal quality_calls
            quality_calls += 1
            raise AssertionError("quality evaluator should not run after grounding parse failure")

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
                semantic_grounding_executor=grounding_executor,
                quality_evaluator_executor=quality_executor,
            )
            runs_text = Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8")

        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "semantic_grounding_parse")
        self.assertEqual(record["failure_code"], "semantic_grounding_parse_failure")
        self.assertEqual(record["provider_invocation_counts"]["semantic_grounding_provider_api_calls"], 1)
        self.assertEqual(record["provider_invocation_counts"]["quality_evaluator_provider_api_calls"], 0)
        self.assertEqual(quality_calls, 0)
        metadata = record["semantic_grounding"]["metadata"]
        diagnostics = metadata["response_diagnostics"]
        self.assertEqual(
            diagnostics["response_structure_classification"],
            SEMANTIC_TRUNCATED_INSIDE_JSON,
        )
        self.assertEqual(diagnostics["provider_finish_reason"], "length")
        self.assertTrue(diagnostics["provider_output_limit_reached"])
        self.assertEqual(diagnostics["provider_visible_output_tokens"], 70)
        self.assertEqual(diagnostics["provider_hidden_output_tokens"], 1726)
        self.assertEqual(diagnostics["provider_combined_output_tokens"], 1796)
        self.assertEqual(diagnostics["provider_output_budget_utilization_percent"], 37.42)
        self.assertEqual(metadata["parser_error_details"]["code"], "malformed_fence")
        self.assertNotIn('{"pass": tru', runs_text)

    def test_downstream_quality_execution_failure_records_provider_diagnostics(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[1]

        def repair_executor(request):
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": case.candidate_payload["post_text"] + " Act?"}),
                provider=request.provider,
                model=request.model,
                prompt_metadata=PromptMetadata(
                    prompt_name=request.rendered_prompt_input.prompt_name,
                    prompt_version=request.rendered_prompt_input.prompt_version,
                    prompt_path=request.rendered_prompt_input.prompt_path,
                ),
                usage={"total_tokens": 10},
                raw_provider_response={"id": "repair"},
            )

        def grounding_executor(request):
            return SemanticGroundingRawResponse(
                raw_text=json.dumps(_passing_grounding_payload()),
                provider=request.provider,
                model=request.model,
                usage={"total_tokens": 11},
                raw_provider_response={"id": "grounding"},
            )

        def quality_executor(request):
            return QualityEvaluatorRawResponse(
                raw_text="",
                provider=request.provider,
                model=request.model,
                execution_error="provider invocation failed",
                execution_diagnostics={
                    "provider_error_type": "RateLimitError",
                    "provider_error_code": "rate_limit_exceeded",
                    "provider_http_status": 429,
                    "provider_error_category": PROVIDER_ERROR_RATE_LIMIT,
                    "provider_error_retryable": True,
                    "provider_endpoint_family": "responses",
                    "provider_model": request.model,
                    "provider_error_message_safe": (
                        "provider execution failed; raw exception message omitted"
                    ),
                    "raw_prompt": "secret prompt text",
                    "raw_provider_message": "secret prompt text",
                },
            )

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
                semantic_grounding_executor=grounding_executor,
                quality_evaluator_executor=quality_executor,
            )

        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "quality_evaluator_execution")
        quality_metadata = record["quality_evaluation"]["metadata"]
        provider_error = quality_metadata["provider_error_diagnostics"]
        self.assertEqual(provider_error["provider_error_category"], PROVIDER_ERROR_RATE_LIMIT)
        self.assertEqual(provider_error["provider_http_status"], 429)
        self.assertNotIn("prompt", json.dumps(provider_error).lower())
        self.assertNotIn("raw_prompt", provider_error)
        self.assertNotIn("raw_provider_message", provider_error)

    def test_gemini_accounting_metadata_is_recorded_in_response_diagnostics(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[2]

        def repair_executor(request):
            return RepairWriterRawResponse(
                raw_text='```json\n{"post_text": "unterminated',
                provider=request.provider,
                model=request.model,
                execution_error=None,
                usage={
                    "prompt_tokens": 2067,
                    "completion_tokens": 70,
                    "total_tokens": 3863,
                },
                raw_provider_response={"id": "gemini"},
                provider_response_metadata={
                    "provider": "gemini",
                    "model": "gemini-3.6-flash",
                    "provider_finish_reason": "length",
                    "provider_max_output_tokens": 1800,
                    "provider_reported_output_tokens": 70,
                    "provider_output_limit_reached": True,
                    "provider_prompt_tokens": 2067,
                    "provider_visible_output_tokens": 70,
                    "provider_total_tokens": 3863,
                    "provider_hidden_output_tokens": 1726,
                    "provider_combined_output_tokens": 1796,
                    "provider_output_budget_utilization_percent": 99.78,
                },
            )

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
            )

        diagnostics = result.run_records[0]["response_diagnostics"]
        self.assertEqual(diagnostics["provider_visible_output_tokens"], 70)
        self.assertEqual(diagnostics["provider_hidden_output_tokens"], 1726)
        self.assertEqual(diagnostics["provider_combined_output_tokens"], 1796)
        self.assertEqual(
            diagnostics["provider_output_budget_utilization_percent"],
            99.78,
        )

    def test_adaptation_failure_records_safe_structural_diagnostics(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[1]
        calls = {"grounding": 0, "quality": 0}
        overlength = "x" * (FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1)

        def repair_executor(request):
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": overlength}),
                provider=request.provider,
                model=request.model,
                prompt_metadata=None,
                usage={"total_tokens": 10},
                raw_provider_response={"id": "repair"},
            )

        def grounding_executor(request):
            calls["grounding"] += 1
            raise AssertionError("semantic grounding should not run")

        def quality_executor(request):
            calls["quality"] += 1
            raise AssertionError("quality evaluator should not run")

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
                semantic_grounding_executor=grounding_executor,
                quality_evaluator_executor=quality_executor,
            )
            runs_text = Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8")

        record = result.run_records[0]
        diagnostics = record["repair_adapter_diagnostics"]
        self.assertEqual(record["failure_stage"], "repair_writer_adaptation")
        self.assertEqual(record["failure_code"], "repair_writer_adaptation_failure")
        self.assertEqual(record["parser_error_details"]["code"], "invalid_candidate_post")
        self.assertEqual(record["adaptation_error_details"]["code"], "invalid_candidate_post")
        self.assertEqual(diagnostics["structural_failure_category"], POST_TEXT_TOO_LONG)
        self.assertEqual(
            diagnostics["post_text_character_count"],
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1,
        )
        self.assertFalse(diagnostics["post_text_within_candidate_max_length"])
        preservation = record["payload_preservation"]
        self.assertIsNotNone(preservation)
        self.assertEqual(
            preservation["repaired_character_count"],
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1,
        )
        self.assertFalse(preservation["within_1300"])
        self.assertEqual(preservation["preservation_basis"], "parsed_post_text_only")
        self.assertEqual(preservation["parsed_top_level_keys"], ["post_text"])
        self.assertEqual(calls, {"grounding": 0, "quality": 0})
        self.assertNotIn(overlength, runs_text)
        self.assertIn("raw_text_sha256", runs_text)
        self.assertIn("raw_text_length", runs_text)
        self.assertNotIn("raw_text\":", runs_text)

    def test_payload_preservation_for_parsed_marks_post_text_only_diagnostic_basis(self) -> None:
        original = {
            "post_text": "Original post text.",
            "hashtags": ["#markets"],
        }
        parsed = {
            "post_text": "Repaired post text.",
            "unexpected": "adapter diagnostics own this structural failure",
        }

        preservation = linkedin_post_repair_writer_benchmark._payload_preservation_for_parsed(
            original,
            parsed,
        )

        self.assertIsNotNone(preservation)
        self.assertEqual(preservation["preservation_basis"], "parsed_post_text_only")
        self.assertEqual(preservation["parsed_top_level_keys"], ["post_text", "unexpected"])
        self.assertFalse(preservation["only_post_text_changed"])

    def test_payload_preservation_reports_added_generic_markers(self) -> None:
        original = {
            "post_text": "Bitcoin market evidence leaves room for doubt. Adoption signals are mixed.",
        }
        repaired = {
            "post_text": "It's easy to call this settled. In my view, Bitcoin market evidence leaves room for doubt.",
        }

        preservation = linkedin_post_repair_writer_benchmark._payload_preservation(
            original,
            repaired,
        )

        self.assertEqual(preservation["added_generic_marker_count"], 2)
        self.assertEqual(
            preservation["added_generic_markers"],
            ["it's easy to", "in my view"],
        )

    def test_payload_preservation_detects_expanded_generic_marker_set(self) -> None:
        original = {
            "post_text": "Bitcoin market evidence leaves room for doubt. Adoption signals are mixed.",
        }
        repaired = {
            "post_text": "For me, I think this is settled. I urge leaders to act. My reading is that it's tempting to overstate momentum.",
        }

        preservation = linkedin_post_repair_writer_benchmark._payload_preservation(
            original,
            repaired,
        )

        self.assertEqual(
            preservation["added_generic_markers"],
            ["it's tempting", "for me", "i think", "i urge", "my reading"],
        )
    def test_payload_preservation_reports_distinctive_fragment_preservation(self) -> None:
        distinctive = "The market is learning to price political enthusiasm without mistaking it for adoption."
        original = {"post_text": distinctive}
        repaired = {"post_text": f"{distinctive} Extra local CTA."}

        preservation = linkedin_post_repair_writer_benchmark._payload_preservation(
            original,
            repaired,
        )

        self.assertGreaterEqual(preservation["distinctive_fragment_count"], 1)
        self.assertGreaterEqual(preservation["distinctive_fragments_preserved"], 1)
        self.assertGreater(preservation["distinctive_preservation_rate"], 0)
        self.assertEqual(preservation["lost_distinctive_fragments"], [])

    def test_within_safe_target_is_reporting_only_not_a_gate(self) -> None:
        original = {"post_text": "x" * 1043}
        repaired = {"post_text": "x" * 1250}

        preservation = linkedin_post_repair_writer_benchmark._payload_preservation(
            original,
            repaired,
        )

        self.assertTrue(preservation["within_1300"])
        self.assertFalse(preservation["within_safe_target"])
        self.assertNotIn(
            "within_safe_target",
            inspect.getsource(linkedin_post_repair_writer_benchmark._gate_passed),
        )
        self.assertNotIn(
            "within_safe_target",
            inspect.getsource(linkedin_post_repair_writer_benchmark._record_gate_passed),
        )

    def test_target_repair_success_derives_from_nested_quality_review_score(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]

        diagnostics = linkedin_post_repair_writer_benchmark._target_repair_diagnostics(
            case=case,
            repair_writer_attempts=1,
            failure_stage=None,
            failure_code=None,
            quality_evaluation={"quality_review": _quality_review_payload(passed=True)},
        )

        self.assertEqual(diagnostics["failed_criterion"], "cta")
        self.assertTrue(diagnostics["target_repair_attempted"])
        self.assertTrue(diagnostics["target_repair_success"])
        self.assertEqual(diagnostics["target_quality_score"], 4)
        self.assertEqual(diagnostics["target_required_minimum"], 4)

    def test_excessive_rewrite_blocks_before_downstream_benchmark_evaluators(self) -> None:
        case = default_repair_writer_benchmark_cases()[2]
        plan = default_repair_writer_benchmark_plans()[2]
        calls = {"grounding": 0, "quality": 0}

        def repair_executor(request):
            return RepairWriterRawResponse(
                raw_text=json.dumps(
                    {
                        "post_text": (
                            "Bitcoin has policy attention, but adoption remains uneven. "
                            "Leaders should avoid treating the market narrative as proof. "
                            "What signal would change your view?"
                        )
                    }
                ),
                provider=request.provider,
                model=request.model,
                usage={"total_tokens": 10},
                raw_provider_response={"id": "repair"},
            )

        def grounding_executor(request):
            calls["grounding"] += 1
            raise AssertionError("semantic grounding should not run")

        def quality_executor(request):
            calls["quality"] += 1
            raise AssertionError("quality evaluator should not run")

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
                semantic_grounding_executor=grounding_executor,
                quality_evaluator_executor=quality_executor,
            )

        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "repair_writer_preservation_gate")
        self.assertEqual(record["failure_code"], "repair_writer_excessive_rewrite")
        self.assertEqual(calls, {"grounding": 0, "quality": 0})
        self.assertEqual(
            record["payload_preservation"]["repair_scope_locality"],
            "broad_rewrite",
        )
        self.assertIn(
            "excessive_rewrite_reason",
            record["payload_preservation"],
        )
        self.assertIsNone(record["target_repair_diagnostics"]["target_repair_success"])

    def test_excessive_rewrite_blocks_low_distinctive_preservation_for_local_repairs(self) -> None:
        case = default_repair_writer_benchmark_cases()[2]
        preservation = {
            "repair_scope_locality": "partial_rewrite",
            "distinctive_fragment_count": 3,
            "distinctive_preservation_rate": 0.49,
        }

        reason = linkedin_post_repair_writer_benchmark._excessive_rewrite_blocker(
            case,
            preservation,
        )

        self.assertEqual(reason, "targeted local repair lost too much distinctive wording")

    def test_excessive_rewrite_gate_does_not_block_non_local_repair_criteria(self) -> None:
        case = default_repair_writer_benchmark_cases()[2]
        non_local_case = replace(
            case,
            repair_instruction={
                **case.repair_instruction,
                "failed_criterion": "human_voice",
            },
        )
        preservation = {
            "repair_scope_locality": "broad_rewrite",
            "distinctive_fragment_count": 3,
            "distinctive_preservation_rate": 0.0,
        }

        reason = linkedin_post_repair_writer_benchmark._excessive_rewrite_blocker(
            non_local_case,
            preservation,
        )

        self.assertIsNone(reason)
    def test_semantic_grounding_execution_failure_is_infrastructure_not_domain_rejection(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[2]

        def repair_executor(request):
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": case.candidate_payload["post_text"] + " Act?"}),
                provider=request.provider,
                model=request.model,
                usage={"total_tokens": 10},
                raw_provider_response={"id": "repair"},
                prompt_metadata=PromptMetadata(
                    prompt_name=request.rendered_prompt_input.prompt_name,
                    prompt_version=request.rendered_prompt_input.prompt_version,
                    prompt_path=request.rendered_prompt_input.prompt_path,
                ),
            )

        def grounding_executor(request):
            return SemanticGroundingRawResponse(
                raw_text="",
                provider=request.provider,
                model=request.model,
                execution_error="provider invocation failed",
            )

        def quality_executor(request):
            raise AssertionError("quality evaluator should not run")

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
                semantic_grounding_executor=grounding_executor,
                quality_evaluator_executor=quality_executor,
            )

        record = result.run_records[0]
        self.assertEqual(record["failure_code"], "semantic_grounding_execution_failure")
        self.assertEqual(
            record["downstream_failure_classification"],
            "semantic_grounding_provider_or_parser_infrastructure_failure",
        )
        self.assertIsNone(record["target_repair_diagnostics"]["target_repair_success"])
        self.assertIn(
            "semantic grounding did not produce quality-evaluator input",
            record["target_repair_diagnostics"]["target_repair_success_reason"],
        )

    def test_parse_failure_does_not_persist_non_json_raw_response_excerpts(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[2]
        raw_prose = "Here is the repaired post with secret live provider prose."

        def repair_executor(request):
            return RepairWriterRawResponse(
                raw_text=raw_prose,
                provider=request.provider,
                model=request.model,
                prompt_metadata=None,
                usage={"total_tokens": 9},
                raw_provider_response={"id": "repair"},
                provider_response_metadata={
                    "provider": request.provider,
                    "model": request.model,
                    "provider_finish_reason": "stop",
                    "provider_max_output_tokens": REPAIR_WRITER_MAX_OUTPUT_TOKENS,
                    "provider_reported_output_tokens": 9,
                    "provider_output_limit_reached": False,
                },
            )

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
            )
            runs_text = Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8")

        diagnostics = result.run_records[0]["response_diagnostics"]
        response_structure = diagnostics["response_structure_diagnostics"]
        self.assertEqual(result.run_records[0]["failure_stage"], "repair_writer_parse")
        self.assertEqual(
            response_structure["response_structure_classification"],
            NON_JSON_RESPONSE,
        )
        self.assertNotIn(raw_prose, runs_text)
        self.assertNotIn("secret live provider prose", runs_text)

    def test_configuration_accepts_gemini_repair_writer_plan_for_benchmark(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(
                        default_repair_writer_benchmark_plans()[0].__class__(
                            "gemini_repair",
                            "gemini",
                            "gemini-3.6-flash",
                        ),
                    ),
                    allow_api=False,
                    output_root=Path(tempdir),
                )
            )

        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.run_count, 1)
        self.assertEqual(result.run_records[0]["provider"], "gemini")
        self.assertEqual(result.run_records[0]["model"], "gemini-3.6-flash")

    def test_gemini_low_reasoning_profile_flows_to_repair_execution_request(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[2].__class__(
            "gemini_low",
            "gemini",
            "gemini-3.6-flash",
            execution_profile=REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING,
        )
        calls = []

        def repair_executor(request):
            calls.append(request)
            return RepairWriterRawResponse(
                raw_text="",
                provider=request.provider,
                model=request.model,
                execution_error="empty provider response",
            )

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                repair_writer_executor=repair_executor,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].reasoning_effort, "low")
        self.assertEqual(
            calls[0].execution_metadata["repair_writer_execution_profile"],
            REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING,
        )
        self.assertEqual(calls[0].execution_metadata["repair_writer_reasoning_effort"], "low")
        config = result.run_records[0]["repair_writer_config"]
        self.assertEqual(config["execution_profile"], REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING)
        self.assertEqual(config["reasoning_effort"], "low")

    def test_gemini_reasoning_calibration_dry_run_uses_three_profiles_without_provider_calls(self) -> None:
        plan_class = default_repair_writer_benchmark_plans()[2].__class__
        plans = (
            plan_class(
                "gemini_repair_default",
                "gemini",
                "gemini-3.6-flash",
                execution_profile=REPAIR_WRITER_EXECUTION_PROFILE_PROVIDER_DEFAULT,
            ),
            plan_class(
                "gemini_repair_minimal_reasoning",
                "gemini",
                "gemini-3.6-flash",
                execution_profile=REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_MINIMAL_REASONING,
            ),
            plan_class(
                "gemini_repair_low_reasoning",
                "gemini",
                "gemini-3.6-flash",
                execution_profile=REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING,
            ),
        )

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    experiment_id="repair-writer-gemini-reasoning-calibration-dry-v1",
                    cases=default_repair_writer_benchmark_cases(),
                    plans=plans,
                    allow_api=False,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )

        self.assertEqual(result.status, BENCHMARK_STATUS_DRY_RUN)
        self.assertEqual(result.run_count, 9)
        self.assertEqual(result.provider_call_count, 0)
        self.assertEqual(
            {record["repair_writer_config"]["reasoning_effort"] for record in result.run_records},
            {None, "minimal", "low"},
        )

    def test_gemini_no_reasoning_profile_is_not_supported(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[2].__class__(
            "gemini_no_reasoning",
            "gemini",
            "gemini-3.6-flash",
            execution_profile="gemini_repair_no_reasoning",
        )

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=False,
                    output_root=Path(tempdir),
                ),
            )

        self.assertEqual(result.status, "config_error")
        self.assertIn("unsupported repair writer execution_profile", result.safe_failure_message)

    def test_reasoning_profiles_are_gemini_only(self) -> None:
        case = default_repair_writer_benchmark_cases()[0]
        plan = default_repair_writer_benchmark_plans()[0].__class__(
            "gpt_low",
            "openai",
            "gpt-4.1-2025-04-14",
            execution_profile=REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING,
        )

        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=(case,),
                    plans=(plan,),
                    allow_api=False,
                    output_root=Path(tempdir),
                ),
            )

        self.assertEqual(result.status, "config_error")
        self.assertIn("supported only for gemini", result.safe_failure_message)

    def test_artifacts_do_not_include_raw_prompts_or_provider_payloads(self) -> None:
        with TemporaryDirectory() as tempdir:
            result = run_repair_writer_benchmark(
                RepairWriterBenchmarkRequest(
                    cases=default_repair_writer_benchmark_cases()[:1],
                    plans=default_repair_writer_benchmark_plans()[:1],
                    allow_api=False,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )
            serialized = json.dumps(result.to_dict(), sort_keys=True)
            runs = Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8")

        self.assertNotIn("prompt_text", serialized)
        self.assertNotIn("raw_provider_response", serialized)
        self.assertNotIn("api_key", runs.lower())

    def test_module_does_not_import_candidate_writer_execution_or_runtime_packaging(self) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_repair_writer_benchmark))
        imports = _imported_modules(tree) | _imported_symbols(tree)

        forbidden = {
            "apps.packaging.models",
            "django.db",
            "services.packaging.generator",
            "services.packaging.linkedin_post_final_post_smoke_runner",
            "services.packaging.linkedin_post_final_post_attempt_execution",
            "execute_final_post_controlled_repair_attempt",
            "execute_candidate_writer_prompt",
            "build_candidate_writer_execution_request",
            "ContentPackage",
        }
        self.assertTrue(forbidden.isdisjoint(imports))


def _passing_grounding_payload() -> dict:
    return {
        "pass": True,
        "claims": [],
        "failed_claim_ids": [],
        "automatic_fail_reason": "",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": False,
        "repair_instructions": [],
    }


def _quality_review_payload(*, passed: bool) -> dict:
    scores = _scores(passed=passed)
    return {
        "scores": scores,
        "total_score": sum(scores.values()),
        "pass": passed,
        "failed_criteria": [] if passed else ["cta"],
        "automatic_fail_reason": "",
        "notes": ["benchmark fake"],
        "criterion_rationales": {
            criterion: {
                "score": score,
                "rationale_summary": "test rationale",
                "evidence": [],
                "improvement_direction": "",
            }
            for criterion, score in scores.items()
        },
        "blocking_factuality_ambiguity": False,
        "human_review_required": False,
        "human_review_reason": "",
    }


def _scores(*, passed: bool) -> dict[str, int]:
    if passed:
        return {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 4,
            "author_point_of_view": 4,
            "human_voice": 4,
            "practical_value": 4,
            "cta": 4,
        }
    return {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 4,
        "author_point_of_view": 4,
        "human_voice": 4,
        "practical_value": 4,
        "cta": 2,
    }


def _fixed_now():
    from datetime import UTC, datetime

    return datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def _imported_modules(tree: ast.AST) -> set[str]:
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def _imported_symbols(tree: ast.AST) -> set[str]:
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
