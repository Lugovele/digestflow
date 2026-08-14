from __future__ import annotations

import ast
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
    REPAIR_WRITER_JSON_MODE,
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
from services.packaging.linkedin_post_repair_writer_structural_diagnostics import (
    COMPLETE_PLAIN_JSON,
    NON_JSON_RESPONSE,
    POST_TEXT_TOO_LONG,
    UNKNOWN,
)
from services.packaging.linkedin_post_semantic_grounding_execution import (
    SemanticGroundingRawResponse,
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
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": "Repaired post with a clearer reflective ending."}),
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
        self.assertNotIn(case.candidate_payload["post_text"], comparison_text)
        self.assertNotIn("Repaired post with a clearer reflective ending.", comparison_text)

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
        self.assertEqual(calls, {"grounding": 0, "quality": 0})

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
        self.assertEqual(calls, {"grounding": 0, "quality": 0})
        self.assertNotIn(overlength, runs_text)
        self.assertIn("raw_text_sha256", runs_text)
        self.assertIn("raw_text_length", runs_text)
        self.assertNotIn("raw_text\":", runs_text)

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
