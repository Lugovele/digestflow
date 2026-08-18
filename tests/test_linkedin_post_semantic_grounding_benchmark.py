from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import tempfile

from django.test import SimpleTestCase

from apps.ai.client import GEMINI_SUPPORTED_MODELS
from services.packaging import linkedin_post_semantic_grounding_benchmark as benchmark
from services.packaging.linkedin_post_semantic_grounding_execution import SemanticGroundingRawResponse
from services.packaging.linkedin_post_model_role_policy import OPENAI_FINAL_POST_MODEL

FIXTURE_ROOT = Path("tests/fixtures/linkedin_post_semantic_grounding_benchmark/claude_sonnet_5_v5")
EXPECTED_CANDIDATE_TEXT_SHA256 = {
    "topic_200_digest_134": "ea906b00cac9764267b8aea65026ea701d9718a47e93ae297a9c999e78a884dc",
    "topic_140_digest_126": "776d84ea1c136ba9ca01673e8ffc35576278ead861def801421d0ae905aea15a",
}


class LinkedInPostSemanticGroundingBenchmarkTests(SimpleTestCase):
    def test_fixed_candidate_text_matches_committed_claude_v5_fixture_hashes(self) -> None:
        for case_id, expected_hash in EXPECTED_CANDIDATE_TEXT_SHA256.items():
            with self.subTest(case_id=case_id):
                case = benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / f"{case_id}.json")
                candidate_text = case.candidate_payload["post_text"]
                self.assertEqual(hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(), expected_hash)
                self.assertEqual(case.candidate_post_character_length, len(candidate_text))

    def test_fixture_provenance_is_canonical(self) -> None:
        for case_id in benchmark.CANONICAL_BENCHMARK_CASE_IDS:
            case = benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / f"{case_id}.json")
            self.assertEqual(case.source_experiment_id, benchmark.SOURCE_EXPERIMENT_ID)
            self.assertEqual(case.source_git_commit, benchmark.SOURCE_GIT_COMMIT)
            self.assertEqual(case.writer_provider, "anthropic")
            self.assertEqual(case.writer_model, "claude-sonnet-5")
            self.assertTrue(case.canonical_candidate_valid)

    def test_remote_work_invalid_claude_case_is_explicitly_excluded(self) -> None:
        case = benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_214_digest_128.json")
        self.assertFalse(case.canonical_candidate_valid)
        self.assertEqual(case.candidate_post_character_length, 1387)
        self.assertEqual(case.canonical_exclusion["reason"], "candidate_post_above_hard_max_length")
        self.assertTrue(case.canonical_exclusion["diagnostic_only"])

    def test_default_cases_exclude_noncanonical_remote_work_case(self) -> None:
        cases = benchmark.default_semantic_grounding_benchmark_cases(FIXTURE_ROOT)
        self.assertEqual([case.case_id for case in cases], ["topic_200_digest_134", "topic_140_digest_126"])

    def test_default_plans_vary_only_semantic_grounding_provider_model(self) -> None:
        plans = benchmark.default_semantic_grounding_benchmark_plans()
        self.assertEqual([plan.plan_id for plan in plans], [benchmark.PLAN_GPT_4_1_SEMANTIC_GROUNDING, benchmark.PLAN_GEMINI_3_6_FLASH_SEMANTIC_GROUNDING])
        self.assertEqual(plans[0].provider, "openai")
        self.assertEqual(plans[0].model, OPENAI_FINAL_POST_MODEL)
        self.assertEqual(plans[1].provider, "gemini")
        self.assertEqual(plans[1].model, GEMINI_SUPPORTED_MODELS[0])
        self.assertEqual(
            plans[0].max_output_tokens,
            benchmark.DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(
            plans[1].max_output_tokens,
            benchmark.GEMINI_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
        )
        self.assertGreater(
            plans[1].max_output_tokens,
            plans[0].max_output_tokens,
        )

    def test_prompt_render_is_deterministic_for_fixed_case(self) -> None:
        case = benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_200_digest_134.json")
        first = benchmark.build_semantic_grounding_benchmark_prompt_render(case)
        second = benchmark.build_semantic_grounding_benchmark_prompt_render(case)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(json.loads(first.variables["candidate_payload_json"]), case.candidate_payload)
        self.assertEqual(
            [item["evidence_id"] for item in json.loads(first.variables["selected_evidence_json"])],
            [item["evidence_id"] for item in case.selected_evidence],
        )

    def test_both_grounding_plans_receive_identical_semantic_input_for_case(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            case = benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_140_digest_126.json")
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_identical_input",
                    cases=(case,),
                    plans=benchmark.default_semantic_grounding_benchmark_plans(),
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )
        summaries = [record["semantic_input_summary"] for record in result.run_records]
        self.assertEqual(summaries[0], summaries[1])
        self.assertNotEqual(result.run_records[0]["provider"], result.run_records[1]["provider"])
        self.assertNotEqual(result.run_records[0]["model"], result.run_records[1]["model"])

    def test_dry_run_expands_two_cases_two_plans_to_four_runs_without_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_dry_run",
                    cases=benchmark.default_semantic_grounding_benchmark_cases(FIXTURE_ROOT),
                    plans=benchmark.default_semantic_grounding_benchmark_plans(),
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )
            self.assertEqual(result.status, benchmark.BENCHMARK_STATUS_DRY_RUN)
            self.assertEqual(result.run_count, 4)
            self.assertEqual(result.provider_call_count, 0)
            self.assertTrue(Path(result.artifacts.runs_jsonl).exists())
            self.assertTrue(Path(result.artifacts.summary_csv).exists())
            self.assertTrue(Path(result.artifacts.report_md).exists())
            self.assertTrue(Path(result.artifacts.grounding_comparison_md).exists())
            self.assertTrue(Path(result.artifacts.manifest_json).exists())
            records = [json.loads(line) for line in Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 4)
            self.assertTrue(all(sum(record["provider_invocation_counts"].values()) == 0 for record in records))

    def test_dry_run_records_provider_specific_plan_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_dry_run_budgets",
                    cases=benchmark.default_semantic_grounding_benchmark_cases(FIXTURE_ROOT),
                    plans=benchmark.default_semantic_grounding_benchmark_plans(),
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )

        observed = {
            record["provider"]: record["max_output_tokens"]
            for record in result.run_records
        }

        self.assertEqual(
            observed,
            {
                "openai": benchmark.DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
                "gemini": benchmark.GEMINI_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
            },
        )

    def test_dry_run_records_grounding_result_schema_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_schema",
                    cases=benchmark.default_semantic_grounding_benchmark_cases(FIXTURE_ROOT),
                    plans=benchmark.default_semantic_grounding_benchmark_plans(),
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )
        record = result.run_records[0]
        for key in ("parse_success", "normalization_success", "failure_stage", "failure_code", "grounding_pass", "blocking_claim_count", "human_review_required", "claim_reviews"):
            self.assertIn(key, record)
        self.assertEqual(record["parser_path"], benchmark.SEMANTIC_GROUNDING_PARSER_PATH)
        self.assertEqual(record["normalization_path"], benchmark.SEMANTIC_GROUNDING_NORMALIZATION_PATH)

    def test_allow_api_is_explicit_live_opt_in_with_fake_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_live_opt_in",
                    cases=(benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_200_digest_134.json"),),
                    plans=(benchmark.default_semantic_grounding_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                semantic_grounding_executor=_fake_executor(_review_payload()),
            )
        self.assertEqual(result.status, benchmark.BENCHMARK_STATUS_COMPLETED)
        self.assertEqual(result.run_count, 1)
        self.assertEqual(result.provider_call_count, 1)

    def test_diagnostic_case_cannot_enter_primary_benchmark(self) -> None:
        result = benchmark.run_semantic_grounding_benchmark(
            benchmark.SemanticGroundingBenchmarkRequest(
                experiment_id="semantic_grounding_noncanonical",
                cases=(benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_214_digest_128.json"),),
                plans=benchmark.default_semantic_grounding_benchmark_plans(),
            )
        )
        self.assertEqual(result.status, benchmark.BENCHMARK_STATUS_CONFIG_ERROR)
        self.assertIn("diagnostic-only", result.safe_failure_message)

    def test_artifacts_do_not_contain_secrets_or_raw_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_safe_artifacts",
                    cases=benchmark.default_semantic_grounding_benchmark_cases(FIXTURE_ROOT),
                    plans=benchmark.default_semantic_grounding_benchmark_plans(),
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )
            candidate_texts = [case.candidate_payload["post_text"] for case in benchmark.default_semantic_grounding_benchmark_cases(FIXTURE_ROOT)]
            evidence_texts = [item["evidence_text"] for case in benchmark.default_semantic_grounding_benchmark_cases(FIXTURE_ROOT) for item in case.selected_evidence]
            artifact_text = "\n".join(
                Path(path).read_text(encoding="utf-8")
                for key, path in result.artifacts.to_dict().items()
                if key != "output_dir" and Path(path).is_file()
            )
        for forbidden in ("api_key", "sk-test", "x-api-key", "provider_payload", "provider_reply", "prompt_text"):
            self.assertNotIn(forbidden, artifact_text.lower())
        for raw_text in candidate_texts + evidence_texts:
            self.assertNotIn(raw_text, artifact_text)


    def test_live_path_with_fake_executor_records_grounding_pass_and_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            case = benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_200_digest_134.json")
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_live_pass",
                    cases=(case,),
                    plans=(benchmark.default_semantic_grounding_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                semantic_grounding_executor=_fake_executor(_review_payload()),
            )
        self.assertEqual(result.status, benchmark.BENCHMARK_STATUS_COMPLETED)
        self.assertEqual(result.provider_call_count, 1)
        record = result.run_records[0]
        self.assertEqual(record["execution_status"], benchmark.GROUNDING_COMPLETED_PASS)
        self.assertTrue(record["execution_success"])
        self.assertTrue(record["parse_success"])
        self.assertTrue(record["normalization_success"])
        self.assertTrue(record["grounding_pass"])
        self.assertEqual(record["blocking_claim_count"], 0)
        self.assertEqual(record["provider_invocation_counts"]["semantic_grounding"], 1)
        self.assertEqual(record["provider_invocation_counts"]["candidate_writer"], 0)
        self.assertEqual(record["provider_invocation_counts"]["quality_evaluator"], 0)
        self.assertEqual(record["provider_invocation_counts"]["repair_writer"], 0)
        self.assertEqual(record["provider_invocation_counts"]["publication_packaging"], 0)

    def test_live_path_with_fake_executor_records_grounding_block(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[{**_claim_payload(), "support_status": "unsupported", "severity": "major"}],
            failed_claim_ids=["c1"],
            automatic_fail_reason="unsupported claim",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_live_block",
                    cases=(benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_200_digest_134.json"),),
                    plans=(benchmark.default_semantic_grounding_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                semantic_grounding_executor=_fake_executor(payload),
            )
        record = result.run_records[0]
        self.assertEqual(record["execution_status"], benchmark.GROUNDING_COMPLETED_BLOCK)
        self.assertFalse(record["grounding_pass"])
        self.assertEqual(record["blocking_claim_count"], 1)
        self.assertEqual(record["blocking_claim_ids"], ["c1"])
        self.assertEqual(record["claim_reviews"][0]["support_status"], "unsupported")


    def test_live_mode_validates_prompt_path_before_output_directory_creation(self) -> None:
        case = benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_200_digest_134.json")
        bad_case = benchmark.SemanticGroundingBenchmarkCase(
            **{**case.to_dict(), "fixture_path": case.fixture_path, "selected_evidence": case.selected_evidence, "prompt_metadata": case.prompt_metadata.__class__(
                prompt_name=case.prompt_metadata.prompt_name,
                prompt_version=case.prompt_metadata.prompt_version,
                prompt_path="prompts/linkedin/missing_semantic_grounding_prompt.txt",
            )}
        )
        with tempfile.TemporaryDirectory() as tempdir:
            output_root = Path(tempdir)
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_missing_prompt",
                    cases=(bad_case,),
                    plans=(benchmark.default_semantic_grounding_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=output_root,
                ),
                now_factory=_fixed_now,
                semantic_grounding_executor=_fake_executor(_review_payload()),
            )
            self.assertFalse((output_root / "semantic_grounding_missing_prompt").exists())
        self.assertEqual(result.status, benchmark.BENCHMARK_STATUS_CONFIG_ERROR)
        self.assertEqual(result.provider_call_count, 0)
        self.assertIn("prompt path", result.safe_failure_message)

    def test_live_path_distinguishes_empty_provider_response(self) -> None:
        result = _single_live_result(_raw("", execution_error="empty provider response"))
        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "execution")
        self.assertEqual(record["failure_code"], benchmark.FAILURE_EMPTY_RESPONSE)
        self.assertFalse(record["execution_success"])
        self.assertFalse(record["parse_success"])
        self.assertFalse(record["normalization_success"])

    def test_live_path_distinguishes_provider_execution_failure(self) -> None:
        result = _single_live_result(_raw("", execution_error="provider invocation failed"))
        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "execution")
        self.assertEqual(record["failure_code"], benchmark.FAILURE_PROVIDER_EXECUTION)
        self.assertEqual(record["provider_invocation_counts"]["semantic_grounding"], 1)

    def test_live_path_distinguishes_parse_failure(self) -> None:
        result = _single_live_result(_raw("{not-json"))
        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "parse")
        self.assertEqual(record["failure_code"], benchmark.FAILURE_PARSE)
        self.assertFalse(record["parse_success"])
        self.assertFalse(record["normalization_success"])
        self.assertEqual(
            record["response_diagnostics"]["raw_response_character_count"],
            len("{not-json"),
        )
        self.assertTrue(record["response_diagnostics"]["starts_with_json_object"])
        self.assertFalse(record["response_diagnostics"]["ends_with_json_object"])
        self.assertEqual(record["response_diagnostics"]["json_brace_balance"], 1)
        self.assertEqual(record["response_diagnostics"]["response_structure_classification"], "UNKNOWN")
        self.assertEqual(record["parser_error_details"]["line"], 1)
        self.assertEqual(record["parser_error_details"]["column"], 2)

    def test_live_path_records_safe_provider_metadata_for_parse_failure(self) -> None:
        result = _single_live_result(
            _raw(
                "Before JSON {not-json",
                provider="gemini",
                model="gemini-3.6-flash",
                provider_response_metadata={
                    "provider_finish_reason": "stop",
                    "provider_stop_reason": None,
                    "provider_max_output_tokens": 4800,
                    "provider_output_limit_reached": False,
                    "provider_prompt_tokens": 100,
                    "provider_visible_output_tokens": 50,
                    "provider_hidden_output_tokens": 0,
                    "provider_combined_output_tokens": 50,
                    "provider_output_budget_utilization_percent": 1.04,
                },
            )
        )
        diagnostics = result.run_records[0]["response_diagnostics"]

        self.assertEqual(
            diagnostics["response_structure_classification"],
            "EXTRA_PROSE_AROUND_JSON",
        )
        self.assertEqual(diagnostics["provider_finish_reason"], "stop")
        self.assertEqual(diagnostics["provider_max_output_tokens"], 4800)
        self.assertFalse(diagnostics["provider_output_limit_reached"])
        self.assertEqual(diagnostics["provider_prompt_tokens"], 100)
        self.assertEqual(diagnostics["provider_visible_output_tokens"], 50)
        self.assertEqual(diagnostics["provider_hidden_output_tokens"], 0)
        self.assertEqual(diagnostics["provider_combined_output_tokens"], 50)
        self.assertEqual(diagnostics["provider_output_budget_utilization_percent"], 1.04)

    def test_live_path_allowlists_provider_metadata_for_artifacts(self) -> None:
        result = _single_live_result(
            _raw(
                "{not-json",
                provider="gemini",
                model="gemini-3.6-flash",
                provider_response_metadata={
                    "provider_finish_reason": "stop",
                    "provider_stop_reason": None,
                    "provider_max_output_tokens": 4800,
                    "provider_output_limit_reached": False,
                    "provider_prompt_tokens": 100,
                    "provider_visible_output_tokens": 50,
                    "provider_hidden_output_tokens": 0,
                    "provider_combined_output_tokens": 50,
                    "provider_output_budget_utilization_percent": 1.04,
                    "transport_details": {"authorization": "redacted"},
                    "provider_body": {"raw": "body"},
                    "raw_text_copy": "not allowed",
                },
            )
        )

        diagnostics = result.run_records[0]["response_diagnostics"]
        self.assertEqual(diagnostics["provider_finish_reason"], "stop")
        self.assertEqual(diagnostics["provider_max_output_tokens"], 4800)
        self.assertFalse(diagnostics["provider_output_limit_reached"])
        self.assertEqual(diagnostics["provider_prompt_tokens"], 100)
        self.assertEqual(diagnostics["provider_visible_output_tokens"], 50)
        self.assertEqual(diagnostics["provider_hidden_output_tokens"], 0)
        self.assertEqual(diagnostics["provider_combined_output_tokens"], 50)
        self.assertEqual(diagnostics["provider_output_budget_utilization_percent"], 1.04)
        self.assertNotIn("transport_details", diagnostics)
        self.assertNotIn("provider_body", diagnostics)
        self.assertNotIn("raw_text_copy", diagnostics)

    def test_live_path_allowlists_usage_for_artifacts(self) -> None:
        result = _single_live_result(
            _raw(
                "{not-json",
                provider="gemini",
                model="gemini-3.6-flash",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "transport_details": {"authorization": "redacted"},
                    "raw_text_copy": "not allowed",
                },
            )
        )

        usage = result.run_records[0]["response_diagnostics"]["usage"]
        self.assertEqual(
            usage,
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )

    def test_live_path_distinguishes_normalization_failure(self) -> None:
        payload = _review_payload(claims=[{**_claim_payload(), "supported_evidence_ids": ["unselected"]}])
        result = _single_live_result(_raw(json.dumps(payload)))
        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "normalization")
        self.assertEqual(record["failure_code"], benchmark.FAILURE_NORMALIZATION)
        self.assertTrue(record["parse_success"])
        self.assertFalse(record["normalization_success"])
        self.assertEqual(
            record["normalization_error_details"],
            {
                "message": (
                    "semantic grounding claim c1 references unselected evidence "
                    "ID: unselected"
                )
            },
        )

    def test_live_artifacts_persist_normalized_claim_findings_without_raw_prompt_input(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            case = benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_200_digest_134.json")
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_live_artifacts",
                    cases=(case,),
                    plans=(benchmark.default_semantic_grounding_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                semantic_grounding_executor=_fake_executor(_review_payload()),
            )
            artifact_text = "\n".join(
                Path(path).read_text(encoding="utf-8")
                for key, path in result.artifacts.to_dict().items()
                if key != "output_dir" and Path(path).is_file()
            )
        self.assertIn('"claim_id": "c1"', artifact_text)
        self.assertIn('"support_status": "supported"', artifact_text)
        self.assertIn('"semantic_input_summary"', artifact_text)
        self.assertIn('"response_diagnostics"', artifact_text)
        self.assertNotIn(case.candidate_payload["post_text"], artifact_text)
        for evidence in case.selected_evidence:
            self.assertNotIn(evidence["evidence_text"], artifact_text)

    def test_live_path_uses_same_semantic_input_summary_across_grounding_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_live_same_input",
                    cases=(benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_140_digest_126.json"),),
                    plans=benchmark.default_semantic_grounding_benchmark_plans(),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                semantic_grounding_executor=_fake_executor(_review_payload()),
            )
        self.assertEqual(result.run_count, 2)
        self.assertEqual(result.provider_call_count, 2)
        self.assertEqual(result.run_records[0]["semantic_input_summary"], result.run_records[1]["semantic_input_summary"])
        self.assertNotEqual(result.run_records[0]["provider"], result.run_records[1]["provider"])
        self.assertNotEqual(result.run_records[0]["model"], result.run_records[1]["model"])

    def test_live_path_passes_provider_specific_budget_to_execution_request(self) -> None:
        captured_requests = []

        def fake_executor(request):
            captured_requests.append(request)
            return _raw(json.dumps(_review_payload()), provider=request.provider, model=request.model)

        with tempfile.TemporaryDirectory() as tempdir:
            result = benchmark.run_semantic_grounding_benchmark(
                benchmark.SemanticGroundingBenchmarkRequest(
                    experiment_id="semantic_grounding_live_request_budgets",
                    cases=(
                        benchmark.load_semantic_grounding_benchmark_case(
                            FIXTURE_ROOT / "topic_200_digest_134.json"
                        ),
                    ),
                    plans=benchmark.default_semantic_grounding_benchmark_plans(),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                semantic_grounding_executor=fake_executor,
            )

        request_budgets = {
            request.provider: request.max_output_tokens for request in captured_requests
        }
        record_budgets = {
            record["provider"]: record["max_output_tokens"]
            for record in result.run_records
        }

        self.assertEqual(
            request_budgets,
            {
                "openai": benchmark.DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
                "gemini": benchmark.GEMINI_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
            },
        )
        self.assertEqual(record_budgets, request_budgets)

    def test_module_imports_no_provider_runtime_writer_quality_repair_or_packaging_boundaries(self) -> None:
        source = Path(benchmark.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
        imported_symbols = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names}
        self.assertNotIn("services.packaging.generator", imported_modules)
        self.assertNotIn("apps.packaging.models", imported_modules)
        self.assertNotIn("services.packaging.linkedin_post_candidate_writer_execution", imported_modules)
        self.assertNotIn("services.packaging.linkedin_post_quality_evaluator_execution", imported_modules)
        self.assertNotIn("services.packaging.linkedin_post_repair_writer_execution", imported_modules)
        self.assertNotIn("services.packaging.linkedin_post_semantic_grounding_execution", imported_modules)
        self.assertNotIn("execute_semantic_grounding_prompt", imported_symbols)
        self.assertNotIn("build_semantic_grounding_execution_request", imported_symbols)
        self.assertNotIn("build_ai_client", source)
        self.assertNotIn("Content" + "Package", source)



def _single_live_result(raw_response: SemanticGroundingRawResponse) -> benchmark.SemanticGroundingBenchmarkResult:
    with tempfile.TemporaryDirectory() as tempdir:
        return benchmark.run_semantic_grounding_benchmark(
            benchmark.SemanticGroundingBenchmarkRequest(
                experiment_id="semantic_grounding_single_live",
                cases=(benchmark.load_semantic_grounding_benchmark_case(FIXTURE_ROOT / "topic_200_digest_134.json"),),
                plans=(benchmark.default_semantic_grounding_benchmark_plans()[0],),
                allow_api=True,
                output_root=Path(tempdir),
            ),
            now_factory=_fixed_now,
            semantic_grounding_executor=lambda request: raw_response,
        )


def _fake_executor(payload: dict):
    def execute(request):
        return _raw(json.dumps(payload), provider=request.provider, model=request.model)
    return execute


def _raw(
    raw_text: str,
    *,
    provider: str = "openai",
    model: str = "gpt-4.1-2025-04-14",
    execution_error: str | None = None,
    provider_response_metadata: dict | None = None,
    usage: dict | None = None,
) -> SemanticGroundingRawResponse:
    return SemanticGroundingRawResponse(
        raw_text=raw_text,
        provider=provider,
        model=model,
        execution_error=execution_error,
        provider_response_metadata=provider_response_metadata,
        usage=usage,
    )


def _review_payload(*, passed: bool = True, **overrides) -> dict:
    payload = {
        "pass": passed,
        "claims": [_claim_payload()],
        "failed_claim_ids": [],
        "automatic_fail_reason": "",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": False,
        "repair_instructions": [],
    }
    payload.update(overrides)
    return payload


def _claim_payload() -> dict:
    return {
        "claim_id": "c1",
        "field_name": "post_text",
        "value_index": None,
        "claim_text": "Bitcoin adoption has security and volatility constraints.",
        "claim_type": "attributed_source_claim",
        "support_status": "supported",
        "severity": "info",
        "supported_evidence_ids": ["a0-summary"],
        "required_qualifications": [],
        "missing_qualifications": [],
        "rationale": "Directly supported.",
        "repair_hint": "",
    }


def _fixed_now() -> datetime:
    return datetime(2026, 8, 12, 12, 0, 0, tzinfo=UTC)
