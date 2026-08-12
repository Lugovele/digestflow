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

    def test_live_api_is_explicitly_not_supported_in_scope_23a(self) -> None:
        result = benchmark.run_semantic_grounding_benchmark(
            benchmark.SemanticGroundingBenchmarkRequest(
                experiment_id="semantic_grounding_live_blocked",
                cases=benchmark.default_semantic_grounding_benchmark_cases(FIXTURE_ROOT),
                plans=benchmark.default_semantic_grounding_benchmark_plans(),
                allow_api=True,
            )
        )
        self.assertEqual(result.status, benchmark.BENCHMARK_STATUS_CONFIG_ERROR)
        self.assertEqual(result.run_count, 0)
        self.assertIn("dry-run only", result.safe_failure_message)

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

    def test_module_imports_no_provider_runtime_writer_quality_repair_or_packaging_boundaries(self) -> None:
        source = Path(benchmark.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imported_symbols = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
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



def _fixed_now() -> datetime:
    return datetime(2026, 8, 12, 12, 0, 0, tzinfo=UTC)
