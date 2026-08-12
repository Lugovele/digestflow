from __future__ import annotations

import ast
import io
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase

from services.packaging.linkedin_post_semantic_grounding_benchmark import (
    BENCHMARK_STATUS_CONFIG_ERROR,
    BENCHMARK_STATUS_DRY_RUN,
    SemanticGroundingBenchmarkArtifacts,
    SemanticGroundingBenchmarkResult,
)

COMMAND_MODULE = "apps.packaging.management.commands.benchmark_linkedin_semantic_grounding"


class BenchmarkLinkedInSemanticGroundingCommandTests(SimpleTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_command_defaults_to_dry_run_and_default_cases_plans(self) -> None:
        fake_runner = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_benchmark", fake_runner):
            call_command(
                "benchmark_linkedin_semantic_grounding",
                "--experiment-id", "semantic_grounding_cmd",
                "--output-root", str(self.root / "outputs"),
                stdout=io.StringIO(),
            )
        request = fake_runner.call_args.args[0]
        self.assertFalse(request.allow_api)
        self.assertEqual(len(request.cases), 2)
        self.assertEqual(len(request.plans), 2)

    def test_command_rejects_conflicting_dry_run_and_allow_api(self) -> None:
        with self.assertRaises(CommandError):
            call_command("benchmark_linkedin_semantic_grounding", "--dry-run", "--allow-api", stdout=io.StringIO())

    def test_command_plan_argument_maps_to_grounding_plan(self) -> None:
        fake_runner = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_benchmark", fake_runner):
            call_command(
                "benchmark_linkedin_semantic_grounding",
                "--experiment-id", "semantic_grounding_plan",
                "--plan", "gpt_plan=openai,gpt-4.1-2025-04-14",
                "--output-root", str(self.root / "outputs"),
                stdout=io.StringIO(),
            )
        plan = fake_runner.call_args.args[0].plans[0]
        self.assertEqual(plan.plan_id, "gpt_plan")
        self.assertEqual(plan.provider, "openai")
        self.assertEqual(plan.model, "gpt-4.1-2025-04-14")

    def test_malformed_plan_argument_is_rejected(self) -> None:
        fake_runner = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_benchmark", fake_runner):
            with self.assertRaises(CommandError):
                call_command("benchmark_linkedin_semantic_grounding", "--plan", "bad-plan", stdout=io.StringIO())
        fake_runner.assert_not_called()

    def test_command_output_reports_zero_non_grounding_role_invocations(self) -> None:
        output = io.StringIO()
        fake_runner = Mock(return_value=_result(run_count=4))
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_benchmark", fake_runner):
            call_command(
                "benchmark_linkedin_semantic_grounding",
                "--experiment-id", "semantic_grounding_output",
                "--output-root", str(self.root / "outputs"),
                stdout=output,
            )
        text = output.getvalue()
        self.assertIn("run_count: 4", text)
        self.assertIn("provider_calls: 0", text)
        self.assertIn("candidate_writer_invocations: 0", text)
        self.assertIn("quality_evaluator_invocations: 0", text)
        self.assertIn("repair_writer_invocations: 0", text)
        self.assertIn("publication_packaging_invocations: 0", text)

    def test_config_error_exits_nonzero(self) -> None:
        fake_runner = Mock(return_value=_config_error_result())
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_benchmark", fake_runner):
            with self.assertRaises(SystemExit) as raised:
                call_command("benchmark_linkedin_semantic_grounding", "--experiment-id", "semantic_grounding_config_error", stdout=io.StringIO())
        self.assertEqual(raised.exception.code, 1)

    def test_command_imports_no_provider_smoke_or_runtime_boundaries(self) -> None:
        source = Path("apps/packaging/management/commands/benchmark_linkedin_semantic_grounding.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        self.assertNotIn("apps.ai.client", imported_modules)
        self.assertNotIn("services.packaging.generator", imported_modules)
        self.assertNotIn("services.packaging.linkedin_post_final_post_smoke_runner", imported_modules)
        self.assertNotIn("build_ai_client", source)
        self.assertNotIn("run_final_post_smoke", source)
        self.assertNotIn("Content" + "Package", source)


def _result(*, run_count: int = 1) -> SemanticGroundingBenchmarkResult:
    artifacts = SemanticGroundingBenchmarkArtifacts(
        output_dir="debug_outputs/final_post_semantic_grounding_benchmarks/exp",
        runs_jsonl="debug_outputs/final_post_semantic_grounding_benchmarks/exp/runs.jsonl",
        summary_csv="debug_outputs/final_post_semantic_grounding_benchmarks/exp/summary.csv",
        report_md="debug_outputs/final_post_semantic_grounding_benchmarks/exp/report.md",
        manifest_json="debug_outputs/final_post_semantic_grounding_benchmarks/exp/manifest.json",
        grounding_comparison_md="debug_outputs/final_post_semantic_grounding_benchmarks/exp/grounding_comparison.md",
    )
    return SemanticGroundingBenchmarkResult(
        status=BENCHMARK_STATUS_DRY_RUN,
        exit_code=0,
        experiment_id="exp",
        run_count=run_count,
        provider_call_count=0,
        artifacts=artifacts,
        run_records=(),
    )


def _config_error_result() -> SemanticGroundingBenchmarkResult:
    return SemanticGroundingBenchmarkResult(
        status=BENCHMARK_STATUS_CONFIG_ERROR,
        exit_code=1,
        experiment_id="exp",
        run_count=0,
        provider_call_count=0,
        artifacts=None,
        safe_failure_code="semantic_grounding_benchmark_configuration_error",
        safe_failure_message="configuration error",
        run_records=(),
    )
