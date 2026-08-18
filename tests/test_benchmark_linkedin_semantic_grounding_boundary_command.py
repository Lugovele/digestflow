from __future__ import annotations

import ast
import io
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase

from services.packaging.linkedin_post_semantic_grounding_boundary_benchmark import (
    DEFAULT_EXPERIMENT_ID,
    SemanticGroundingBoundaryArtifacts,
    SemanticGroundingBoundaryResult,
    STATUS_COMPLETED,
    STATUS_CONFIG_ERROR,
    STATUS_DRY_RUN,
    SemanticGroundingBoundaryExecutionPolicy,
)


COMMAND_MODULE = (
    "apps.packaging.management.commands."
    "benchmark_linkedin_semantic_grounding_boundary"
)


class BenchmarkLinkedInSemanticGroundingBoundaryCommandTests(SimpleTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_command_defaults_to_dry_run_cases_and_plans(self) -> None:
        fake_runner = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_boundary_benchmark", fake_runner):
            call_command(
                "benchmark_linkedin_semantic_grounding_boundary",
                "--output-root",
                str(self.root / "outputs"),
                stdout=io.StringIO(),
            )

        request = fake_runner.call_args.args[0]
        self.assertEqual(request.experiment_id, DEFAULT_EXPERIMENT_ID)
        self.assertFalse(request.allow_api)
        self.assertEqual(len(request.cases), 16)
        self.assertEqual(len(request.plans), 2)

    def test_command_rejects_conflicting_dry_run_and_allow_api(self) -> None:
        with self.assertRaises(CommandError):
            call_command(
                "benchmark_linkedin_semantic_grounding_boundary",
                "--dry-run",
                "--allow-api",
                stdout=io.StringIO(),
            )

    def test_command_plan_argument_accepts_explicit_budget(self) -> None:
        fake_runner = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_boundary_benchmark", fake_runner):
            call_command(
                "benchmark_linkedin_semantic_grounding_boundary",
                "--plan",
                "gemini_grounding=gemini,gemini-3.6-flash,4800",
                "--output-root",
                str(self.root / "outputs"),
                stdout=io.StringIO(),
            )

        plan = fake_runner.call_args.args[0].plans[0]
        self.assertEqual(plan.plan_id, "gemini_grounding")
        self.assertEqual(plan.provider, "gemini")
        self.assertEqual(plan.model, "gemini-3.6-flash")
        self.assertEqual(plan.max_output_tokens, 4800)


    def test_command_plan_argument_accepts_execution_profile(self) -> None:
        fake_runner = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_boundary_benchmark", fake_runner):
            call_command(
                "benchmark_linkedin_semantic_grounding_boundary",
                "--plan",
                "grounding_low=gemini,gemini-3.6-flash,4096,grounding_low_reasoning",
                "--output-root",
                str(self.root / "outputs"),
                stdout=io.StringIO(),
            )

        plan = fake_runner.call_args.args[0].plans[0]
        self.assertEqual(plan.plan_id, "grounding_low")
        self.assertEqual(plan.provider, "gemini")
        self.assertEqual(plan.model, "gemini-3.6-flash")
        self.assertEqual(plan.max_output_tokens, 4096)
        self.assertEqual(plan.execution_profile, "grounding_low_reasoning")
        self.assertEqual(plan.reasoning_effort, "low")

    def test_command_rejects_non_integer_plan_budget(self) -> None:
        with self.assertRaises(CommandError):
            call_command(
                "benchmark_linkedin_semantic_grounding_boundary",
                "--plan",
                "gemini_grounding=gemini,gemini-3.6-flash,not-an-int",
                "--output-root",
                str(self.root / "outputs"),
                stdout=io.StringIO(),
            )


    def test_command_rejects_non_integer_four_field_plan_budget(self) -> None:
        with self.assertRaises(CommandError):
            call_command(
                "benchmark_linkedin_semantic_grounding_boundary",
                "--plan",
                "grounding_low=gemini,gemini-3.6-flash,not-an-int,grounding_low_reasoning",
                "--output-root",
                str(self.root / "outputs"),
                stdout=io.StringIO(),
            )

    def test_command_allow_api_sets_live_opt_in_without_other_agent_options(self) -> None:
        output = io.StringIO()
        fake_runner = Mock(
            return_value=_result(status=STATUS_COMPLETED, provider_call_count=32)
        )
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_boundary_benchmark", fake_runner):
            call_command(
                "benchmark_linkedin_semantic_grounding_boundary",
                "--experiment-id",
                "semantic-grounding-boundary-gpt-vs-gemini-v1",
                "--allow-api",
                "--output-root",
                str(self.root / "outputs"),
                stdout=output,
            )

        request = fake_runner.call_args.args[0]
        self.assertTrue(request.allow_api)
        text = output.getvalue()
        self.assertIn("provider_calls: 32", text)
        self.assertIn("candidate_writer_invocations: 0", text)
        self.assertIn("quality_evaluator_invocations: 0", text)
        self.assertIn("repair_writer_invocations: 0", text)
        self.assertIn("publication_packaging_invocations: 0", text)

        source = Path(
            "apps/packaging/management/commands/"
            "benchmark_linkedin_semantic_grounding_boundary.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("candidate_writer_model", source)
        self.assertNotIn("quality_evaluator_model", source)
        self.assertNotIn("repair_model", source)

    def test_config_error_exits_nonzero(self) -> None:
        fake_runner = Mock(return_value=_config_error_result())
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_boundary_benchmark", fake_runner):
            with self.assertRaises(SystemExit) as raised:
                call_command(
                    "benchmark_linkedin_semantic_grounding_boundary",
                    stdout=io.StringIO(),
                )
        self.assertEqual(raised.exception.code, 1)


    def test_command_passes_retry_failed_from_and_execution_policy(self) -> None:
        fake_runner = Mock(return_value=_result(provider_call_count=0))
        with patch(f"{COMMAND_MODULE}.run_semantic_grounding_boundary_benchmark", fake_runner):
            call_command(
                "benchmark_linkedin_semantic_grounding_boundary",
                "--retry-failed-from",
                str(self.root / "previous"),
                "--inter-call-delay-seconds",
                "0",
                "--output-root",
                str(self.root / "outputs"),
                stdout=io.StringIO(),
            )

        request = fake_runner.call_args.args[0]
        self.assertEqual(request.retry_failed_from, self.root / "previous")
        self.assertIsInstance(
            request.execution_policy,
            SemanticGroundingBoundaryExecutionPolicy,
        )
        self.assertEqual(request.execution_policy.inter_call_delay_seconds, 0)

    def test_command_imports_no_provider_smoke_or_runtime_boundaries(self) -> None:
        source = Path(
            "apps/packaging/management/commands/"
            "benchmark_linkedin_semantic_grounding_boundary.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }

        self.assertNotIn("apps.ai.client", imported_modules)
        self.assertNotIn("services.packaging.generator", imported_modules)
        self.assertNotIn(
            "services.packaging.linkedin_post_final_post_smoke_runner",
            imported_modules,
        )
        self.assertNotIn("build_ai_client", source)
        self.assertNotIn("run_final_post_smoke", source)
        self.assertNotIn("Content" + "Package", source)


def _result(
    *,
    status: str = STATUS_DRY_RUN,
    provider_call_count: int = 0,
) -> SemanticGroundingBoundaryResult:
    artifacts = SemanticGroundingBoundaryArtifacts(
        output_dir="debug_outputs/final_post_semantic_grounding_boundary_benchmarks/exp",
        manifest_json=(
            "debug_outputs/final_post_semantic_grounding_boundary_benchmarks/"
            "exp/manifest.json"
        ),
        runs_jsonl=(
            "debug_outputs/final_post_semantic_grounding_boundary_benchmarks/"
            "exp/runs.jsonl"
        ),
        summary_csv=(
            "debug_outputs/final_post_semantic_grounding_boundary_benchmarks/"
            "exp/summary.csv"
        ),
        boundary_metrics_json=(
            "debug_outputs/final_post_semantic_grounding_boundary_benchmarks/"
            "exp/boundary_metrics.json"
        ),
        boundary_comparison_md=(
            "debug_outputs/final_post_semantic_grounding_boundary_benchmarks/"
            "exp/boundary_comparison.md"
        ),
        report_md=(
            "debug_outputs/final_post_semantic_grounding_boundary_benchmarks/"
            "exp/report.md"
        ),
    )
    return SemanticGroundingBoundaryResult(
        status=status,
        exit_code=0,
        experiment_id="exp",
        run_count=32,
        provider_call_count=provider_call_count,
        planned_provider_call_count=32,
        artifacts=artifacts,
        run_records=(),
        boundary_metrics={"plans": {}},
    )


def _config_error_result() -> SemanticGroundingBoundaryResult:
    return SemanticGroundingBoundaryResult(
        status=STATUS_CONFIG_ERROR,
        exit_code=1,
        experiment_id="exp",
        run_count=0,
        provider_call_count=0,
        planned_provider_call_count=0,
        artifacts=None,
        safe_failure_code="semantic_grounding_boundary_configuration_error",
        safe_failure_message="configuration error",
        run_records=(),
    )
