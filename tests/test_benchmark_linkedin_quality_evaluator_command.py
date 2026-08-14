from __future__ import annotations

import ast
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase

from apps.packaging.management.commands.benchmark_linkedin_quality_evaluator import _parse_plan
from services.packaging.linkedin_post_quality_evaluator_benchmark import (
    BENCHMARK_STATUS_DRY_RUN,
    DEFAULT_EXPERIMENT_ID,
    GEMINI_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
    QualityEvaluatorBenchmarkArtifacts,
    QualityEvaluatorBenchmarkResult,
)


class BenchmarkLinkedInQualityEvaluatorCommandTests(SimpleTestCase):
    def test_command_defaults_to_dry_run_and_calls_benchmark_runner(self) -> None:
        result = _command_result()
        stdout = StringIO()

        with patch(
            "apps.packaging.management.commands.benchmark_linkedin_quality_evaluator.run_quality_evaluator_benchmark",
            return_value=result,
        ) as runner:
            call_command("benchmark_linkedin_quality_evaluator", stdout=stdout)

        request = runner.call_args.args[0]
        self.assertFalse(request.allow_api)
        self.assertEqual(request.experiment_id, DEFAULT_EXPERIMENT_ID)
        self.assertEqual(len(request.cases), 2)
        self.assertEqual(len(request.plans), 2)
        self.assertIn("status: dry_run", stdout.getvalue())
        self.assertIn("provider_calls: 0", stdout.getvalue())
        self.assertIn("candidate_writer_invocations: 0", stdout.getvalue())
        self.assertIn("semantic_grounding_invocations: 0", stdout.getvalue())
        self.assertIn("repair_writer_invocations: 0", stdout.getvalue())
        self.assertIn("publication_packaging_invocations: 0", stdout.getvalue())

    def test_command_rejects_dry_run_and_allow_api_together(self) -> None:
        with self.assertRaisesRegex(CommandError, "--dry-run and --allow-api"):
            call_command("benchmark_linkedin_quality_evaluator", dry_run=True, allow_api=True)

    def test_parse_plan_uses_provider_budget_default(self) -> None:
        plan = _parse_plan("gemini_quality=gemini,gemini-3.6-flash")

        self.assertEqual(plan.plan_id, "gemini_quality")
        self.assertEqual(plan.provider, "gemini")
        self.assertEqual(plan.model, "gemini-3.6-flash")
        self.assertEqual(plan.max_output_tokens, GEMINI_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS)

    def test_parse_plan_accepts_custom_output_budget(self) -> None:
        plan = _parse_plan("gpt_quality=openai,gpt-4.1-2025-04-14,2600")

        self.assertEqual(plan.max_output_tokens, 2600)

    def test_command_module_does_not_import_provider_or_runtime_execution(self) -> None:
        source = Path(
            "apps/packaging/management/commands/benchmark_linkedin_quality_evaluator.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        forbidden_fragments = (
            "apps.ai.client",
            "generator",
            "final_post_smoke_runner",
            "candidate_writer_execution",
            "semantic_grounding_execution",
            "repair_writer_execution",
        )
        for module_name in imported:
            with self.subTest(module=module_name):
                self.assertFalse(any(fragment in module_name for fragment in forbidden_fragments))


def _command_result() -> QualityEvaluatorBenchmarkResult:
    return QualityEvaluatorBenchmarkResult(
        status=BENCHMARK_STATUS_DRY_RUN,
        exit_code=0,
        experiment_id=DEFAULT_EXPERIMENT_ID,
        run_count=4,
        provider_call_count=0,
        artifacts=QualityEvaluatorBenchmarkArtifacts(
            output_dir="debug_outputs/final_post_quality_evaluator_benchmarks/test",
            runs_jsonl="debug_outputs/final_post_quality_evaluator_benchmarks/test/runs.jsonl",
            summary_csv="debug_outputs/final_post_quality_evaluator_benchmarks/test/summary.csv",
            report_md="debug_outputs/final_post_quality_evaluator_benchmarks/test/report.md",
            manifest_json="debug_outputs/final_post_quality_evaluator_benchmarks/test/manifest.json",
            quality_comparison_md="debug_outputs/final_post_quality_evaluator_benchmarks/test/quality_comparison.md",
        ),
        run_records=(),
    )
