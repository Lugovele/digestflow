from __future__ import annotations

import io
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from django.core.management import call_command
from django.test import SimpleTestCase

from services.packaging.postflow_product_validation_runner import (
    DEFAULT_EXPERIMENT_ID,
    ProductValidationArtifacts,
    ProductValidationResult,
    STATUS_DRY_RUN,
)

COMMAND_MODULE = "apps.packaging.management.commands.benchmark_postflow_product_validation"


class BenchmarkPostFlowProductValidationCommandTests(SimpleTestCase):
    def test_command_builds_provider_free_default_request(self) -> None:
        fake_runner = Mock(return_value=_result())
        with tempfile.TemporaryDirectory() as tempdir:
            with patch(f"{COMMAND_MODULE}.run_product_validation_benchmark", fake_runner):
                call_command(
                    "benchmark_postflow_product_validation",
                    "--output-root",
                    str(Path(tempdir) / "outputs"),
                    stdout=io.StringIO(),
                )

        request = fake_runner.call_args.args[0]
        self.assertEqual(request.experiment_id, DEFAULT_EXPERIMENT_ID)
        self.assertEqual(request.manifest_path.as_posix(), "tests/fixtures/postflow_product_validation_benchmark/manifest_v1.json")
        self.assertIsNone(request.compare_to)
        self.assertTrue(request.export_human_review)
        self.assertTrue(request.fail_on_corpus_invalid)

    def test_command_has_no_allow_api_argument_or_provider_execution_imports(self) -> None:
        source = Path(
            "apps/packaging/management/commands/benchmark_postflow_product_validation.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("allow_api", source)
        self.assertNotIn("execute_candidate_writer_prompt", source)
        self.assertNotIn("execute_semantic_grounding_prompt", source)
        self.assertNotIn("execute_quality_evaluator_prompt", source)
        self.assertNotIn("execute_repair_writer_prompt", source)

    def test_command_can_disable_fail_on_corpus_invalid(self) -> None:
        fake_runner = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_product_validation_benchmark", fake_runner):
            call_command(
                "benchmark_postflow_product_validation",
                "--no-fail-on-corpus-invalid",
                stdout=io.StringIO(),
            )

        request = fake_runner.call_args.args[0]
        self.assertFalse(request.fail_on_corpus_invalid)

    def test_command_prints_zero_provider_calls(self) -> None:
        output = io.StringIO()
        fake_runner = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_product_validation_benchmark", fake_runner):
            call_command("benchmark_postflow_product_validation", stdout=output)

        text = output.getvalue()
        self.assertIn("provider_calls: 0", text)
        self.assertIn("candidate_writer_provider_calls: 0", text)
        self.assertIn("semantic_grounding_provider_calls: 0", text)
        self.assertIn("quality_evaluator_provider_calls: 0", text)
        self.assertIn("repair_writer_provider_calls: 0", text)

    def test_command_passes_compare_to_and_can_disable_human_review(self) -> None:
        fake_runner = Mock(return_value=_result())
        with tempfile.TemporaryDirectory() as tempdir:
            baseline = Path(tempdir) / "baseline.json"
            baseline.write_text('{"metrics":{"case_count":27}}', encoding="utf-8")
            with patch(f"{COMMAND_MODULE}.run_product_validation_benchmark", fake_runner):
                call_command(
                    "benchmark_postflow_product_validation",
                    "--compare-to",
                    str(baseline),
                    "--no-export-human-review",
                    stdout=io.StringIO(),
                )

        request = fake_runner.call_args.args[0]
        self.assertEqual(request.compare_to, baseline)
        self.assertFalse(request.export_human_review)


def _result() -> ProductValidationResult:
    artifacts = ProductValidationArtifacts(
        output_dir="out",
        corpus_manifest_json="out/corpus_manifest.json",
        run_records_jsonl="out/product_validation_runs.jsonl",
        cases_csv="out/product_validation_cases.csv",
        metrics_json="out/product_validation_metrics.json",
        report_md="out/product_validation_report.md",
        human_review_csv="out/human_review.csv",
        human_review_md="out/human_review.md",
        configuration_fingerprint_json="out/configuration_fingerprint.json",
    )
    return ProductValidationResult(
        status=STATUS_DRY_RUN,
        exit_code=0,
        experiment_id=DEFAULT_EXPERIMENT_ID,
        run_count=27,
        provider_call_count=0,
        corpus_case_count=27,
        artifacts=artifacts,
        metrics={"case_count": 27},
        configuration_fingerprint={"provider_call_count": 0},
    )
