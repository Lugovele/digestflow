from __future__ import annotations

import ast
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase

from services.packaging.linkedin_post_model_role_policy import OPENAI_FINAL_POST_MODEL
from services.packaging.linkedin_post_model_experiment_harness import (
    FinalPostModelExperimentArtifacts,
    FinalPostModelExperimentResult,
    EXPERIMENT_STATUS_COMPLETED,
    EXPERIMENT_STATUS_CONFIG_ERROR,
)

COMMAND_MODULE = "apps.packaging.management.commands.experiment_linkedin_final_post_models"


class ExperimentLinkedInFinalPostModelsCommandTests(SimpleTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.case_path = self.root / "case.json"
        self.case_path.write_text("{}", encoding="utf-8")

    def test_command_defaults_to_dry_run(self) -> None:
        fake_harness = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            call_command(
                "experiment_linkedin_final_post_models",
                "--experiment-id",
                "exp_cmd_dry",
                "--case",
                f"case_a={self.case_path}",
                stdout=io.StringIO(),
            )

        request = fake_harness.call_args.args[0]
        self.assertFalse(request.allow_api)

    def test_command_refuses_conflicting_dry_run_and_allow_api(self) -> None:
        with self.assertRaises(CommandError):
            call_command(
                "experiment_linkedin_final_post_models",
                "--experiment-id",
                "exp_conflict",
                "--case",
                f"case_a={self.case_path}",
                "--dry-run",
                "--allow-api",
                stdout=io.StringIO(),
            )

    def test_explicit_dry_run_invokes_harness_safely(self) -> None:
        fake_harness = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            call_command(
                "experiment_linkedin_final_post_models",
                "--experiment-id",
                "exp_explicit_dry",
                "--case",
                f"case_a={self.case_path}",
                "--dry-run",
                stdout=io.StringIO(),
            )

        self.assertFalse(fake_harness.call_args.args[0].allow_api)

    def test_allow_api_must_be_explicit_for_live_request(self) -> None:
        fake_harness = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            call_command(
                "experiment_linkedin_final_post_models",
                "--experiment-id",
                "exp_live",
                "--case",
                f"case_a={self.case_path}",
                "--allow-api",
                stdout=io.StringIO(),
            )

        self.assertTrue(fake_harness.call_args.args[0].allow_api)

    def test_malformed_case_argument_is_rejected(self) -> None:
        fake_harness = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            with self.assertRaises(CommandError):
                call_command(
                    "experiment_linkedin_final_post_models",
                    "--experiment-id",
                    "exp_bad_case",
                    "--case",
                    "missing_equals",
                    stdout=io.StringIO(),
                )
        fake_harness.assert_not_called()

    def test_malformed_plan_argument_is_rejected(self) -> None:
        fake_harness = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            with self.assertRaises(CommandError):
                call_command(
                    "experiment_linkedin_final_post_models",
                    "--experiment-id",
                    "exp_bad_plan",
                    "--case",
                    f"case_a={self.case_path}",
                    "--plan",
                    "bad-plan",
                    stdout=io.StringIO(),
                )
        fake_harness.assert_not_called()

    def test_invalid_runs_per_plan_is_passed_to_harness_for_atomic_validation(self) -> None:
        fake_harness = Mock(return_value=_config_error_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            with self.assertRaises(SystemExit) as raised:
                call_command(
                    "experiment_linkedin_final_post_models",
                    "--experiment-id",
                    "exp_bad_runs",
                    "--case",
                    f"case_a={self.case_path}",
                    "--runs-per-plan",
                    "0",
                    stdout=io.StringIO(),
                )

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(fake_harness.call_args.args[0].runs_per_plan, 0)

    def test_invalid_specification_causes_zero_reported_runs(self) -> None:
        output = io.StringIO()
        fake_harness = Mock(return_value=_config_error_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            with self.assertRaises(SystemExit):
                call_command(
                    "experiment_linkedin_final_post_models",
                    "--experiment-id",
                    "exp_invalid_spec",
                    "--case",
                    f"case_a={self.case_path}",
                    stdout=output,
                )

        self.assertIn("run_count: 0", output.getvalue())

    def test_command_output_contains_no_secret_values(self) -> None:
        output = io.StringIO()
        fake_harness = Mock(
            return_value=_result(safe_failure_message="redacted safe message")
        )
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            call_command(
                "experiment_linkedin_final_post_models",
                "--experiment-id",
                "exp_secret_output",
                "--case",
                f"case_a={self.case_path}",
                stdout=output,
            )

        self.assertNotIn("sk-test-secret", output.getvalue())
        self.assertNotIn("provider payload secret", output.getvalue())
        self.assertIn("redacted safe message", output.getvalue())

    def test_command_output_contains_concise_experiment_summary(self) -> None:
        output = io.StringIO()
        fake_harness = Mock(return_value=_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            call_command(
                "experiment_linkedin_final_post_models",
                "--experiment-id",
                "exp_summary",
                "--case",
                f"case_a={self.case_path}",
                stdout=output,
            )

        text = output.getvalue()
        self.assertIn("=== POSTFLOW FINAL POST MODEL EXPERIMENT ===", text)
        self.assertIn("accepted_runs:", text)
        self.assertIn("failed_runs:", text)

    def test_command_returns_failure_status_for_configuration_errors(self) -> None:
        fake_harness = Mock(return_value=_config_error_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            with self.assertRaises(SystemExit) as raised:
                call_command(
                    "experiment_linkedin_final_post_models",
                    "--experiment-id",
                    "exp_config_error",
                    "--case",
                    f"case_a={self.case_path}",
                    stdout=io.StringIO(),
                )

        self.assertEqual(raised.exception.code, 1)

    def test_no_output_directory_is_created_by_command_on_preflight_failure(self) -> None:
        output_root = self.root / "outputs"
        fake_harness = Mock(return_value=_config_error_result())
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            with self.assertRaises(SystemExit):
                call_command(
                    "experiment_linkedin_final_post_models",
                    "--experiment-id",
                    "exp_no_output",
                    "--case",
                    f"case_a={self.case_path}",
                    "--output-root",
                    str(output_root),
                    stdout=io.StringIO(),
                )

        self.assertFalse((output_root / "exp_no_output").exists())

    def test_no_provider_or_smoke_lower_level_calls_are_imported_by_command(self) -> None:
        source = Path(
            "apps/packaging/management/commands/experiment_linkedin_final_post_models.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("apps.ai.client", imported_modules)
        self.assertNotIn("services.packaging.generator", imported_modules)
        self.assertNotIn("services.packaging.linkedin_post_final_post_smoke_runner", imported_modules)
        self.assertNotIn("build_ai_client", source)
        self.assertNotIn("OpenAI" + "(", source)
        self.assertNotIn("Anthropic" + "(", source)
        self.assertNotIn("Gemini" + "(", source)

    def test_plan_argument_maps_to_explicit_role_plan(self) -> None:
        fake_harness = Mock(return_value=_result())
        plan_arg = ",".join(
            [
                "standalone",
                "openai",
                OPENAI_FINAL_POST_MODEL,
                "openai",
                OPENAI_FINAL_POST_MODEL,
                "openai",
                OPENAI_FINAL_POST_MODEL,
            ]
        )
        with patch(f"{COMMAND_MODULE}.run_linkedin_final_post_model_experiment", fake_harness):
            call_command(
                "experiment_linkedin_final_post_models",
                "--experiment-id",
                "exp_plan",
                "--case",
                f"case_a={self.case_path}",
                "--plan",
                f"plan_a={plan_arg}",
                stdout=io.StringIO(),
            )

        plan = fake_harness.call_args.args[0].plans[0]
        self.assertEqual(plan.plan_id, "plan_a")
        self.assertEqual(plan.candidate_writer.provider, "openai")
        self.assertEqual(plan.repair_writer, None)


def _result(*, safe_failure_message: str = "") -> FinalPostModelExperimentResult:
    artifacts = FinalPostModelExperimentArtifacts(
        output_dir="debug_outputs/final_post_model_experiments/exp",
        runs_jsonl="debug_outputs/final_post_model_experiments/exp/runs.jsonl",
        summary_csv="debug_outputs/final_post_model_experiments/exp/summary.csv",
        report_md="debug_outputs/final_post_model_experiments/exp/report.md",
        manifest_json="debug_outputs/final_post_model_experiments/exp/manifest.json",
    )
    return FinalPostModelExperimentResult(
        status=EXPERIMENT_STATUS_COMPLETED,
        exit_code=0,
        experiment_id="exp",
        run_count=1,
        artifacts=artifacts,
        safe_failure_message=safe_failure_message,
        run_records=({"accepted": True},),
    )


def _config_error_result() -> FinalPostModelExperimentResult:
    return FinalPostModelExperimentResult(
        status=EXPERIMENT_STATUS_CONFIG_ERROR,
        exit_code=1,
        experiment_id="exp",
        run_count=0,
        artifacts=None,
        safe_failure_code="experiment_configuration_error",
        safe_failure_message="configuration error",
        run_records=(),
    )
