from __future__ import annotations

from datetime import UTC, datetime
import ast
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock

from django.test import SimpleTestCase

from apps.ai.client import AI_THINKING_MODE_DISABLED
from services.packaging import linkedin_post_model_experiment_harness as harness
from services.packaging.linkedin_post_final_post_smoke_runner import (
    FinalPostSmokeRunResult,
    SMOKE_MODE_CONTROLLED_REPAIR,
    SMOKE_MODE_STANDALONE,
    SMOKE_STATUS_DRY_RUN,
)
from services.packaging.linkedin_post_model_role_policy import OPENAI_FINAL_POST_MODEL


class LinkedInPostModelExperimentHarnessTests(SimpleTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.case_path = self.root / "case.json"
        self.case_path.write_text("{}", encoding="utf-8")
        self.case = harness.FinalPostExperimentCase("case_a", self.case_path)
        gpt = harness.FinalPostExperimentRoleModel("openai", OPENAI_FINAL_POST_MODEL)
        self.plan = harness.FinalPostExperimentPlan(
            plan_id="gpt_baseline",
            mode=SMOKE_MODE_STANDALONE,
            candidate_writer=gpt,
            semantic_grounding=gpt,
            quality_evaluator=gpt,
        )

    def test_valid_dry_run_spec_is_accepted_and_writes_artifacts(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_valid",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertEqual(result.status, harness.EXPERIMENT_STATUS_COMPLETED)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.run_count, 1)
        self.assertTrue(Path(result.artifacts.runs_jsonl).exists())
        self.assertTrue(Path(result.artifacts.summary_csv).exists())
        self.assertTrue(Path(result.artifacts.report_md).exists())
        self.assertTrue(Path(result.artifacts.manifest_json).exists())

    def test_dry_run_is_default_and_delegates_to_smoke(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_dry",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        smoke_request = smoke_runner.call_args.args[0]
        self.assertFalse(smoke_request.allow_api)
        self.assertFalse(smoke_request.save_output)
        self.assertFalse(smoke_request.include_raw_responses)
        self.assertEqual(smoke_request.input_path, self.case_path)

    def test_live_mode_requires_explicit_api_authorization(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_live",
                cases=(self.case,),
                plans=(self.plan,),
                allow_api=True,
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertTrue(smoke_runner.call_args.args[0].allow_api)

    def test_policy_invalid_candidate_writer_plan_is_rejected_before_smoke(self) -> None:
        bad_plan = harness.FinalPostExperimentPlan(
            plan_id="bad_candidate",
            mode=SMOKE_MODE_STANDALONE,
            candidate_writer=harness.FinalPostExperimentRoleModel("openai", "gemini-3.6-flash"),
            semantic_grounding=self.plan.semantic_grounding,
            quality_evaluator=self.plan.quality_evaluator,
        )
        result, smoke_runner = self._run_invalid_plan(bad_plan)

        self.assertEqual(result.status, harness.EXPERIMENT_STATUS_CONFIG_ERROR)
        smoke_runner.assert_not_called()
        self.assertFalse((self.root / "outputs" / "exp_invalid").exists())

    def test_policy_invalid_semantic_grounding_plan_is_rejected_before_smoke(self) -> None:
        bad_plan = harness.FinalPostExperimentPlan(
            plan_id="bad_grounding",
            mode=SMOKE_MODE_STANDALONE,
            candidate_writer=self.plan.candidate_writer,
            semantic_grounding=harness.FinalPostExperimentRoleModel("openai", "gemini-3.6-flash"),
            quality_evaluator=self.plan.quality_evaluator,
        )
        result, smoke_runner = self._run_invalid_plan(bad_plan)

        self.assertEqual(result.status, harness.EXPERIMENT_STATUS_CONFIG_ERROR)
        smoke_runner.assert_not_called()

    def test_non_openai_quality_evaluator_is_rejected_before_smoke(self) -> None:
        bad_plan = harness.FinalPostExperimentPlan(
            plan_id="bad_quality",
            mode=SMOKE_MODE_STANDALONE,
            candidate_writer=self.plan.candidate_writer,
            semantic_grounding=self.plan.semantic_grounding,
            quality_evaluator=harness.FinalPostExperimentRoleModel("gemini", "gemini-3.6-flash"),
        )
        result, smoke_runner = self._run_invalid_plan(bad_plan)

        self.assertEqual(result.status, harness.EXPERIMENT_STATUS_CONFIG_ERROR)
        smoke_runner.assert_not_called()

    def test_non_openai_repair_writer_is_rejected_before_smoke(self) -> None:
        bad_plan = harness.FinalPostExperimentPlan(
            plan_id="bad_repair",
            mode=SMOKE_MODE_CONTROLLED_REPAIR,
            candidate_writer=self.plan.candidate_writer,
            semantic_grounding=self.plan.semantic_grounding,
            quality_evaluator=self.plan.quality_evaluator,
            repair_writer=harness.FinalPostExperimentRoleModel("gemini", "gemini-3.6-flash"),
        )
        result, smoke_runner = self._run_invalid_plan(bad_plan)

        self.assertEqual(result.status, harness.EXPERIMENT_STATUS_CONFIG_ERROR)
        smoke_runner.assert_not_called()

    def test_standalone_plan_forbids_repair_writer(self) -> None:
        bad_plan = harness.FinalPostExperimentPlan(
            plan_id="bad_repair_mode",
            mode=SMOKE_MODE_STANDALONE,
            candidate_writer=self.plan.candidate_writer,
            semantic_grounding=self.plan.semantic_grounding,
            quality_evaluator=self.plan.quality_evaluator,
            repair_writer=self.plan.quality_evaluator,
        )
        result, smoke_runner = self._run_invalid_plan(bad_plan)

        self.assertIn("repair_writer is not allowed", result.safe_failure_message)
        smoke_runner.assert_not_called()

    def test_controlled_repair_plan_requires_repair_writer(self) -> None:
        bad_plan = harness.FinalPostExperimentPlan(
            plan_id="missing_repair",
            mode=SMOKE_MODE_CONTROLLED_REPAIR,
            candidate_writer=self.plan.candidate_writer,
            semantic_grounding=self.plan.semantic_grounding,
            quality_evaluator=self.plan.quality_evaluator,
        )
        result, smoke_runner = self._run_invalid_plan(bad_plan)

        self.assertIn("repair_writer is required", result.safe_failure_message)
        smoke_runner.assert_not_called()

    def test_canonical_role_policy_is_reused_for_thinking_mode(self) -> None:
        claude = harness.FinalPostExperimentRoleModel("anthropic", "claude-sonnet-5")
        plan = harness.FinalPostExperimentPlan(
            plan_id="claude_candidate",
            mode=SMOKE_MODE_STANDALONE,
            candidate_writer=claude,
            semantic_grounding=self.plan.semantic_grounding,
            quality_evaluator=self.plan.quality_evaluator,
        )
        smoke_runner = Mock(return_value=_smoke_result(role_diagnostics=[]))
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_thinking",
                cases=(self.case,),
                plans=(plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        candidate = result.run_records[0]["role_diagnostics"][0]
        self.assertEqual(candidate["role"], harness.ROLE_CANDIDATE_WRITER)
        self.assertEqual(candidate["thinking_mode"], AI_THINKING_MODE_DISABLED)

    def test_multiple_cases_expand_deterministically(self) -> None:
        second_path = self.root / "case_b.json"
        second_path.write_text("{}", encoding="utf-8")
        second_case = harness.FinalPostExperimentCase("case_b", second_path)
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_cases",
                cases=(self.case, second_case),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertEqual([record["case_id"] for record in result.run_records], ["case_a", "case_b"])

    def test_repeated_runs_receive_deterministic_indexes(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_repeats",
                cases=(self.case,),
                plans=(self.plan,),
                runs_per_plan=3,
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertEqual([record["run_index"] for record in result.run_records], [1, 2, 3])

    def test_all_specs_validated_before_first_smoke_call(self) -> None:
        missing_case = harness.FinalPostExperimentCase("missing", self.root / "missing.json")
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_missing_case",
                cases=(self.case, missing_case),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertEqual(result.run_count, 0)
        smoke_runner.assert_not_called()
        self.assertFalse((self.root / "outputs" / "exp_missing_case").exists())

    def test_invalid_later_plan_causes_zero_smoke_calls(self) -> None:
        bad_plan = harness.FinalPostExperimentPlan(
            plan_id="bad_later",
            mode=SMOKE_MODE_STANDALONE,
            candidate_writer=self.plan.candidate_writer,
            semantic_grounding=self.plan.semantic_grounding,
            quality_evaluator=harness.FinalPostExperimentRoleModel("anthropic", "claude-sonnet-5"),
        )
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_bad_later",
                cases=(self.case,),
                plans=(self.plan, bad_plan),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertEqual(result.run_count, 0)
        smoke_runner.assert_not_called()

    def test_case_path_failure_causes_zero_smoke_calls(self) -> None:
        missing_case = harness.FinalPostExperimentCase("missing", self.root / "missing.json")
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_case_path",
                cases=(missing_case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertIn("case input path does not exist", result.safe_failure_message)
        smoke_runner.assert_not_called()

    def test_existing_output_directory_collision_fails_before_execution(self) -> None:
        (self.root / "outputs" / "exp_collision").mkdir(parents=True)
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_collision",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertIn("already exists", result.safe_failure_message)
        smoke_runner.assert_not_called()

    def test_sanitized_run_record_contains_allowlisted_fields(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_schema",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )
        record = result.run_records[0]

        self.assertEqual(record["experiment_id"], "exp_schema")
        self.assertEqual(record["case_id"], "case_a")
        self.assertEqual(record["plan_id"], "gpt_baseline")
        self.assertIn("repair_diagnostics", record)
        self.assertEqual(record["final_post_text"], "Accepted final post")

    def test_secret_sentinel_is_excluded_from_artifacts(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result(secret_fields=True))
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_secret",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )
        serialized = json.dumps(result.to_dict(), sort_keys=True)
        for artifact in result.artifacts.to_dict().values():
            if Path(artifact).is_file():
                serialized += Path(artifact).read_text(encoding="utf-8")

        self.assertNotIn("sk-test-secret", serialized)
        self.assertNotIn("provider payload secret", serialized)
        self.assertNotIn("provider reply secret", serialized)

    def test_candidate_writer_structural_diagnostics_are_allowlisted_into_artifacts(
        self,
    ) -> None:
        diagnostics = {
            "schema_version": "1.1",
            "failure_stage": "candidate_writer_adaptation",
            "parser_error_code": None,
            "adapter_error_code": "missing_required_fields",
            "top_level_json_type": "dict",
            "received_top_level_keys": ["post_text"],
            "missing_required_fields": ["hook_variants", "quality_checks"],
            "unexpected_fields": ["debug"],
            "invalid_field_names": [],
            "candidate_text_length": None,
            "diagnostics_truncated": False,
            "redacted_key_count": 0,
            "field_violations": [
                {
                    "field_name": "post_text",
                    "reason_code": "above_max_length",
                    "actual_type": "str",
                    "actual_length": 1501,
                    "minimum_required": None,
                    "maximum_allowed": 1300,
                }
            ],
            "parser_error_detail_code": None,
            "parser_error_line": None,
            "parser_error_column": None,
            "parser_error_position": None,
        }
        smoke_runner = Mock(
            return_value=_smoke_result(
                sanitized_result={
                    "stage_statuses": [
                        {
                            "stage": "candidate_writer_adaptation",
                            "status": "failed",
                            "metadata": {
                                "candidate_writer_structural_diagnostics": diagnostics,
                                "raw_response": "secret raw response",
                                "post_text": "secret post text",
                            },
                        }
                    ],
                    "candidate_payload": {},
                    "provider_response_diagnostics": {},
                }
            )
        )

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_writer_diagnostics",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        record = result.run_records[0]
        self.assertEqual(
            record["candidate_writer_structural_diagnostics"]["adapter_error_code"],
            "missing_required_fields",
        )
        self.assertEqual(record["candidate_writer_missing_field_count"], 2)
        self.assertEqual(record["candidate_writer_unexpected_field_count"], 1)
        self.assertEqual(record["candidate_writer_field_violation_count"], 1)
        self.assertEqual(record["candidate_writer_primary_invalid_field"], "post_text")
        self.assertEqual(
            record["candidate_writer_primary_violation_reason"],
            "above_max_length",
        )
        self.assertEqual(record["candidate_writer_primary_actual_length"], 1501)
        self.assertEqual(record["candidate_writer_primary_maximum_allowed"], 1300)
        runs_text = Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8")
        summary_text = Path(result.artifacts.summary_csv).read_text(encoding="utf-8")
        report_text = Path(result.artifacts.report_md).read_text(encoding="utf-8")
        self.assertIn("candidate_writer_structural_diagnostics", runs_text)
        self.assertIn("field_violations", runs_text)
        self.assertIn("candidate_writer_missing_field_count", summary_text)
        self.assertIn("candidate_writer_primary_violation_reason", summary_text)
        self.assertIn("missing_required_fields", report_text)
        self.assertIn("missing=2", report_text)
        self.assertIn("post_text:above_max_length", report_text)
        self.assertIn("maximum_allowed=1300", report_text)
        self.assertNotIn("secret raw response", runs_text + summary_text + report_text)
        self.assertNotIn("secret post text", runs_text + summary_text + report_text)
        self.assertNotIn("hook_variants", summary_text + report_text)

    def test_candidate_writer_structural_diagnostics_are_resanitized_for_artifacts(
        self,
    ) -> None:
        diagnostics = {
            "schema_version": "1.0",
            "failure_stage": "candidate_writer_adaptation",
            "parser_error_code": None,
            "adapter_error_code": "missing_required_fields",
            "top_level_json_type": "prompt: secret raw response text",
            "received_top_level_keys": ["post_text", "api_key"],
            "missing_required_fields": ["hook_variants"],
            "unexpected_fields": ["provider_payload", "debug"],
            "invalid_field_names": ["post_text"],
            "candidate_text_length": None,
            "diagnostics_truncated": False,
            "redacted_key_count": 0,
        }
        smoke_runner = Mock(
            return_value=_smoke_result(
                sanitized_result={
                    "stage_statuses": [
                        {
                            "stage": "candidate_writer_adaptation",
                            "status": "failed",
                            "metadata": {
                                "candidate_writer_structural_diagnostics": diagnostics,
                            },
                        }
                    ],
                    "candidate_payload": {},
                    "provider_response_diagnostics": {},
                }
            )
        )

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_writer_diagnostics_resanitized",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        record = result.run_records[0]
        diagnostics = record["candidate_writer_structural_diagnostics"]
        self.assertIsNone(diagnostics["top_level_json_type"])
        self.assertEqual(diagnostics["received_top_level_keys"], ["post_text"])
        self.assertEqual(diagnostics["unexpected_fields"], ["debug"])
        self.assertTrue(diagnostics["diagnostics_truncated"])
        artifact_text = (
            Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8")
            + Path(result.artifacts.summary_csv).read_text(encoding="utf-8")
            + Path(result.artifacts.report_md).read_text(encoding="utf-8")
        )
        self.assertNotIn("api_key", artifact_text)
        self.assertNotIn("provider_payload", artifact_text)
        self.assertNotIn("secret raw response text", artifact_text)

    def test_candidate_writer_parser_detail_scalars_are_allowlisted_into_artifacts(
        self,
    ) -> None:
        diagnostics = {
            "schema_version": "1.1",
            "failure_stage": "candidate_writer_parse",
            "parser_error_code": "malformed_json",
            "adapter_error_code": None,
            "top_level_json_type": None,
            "received_top_level_keys": [],
            "missing_required_fields": [],
            "unexpected_fields": [],
            "invalid_field_names": [],
            "candidate_text_length": 21,
            "diagnostics_truncated": False,
            "redacted_key_count": 0,
            "field_violations": [],
            "parser_error_detail_code": "trailing_comma",
            "parser_error_line": 1,
            "parser_error_column": 22,
            "parser_error_position": 21,
        }
        smoke_runner = Mock(
            return_value=_smoke_result(
                sanitized_result={
                    "stage_statuses": [
                        {
                            "stage": "candidate_writer_parse",
                            "status": "failed",
                            "metadata": {
                                "candidate_writer_structural_diagnostics": diagnostics,
                            },
                        }
                    ],
                    "candidate_payload": {},
                    "provider_response_diagnostics": {},
                }
            )
        )

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_writer_parser_detail",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        record = result.run_records[0]
        self.assertEqual(
            record["candidate_writer_parser_error_detail_code"],
            "trailing_comma",
        )
        self.assertEqual(record["candidate_writer_parser_error_line"], 1)
        self.assertEqual(record["candidate_writer_parser_error_column"], 22)
        self.assertEqual(record["candidate_writer_parser_error_position"], 21)
        summary_text = Path(result.artifacts.summary_csv).read_text(encoding="utf-8")
        report_text = Path(result.artifacts.report_md).read_text(encoding="utf-8")
        self.assertIn("candidate_writer_parser_error_detail_code", summary_text)
        self.assertIn("trailing_comma", report_text)

    def test_missing_metrics_remain_null(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result(sanitized_result={}))
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_missing_metrics",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )
        record = result.run_records[0]

        self.assertIsNone(record["semantic_grounding_pass"])
        self.assertIsNone(record["quality_total_score"])
        self.assertIsNone(record["estimated_cost"])

    def test_human_editing_fields_default_to_null(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_human_fields",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )
        record = result.run_records[0]

        self.assertIsNone(record["human_editing_distance"])
        self.assertIsNone(record["human_reviewer_notes"])

    def test_repair_writer_diagnostics_are_explicit_for_standalone(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_repair_diag",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )
        record = result.run_records[0]
        repair_role = record["role_diagnostics"][-1]

        self.assertEqual(repair_role["role"], harness.ROLE_REPAIR_WRITER)
        self.assertEqual(repair_role["validation_status"], "not_applicable")
        self.assertEqual(record["repair_diagnostics"]["schema_status"], "not_applicable")

    def test_artifacts_are_written_to_requested_debug_tree(self) -> None:
        output_root = self.root / "debug_outputs" / "final_post_model_experiments"
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_artifacts",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=output_root,
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertTrue(str(result.artifacts.output_dir).endswith("exp_artifacts"))
        self.assertTrue(Path(result.artifacts.runs_jsonl).exists())

    def test_output_root_inside_repo_must_be_debug_experiment_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_bad_root",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=repo_root / "tests" / "bad_experiments",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertEqual(result.status, harness.EXPERIMENT_STATUS_CONFIG_ERROR)
        smoke_runner.assert_not_called()

    def test_default_output_root_requires_debug_outputs_to_be_ignored(self) -> None:
        self.assertTrue((Path(__file__).resolve().parents[1] / ".gitignore").exists())
        self.assertIn(
            "debug_outputs/",
            (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8"),
        )

    def test_result_output_is_json_serializable(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_json",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)

    def test_module_does_not_import_provider_generator_or_persistence_boundaries(self) -> None:
        source = Path(harness.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("services.packaging.generator", imported_modules)
        self.assertNotIn("apps.packaging.models", imported_modules)
        self.assertNotIn("apps.ai.client.OpenAICompatibleClient", imported_modules)
        self.assertNotIn("apps.ai.client.AnthropicMessagesClient", imported_modules)
        self.assertNotIn("apps.ai.client.build_ai_client", imported_modules)
        self.assertNotIn("Content" + "Package", source)
        self.assertNotIn("objects." + "create", source)
        self.assertNotIn("objects." + "update", source)
        self.assertNotIn("build_ai_client", source)
        self.assertNotIn("OpenAI" + "(", source)
        self.assertNotIn("Anthropic" + "(", source)
        self.assertNotIn("Gemini" + "(", source)

    def _run_invalid_plan(self, bad_plan: harness.FinalPostExperimentPlan):
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_invalid",
                cases=(self.case,),
                plans=(bad_plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )
        return result, smoke_runner


def _fixed_now() -> datetime:
    return datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)


def _smoke_result(
    *,
    sanitized_result: dict | None = None,
    role_diagnostics: list[dict] | None = None,
    secret_fields: bool = False,
) -> FinalPostSmokeRunResult:
    if sanitized_result is None:
        sanitized_result = {
            "role_diagnostics": role_diagnostics
            if role_diagnostics is not None
            else [
                {
                    "role": "candidate_writer",
                    "provider": "openai",
                    "model": OPENAI_FINAL_POST_MODEL,
                    "thinking_mode": "provider_default",
                    "validation_status": "valid",
                },
                {
                    "role": "semantic_grounding",
                    "provider": "openai",
                    "model": OPENAI_FINAL_POST_MODEL,
                    "thinking_mode": "provider_default",
                    "validation_status": "valid",
                },
                {
                    "role": "quality_evaluator",
                    "provider": "openai",
                    "model": OPENAI_FINAL_POST_MODEL,
                    "thinking_mode": "provider_default",
                    "validation_status": "valid",
                },
            ],
            "quality_review": {
                "pass": True,
                "total_score": 41,
                "failed_criteria": [],
            },
            "semantic_grounding_review": {
                "pass": True,
                "blocking_claim_ids": [],
            },
            "candidate_payload": {
                "hook_variants_count": 3,
                "cta_variants_count": 2,
                "hashtags": ["#PostFlow", "#LinkedIn"],
            },
            "final_attempt_outcome": {
                "terminal_outcome": "accepted",
                "terminal_reason": "quality passed",
            },
            "provider_response_diagnostics": {},
        }
    if secret_fields:
        sanitized_result = dict(sanitized_result)
        sanitized_result["api_key"] = "sk-test-secret"
        sanitized_result["provider_payload_secret"] = "provider payload secret"
        sanitized_result["provider_reply_secret"] = "provider reply secret"
    return FinalPostSmokeRunResult(
        status=SMOKE_STATUS_DRY_RUN,
        exit_code=0,
        mode=SMOKE_MODE_STANDALONE,
        input_path="case.json",
        provider_models={
            "candidate_writer": {"provider": "openai", "model": OPENAI_FINAL_POST_MODEL},
            "semantic_grounding": {"provider": "openai", "model": OPENAI_FINAL_POST_MODEL},
            "quality_evaluator": {"provider": "openai", "model": OPENAI_FINAL_POST_MODEL},
        },
        invocation_budget={
            "candidate_writer": 1,
            "semantic_grounding": 1,
            "quality_evaluator": 1,
            "repair_writer": 0,
        },
        invocation_counts={
            "candidate_writer": 0,
            "semantic_grounding": 0,
            "quality_evaluator": 0,
            "repair_writer": 0,
        },
        dry_run=True,
        repair_enabled=False,
        repair_executed=False,
        initial_outcome=None,
        final_outcome=None,
        accepted=False,
        final_post_text="Accepted final post",
        safe_failure_code=None,
        safe_failure_message="",
        deterministic_gate_passed=None,
        quality_passed=None,
        saved_output_path=None,
        sanitized_result=sanitized_result,
    )
