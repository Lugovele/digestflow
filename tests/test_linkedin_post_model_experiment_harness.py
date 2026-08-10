from __future__ import annotations

import csv
from datetime import UTC, datetime
import ast
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock

from django.test import SimpleTestCase

from apps.ai.client import AI_THINKING_MODE_DISABLED
from services.packaging import linkedin_post_model_experiment_harness as harness
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FAILURE_CANDIDATE_WRITER_ADAPTATION,
    STAGE_CANDIDATE_WRITER_ADAPTATION,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
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
        self.assertEqual(record["candidate_post_length"], len("Accepted final post"))
        self.assertEqual(record["hook_variant_count"], 0)
        self.assertEqual(record["cta_variant_count"], 0)
        self.assertEqual(record["hashtag_count"], 0)

    def test_writer_comparison_plans_create_six_durable_run_records(self) -> None:
        cases = tuple(
            harness.FinalPostExperimentCase(f"case_{index}", self.root / f"case_{index}.json")
            for index in range(1, 4)
        )
        for case in cases:
            case.input_path.write_text("{}", encoding="utf-8")
        plans = (_claude_writer_plan(), self.plan)
        smoke_runner = Mock(
            side_effect=[
                _smoke_result(
                    post_text=f"Candidate post {case.case_id} {plan.plan_id}",
                    candidate_provider=plan.candidate_writer.provider,
                    candidate_model=plan.candidate_writer.model,
                )
                for case in cases
                for plan in plans
            ]
        )

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="writer_compare",
                cases=cases,
                plans=plans,
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        self.assertEqual(result.run_count, 6)
        self.assertEqual(len({record["run_id"] for record in result.run_records}), 6)
        self.assertEqual(
            {record["candidate_writer_model"] for record in result.run_records},
            {"claude-sonnet-5", OPENAI_FINAL_POST_MODEL},
        )
        self.assertNotIn(
            "gemini-3.6-flash",
            json.dumps(result.to_dict(), sort_keys=True),
        )

    def test_writer_experiment_records_initial_candidate_post_text(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result(post_text="Canonical candidate post"))

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_candidate_text",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        record = result.run_records[0]
        self.assertEqual(record["candidate_post_text"], "Canonical candidate post")
        self.assertEqual(record["candidate_post_character_length"], 24)
        self.assertEqual(record["parsed_candidate_post_text"], "Canonical candidate post")
        self.assertEqual(record["parsed_candidate_post_character_length"], 24)
        self.assertEqual(record["text_source_status"], "CANONICAL_VALID")
        self.assertTrue(record["candidate_post_within_limit"])
        self.assertTrue(record["candidate_parse_success"])
        self.assertTrue(record["candidate_adapter_success"])
        runs_text = Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8")
        self.assertIn("Canonical candidate post", runs_text)
        self.assertIn('"candidate_post_text"', runs_text)

    def test_writer_experiment_records_parsed_invalid_candidate_post_text(self) -> None:
        parsed_text = "Invalid but parseable candidate " + "x" * 1301
        smoke_runner = Mock(
            return_value=_smoke_result(
                sanitized_result=_adaptation_failure_sanitized_result(parsed_text),
                safe_failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
                safe_failure_message=(
                    "parsed candidate does not satisfy FinalPostPayload structure."
                ),
                final_post_text="",
                quality_passed=None,
            )
        )

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_parsed_invalid_text",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        record = result.run_records[0]
        self.assertEqual(record["candidate_post_text"], "")
        self.assertIsNone(record["candidate_post_character_length"])
        self.assertEqual(record["parsed_candidate_post_text"], parsed_text)
        self.assertEqual(record["parsed_candidate_post_character_length"], len(parsed_text))
        self.assertEqual(
            record["characters_over_limit"],
            len(parsed_text) - FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
        )
        self.assertEqual(
            record["text_source_status"],
            "PARSED_INVALID - above hard length maximum",
        )
        self.assertEqual(record["candidate_writer_hard_failure_category"], "ADAPTATION_FAILURE")
        self.assertTrue(record["candidate_parse_success"])
        self.assertFalse(record["candidate_adapter_success"])
        runs_text = Path(result.artifacts.runs_jsonl).read_text(encoding="utf-8")
        self.assertIn('"parsed_candidate_post_text"', runs_text)
        self.assertIn(parsed_text, runs_text)

    def test_writer_comparison_displays_parsed_invalid_candidate_text(self) -> None:
        parsed_text = "Parsed invalid markdown candidate " + "y" * 1301
        smoke_runner = Mock(
            return_value=_smoke_result(
                sanitized_result=_adaptation_failure_sanitized_result(parsed_text),
                safe_failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
                safe_failure_message=(
                    "parsed candidate does not satisfy FinalPostPayload structure."
                ),
                final_post_text="",
                quality_passed=None,
            )
        )

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_parsed_invalid_markdown",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        comparison_text = Path(result.artifacts.writer_comparison_md).read_text(
            encoding="utf-8"
        )
        self.assertIn("Candidate A", comparison_text)
        self.assertIn("text_source_status: PARSED_INVALID - above hard length maximum", comparison_text)
        self.assertIn(f"character_length: {len(parsed_text)}", comparison_text)
        self.assertIn(
            f"characters_over_limit: {len(parsed_text) - FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS}",
            comparison_text,
        )
        self.assertIn("failure_stage: candidate_writer_adaptation", comparison_text)
        report_text = Path(result.artifacts.report_md).read_text(encoding="utf-8")
        self.assertIn("ADAPTATION_FAILURE", report_text)
        self.assertIn("above_max_length", report_text)
        self.assertIn(parsed_text, comparison_text)
        self.assertNotIn("gpt-4.1-2025-04-14", comparison_text.split("```text", 1)[0])

    def test_summary_csv_exposes_parsed_invalid_candidate_length_scalars(self) -> None:
        parsed_text = "Parsed invalid csv candidate " + "z" * 1301
        smoke_runner = Mock(
            return_value=_smoke_result(
                sanitized_result=_adaptation_failure_sanitized_result(parsed_text),
                safe_failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
                safe_failure_message=(
                    "parsed candidate does not satisfy FinalPostPayload structure."
                ),
                final_post_text="",
                quality_passed=None,
            )
        )

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_parsed_invalid_csv",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        with Path(result.artifacts.summary_csv).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        row = rows[0]
        self.assertEqual(row["candidate_post_character_length"], "")
        self.assertEqual(row["parsed_candidate_post_character_length"], str(len(parsed_text)))
        self.assertEqual(
            row["characters_over_limit"],
            str(len(parsed_text) - FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS),
        )
        self.assertEqual(
            row["text_source_status"],
            "PARSED_INVALID - above hard length maximum",
        )

    def test_writer_comparison_artifact_preserves_utf8_punctuation(self) -> None:
        parsed_text = (
            "Human text with \u2019smart quotes\u2019, caf\u00e9, "
            "and an em dash \u2014 kept."
        )
        smoke_runner = Mock(
            return_value=_smoke_result(
                sanitized_result=_adaptation_failure_sanitized_result(parsed_text),
                safe_failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
                safe_failure_message=(
                    "parsed candidate does not satisfy FinalPostPayload structure."
                ),
                final_post_text="",
                quality_passed=None,
            )
        )

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_utf8_writer_artifact",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        data = Path(result.artifacts.writer_comparison_md).read_bytes()
        decoded = data.decode("utf-8")
        self.assertIn(parsed_text, decoded)
        self.assertNotIn("\u0432\u0402", decoded)

    def test_grounding_and_quality_are_projected_for_writer_comparison(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())

        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_projection",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )
        record = result.run_records[0]

        self.assertTrue(record["semantic_grounding_pass"])
        self.assertFalse(record["semantic_grounding_automatic_fail"])
        self.assertFalse(record["semantic_grounding_human_review_required"])
        self.assertEqual(record["blocking_claim_count"], 0)
        self.assertEqual(record["unsupported_claim_count"], 1)
        self.assertEqual(record["contradicted_claim_count"], 1)
        self.assertEqual(record["grounding_repair_instruction_count"], 1)
        self.assertTrue(record["quality_pass"])
        self.assertEqual(record["quality_total_score"], 41)
        self.assertEqual(record["quality_human_voice_score"], 5)
        self.assertEqual(record["quality_author_point_of_view_score"], 4)

    def test_summary_csv_exposes_writer_comparison_scalars(self) -> None:
        smoke_runner = Mock(
            return_value=_smoke_result(
                post_text="Canonical candidate post is longer",
                final_post_text="Final post",
                invocation_counts={
                    "candidate_writer": 1,
                    "semantic_grounding": 1,
                    "quality_evaluator": 1,
                    "repair_writer": 0,
                }
            )
        )
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_summary_scalars",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        with Path(result.artifacts.summary_csv).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        row = rows[0]
        self.assertEqual(row["writer_model"], OPENAI_FINAL_POST_MODEL)
        self.assertEqual(row["post_length"], str(len("Final post")))
        self.assertEqual(
            row["candidate_post_character_length"],
            str(len("Canonical candidate post is longer")),
        )
        self.assertEqual(row["within_length_limit"], "True")
        self.assertEqual(row["grounding_pass"], "True")
        self.assertEqual(row["blocking_claim_count"], "0")
        self.assertEqual(row["quality_total_score"], "41")
        self.assertEqual(row["author_point_of_view_score"], "4")
        self.assertEqual(row["human_voice_score"], "5")
        self.assertEqual(row["provider_call_count"], "3")

    def test_writer_comparison_markdown_contains_posts_and_blank_review_fields(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result(post_text="Human review candidate"))
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_writer_markdown",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        comparison_text = Path(result.artifacts.writer_comparison_md).read_text(
            encoding="utf-8"
        )
        self.assertIn("Candidate A", comparison_text)
        self.assertIn("Human review candidate", comparison_text)
        self.assertIn("preferred_candidate:\nhuman_voice:\nspecificity:", comparison_text)
        self.assertNotIn("preferred_candidate: Candidate", comparison_text)

    def test_report_retains_case_level_results_and_descriptive_summary(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_report",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        report_text = Path(result.artifacts.report_md).read_text(encoding="utf-8")
        self.assertIn("## Case-Level Results", report_text)
        self.assertIn("## Model-Level Descriptive Summary", report_text)
        self.assertIn("do not treat means as statistically robust", report_text)
        self.assertIn("case_a", report_text)
        self.assertIn(OPENAI_FINAL_POST_MODEL, report_text)

    def test_manifest_records_git_commit_for_reproducibility(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_manifest",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        manifest = json.loads(Path(result.artifacts.manifest_json).read_text(encoding="utf-8"))
        self.assertIsInstance(manifest["git_commit"], str)
        self.assertTrue(manifest["git_commit"])
        self.assertEqual(manifest["runs_per_plan"], 1)

    def test_writer_comparison_smoke_requests_keep_repair_disabled(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result())
        harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_no_repair",
                cases=(self.case,),
                plans=(_claude_writer_plan(), self.plan),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )

        for call in smoke_runner.call_args_list:
            smoke_request = call.args[0]
            self.assertEqual(smoke_request.mode, SMOKE_MODE_STANDALONE)
            self.assertIsNone(smoke_request.repair_provider)
            self.assertIsNone(smoke_request.repair_model)
            self.assertFalse(smoke_request.include_raw_responses)
            self.assertTrue(smoke_request.include_candidate_post_text)
            self.assertFalse(smoke_request.save_output)

    def test_publication_package_fields_are_not_treated_as_writer_output(self) -> None:
        smoke_runner = Mock(return_value=_smoke_result(post_text="Core candidate"))
        result = harness.run_linkedin_final_post_model_experiment(
            harness.FinalPostModelExperimentRequest(
                experiment_id="exp_core_only",
                cases=(self.case,),
                plans=(self.plan,),
                output_root=self.root / "outputs",
            ),
            smoke_runner=smoke_runner,
            now_factory=_fixed_now,
        )
        record = result.run_records[0]

        self.assertEqual(record["candidate_post_text"], "Core candidate")
        self.assertNotIn("hook_variants", record["candidate_post_text"])
        self.assertNotIn("quality_checks", record["candidate_post_text"])

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
    post_text: str = "Accepted final post",
    final_post_text: str | None = None,
    candidate_provider: str = "openai",
    candidate_model: str = OPENAI_FINAL_POST_MODEL,
    invocation_counts: dict[str, int] | None = None,
    safe_failure_code: str | None = None,
    safe_failure_message: str = "",
    deterministic_gate_passed: bool | None = None,
    quality_passed: bool | None = True,
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
                "scores": _quality_scores(),
                "total_score": 41,
                "failed_criteria": [],
                "criterion_rationales": {
                    criterion: {
                        "score": score,
                        "max_score": 5,
                        "rationale": f"{criterion} rationale",
                        "post_text_evidence": f"{criterion} evidence",
                        "failure_reason": "",
                    }
                    for criterion, score in _quality_scores().items()
                },
            },
            "semantic_grounding_review": {
                "pass": True,
                "claim_reviews": [
                    {
                        "claim_id": "c1",
                        "field_name": "post_text",
                        "value_index": None,
                        "claim_text": "Supported claim",
                        "claim_type": "author_interpretation",
                        "support_status": "supported",
                        "severity": "info",
                        "supported_evidence_ids": ["a0-summary"],
                        "required_qualifications": [],
                        "missing_qualifications": [],
                        "rationale": "grounded",
                        "repair_hint": "",
                    },
                    {
                        "claim_id": "c2",
                        "field_name": "post_text",
                        "value_index": None,
                        "claim_text": "Unsupported claim",
                        "claim_type": "causal_claim",
                        "support_status": "unsupported",
                        "severity": "major",
                        "supported_evidence_ids": [],
                        "required_qualifications": [],
                        "missing_qualifications": [],
                        "rationale": "not grounded enough",
                        "repair_hint": "qualify it",
                    },
                    {
                        "claim_id": "c3",
                        "field_name": "post_text",
                        "value_index": None,
                        "claim_text": "Contradicted claim",
                        "claim_type": "market_condition",
                        "support_status": "contradicted",
                        "severity": "minor",
                        "supported_evidence_ids": [],
                        "required_qualifications": [],
                        "missing_qualifications": [],
                        "rationale": "conflicts with evidence",
                        "repair_hint": "remove it",
                    },
                ],
                "blocking_claim_ids": [],
                "automatic_fail_reason": "",
                "requires_human_review": False,
                "repairable": False,
                "repair_instructions": [
                    {
                        "claim_id": "c2",
                        "instruction": "Remove the unsupported claim.",
                    }
                ],
            },
            "candidate_payload": {
                "post_text": post_text,
                "post_text_length": len(post_text),
            },
            "parsed_candidate_payload": {
                "post_text": post_text,
                "post_text_length": len(post_text),
            },
            "publication_package": {
                "post_text": post_text,
                "hook_variants": [],
                "cta_variants": [],
                "hashtags": [],
                "carousel_outline": [],
                "quality_checks": {
                    "linkedin_ready": True,
                    "uses_only_provided_facts": True,
                    "has_clear_point_of_view": True,
                },
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
            "candidate_writer": {
                "provider": candidate_provider,
                "model": candidate_model,
            },
            "semantic_grounding": {"provider": "openai", "model": OPENAI_FINAL_POST_MODEL},
            "quality_evaluator": {"provider": "openai", "model": OPENAI_FINAL_POST_MODEL},
        },
        invocation_budget={
            "candidate_writer": 1,
            "semantic_grounding": 1,
            "quality_evaluator": 1,
            "repair_writer": 0,
        },
        invocation_counts=invocation_counts
        or {
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
        final_post_text=final_post_text or post_text,
        safe_failure_code=safe_failure_code,
        safe_failure_message=safe_failure_message,
        deterministic_gate_passed=deterministic_gate_passed,
        quality_passed=quality_passed,
        saved_output_path=None,
        sanitized_result=sanitized_result,
    )





def _adaptation_failure_sanitized_result(parsed_text: str) -> dict:
    return {
        "failure_stage": STAGE_CANDIDATE_WRITER_ADAPTATION,
        "candidate_payload": None,
        "parsed_candidate_payload": {
            "post_text": parsed_text,
            "post_text_length": len(parsed_text),
        },
        "publication_package": None,
        "accepted_core_post": None,
        "final_attempt_outcome": None,
        "quality_review": None,
        "semantic_grounding_review": None,
        "provider_response_diagnostics": {},
        "stage_statuses": [
            {
                "stage": STAGE_CANDIDATE_WRITER_ADAPTATION,
                "status": "failed",
                "error_code": FAILURE_CANDIDATE_WRITER_ADAPTATION,
                "error_message": (
                    "parsed candidate does not satisfy FinalPostPayload structure."
                ),
                "metadata": {
                    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS: {
                        "schema_version": "1.0",
                        "failure_stage": STAGE_CANDIDATE_WRITER_ADAPTATION,
                        "parser_error_code": None,
                        "adapter_error_code": "payload_contract_violation",
                        "top_level_json_type": "dict",
                        "received_top_level_keys": ["post_text"],
                        "missing_required_fields": [],
                        "unexpected_fields": [],
                        "invalid_field_names": ["post_text"],
                        "field_violations": [
                            {
                                "field_name": "post_text",
                                "reason_code": "above_max_length",
                                "actual_length": len(parsed_text),
                                "maximum_allowed": FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
                            }
                        ],
                        "candidate_text_length": len(parsed_text),
                        "diagnostics_truncated": False,
                        "redacted_key_count": 0,
                    }
                },
            }
        ],
    }


def _claude_writer_plan() -> harness.FinalPostExperimentPlan:
    claude = harness.FinalPostExperimentRoleModel("anthropic", "claude-sonnet-5")
    gpt = harness.FinalPostExperimentRoleModel("openai", OPENAI_FINAL_POST_MODEL)
    return harness.FinalPostExperimentPlan(
        plan_id="claude_writer_gpt_fixed",
        mode=SMOKE_MODE_STANDALONE,
        candidate_writer=claude,
        semantic_grounding=gpt,
        quality_evaluator=gpt,
    )


def _quality_scores() -> dict[str, int]:
    return {
        "hook": 5,
        "controlling_angle": 5,
        "reader_problem": 5,
        "pattern_interrupt": 4,
        "evidence": 4,
        "author_point_of_view": 4,
        "human_voice": 5,
        "practical_value": 4,
        "cta": 5,
    }
