from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.conf import settings
from django.core.management import call_command
from django.test import SimpleTestCase
from django.test import override_settings

from apps.ai.client import THINKING_MODE_DISABLED, THINKING_MODE_PROVIDER_DEFAULT
from apps.packaging.management.commands import smoke_linkedin_final_post
from services.packaging import linkedin_post_final_post_smoke_runner
from services.packaging.linkedin_post_final_post_smoke_runner import (
    EXIT_CONFIG_ERROR,
    EXIT_EXECUTION_FAILURE,
    EXIT_OK,
    EXIT_SCENARIO_MISMATCH,
    SMOKE_MODE_CONTROLLED_REPAIR,
    SMOKE_MODE_STANDALONE,
    SMOKE_STATUS_COMPLETED,
    SMOKE_STATUS_CONFIG_ERROR,
    SMOKE_STATUS_DRY_RUN,
    SMOKE_STATUS_EXECUTION_FAILED,
    SMOKE_STATUS_SCENARIO_MISMATCH,
    FinalPostSmokeRunRequest,
    run_final_post_smoke,
)


FIXTURE_PATH = (
    Path(settings.BASE_DIR)
    / "tests"
    / "fixtures"
    / "linkedin_post_smoke"
    / "remote_work_policy.json"
)


class FinalPostSmokeRunnerTests(SimpleTestCase):
    def test_dry_run_does_not_invoke_standalone_api(self) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=FIXTURE_PATH)
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_DRY_RUN)
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertEqual(result.invocation_counts["candidate_writer"], 0)
        self.assertIn(
            "selected_evidence_ids",
            result.sanitized_result,
        )

    def test_dry_run_can_display_gemini_candidate_and_grounding_selections(
        self,
    ) -> None:
        result = run_final_post_smoke(
            FinalPostSmokeRunRequest(
                input_path=FIXTURE_PATH,
                candidate_provider="gemini",
                candidate_model="gemini-3.6-flash",
                semantic_grounding_provider="gemini",
                semantic_grounding_model="gemini-3.6-flash",
                quality_evaluator_provider="openai",
                quality_evaluator_model="gpt-4.1-2025-04-14",
            )
        )

        self.assertEqual(result.status, SMOKE_STATUS_DRY_RUN)
        self.assertEqual(
            result.invocation_counts,
            {
                "candidate_writer": 0,
                "semantic_grounding": 0,
                "quality_evaluator": 0,
                "repair_writer": 0,
            },
        )
        self.assertEqual(
            result.provider_models["candidate_writer"],
            {"provider": "gemini", "model": "gemini-3.6-flash"},
        )
        self.assertEqual(
            result.provider_models["semantic_grounding"],
            {"provider": "gemini", "model": "gemini-3.6-flash"},
        )
        self.assertEqual(
            result.provider_models["quality_evaluator"],
            {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
        )
        self.assertEqual(
            result.sanitized_result["thinking_modes"]["candidate_writer"],
            THINKING_MODE_PROVIDER_DEFAULT,
        )
        self.assertEqual(
            result.sanitized_result["thinking_modes"]["semantic_grounding"],
            THINKING_MODE_PROVIDER_DEFAULT,
        )

    def test_dry_run_exposes_anthropic_candidate_writer_disabled_thinking_mode(
        self,
    ) -> None:
        result = run_final_post_smoke(
            FinalPostSmokeRunRequest(
                input_path=FIXTURE_PATH,
                candidate_provider="anthropic",
                candidate_model="claude-sonnet-5",
                candidate_max_output_tokens=4000,
                semantic_grounding_provider="anthropic",
                semantic_grounding_model="claude-sonnet-5",
                quality_evaluator_provider="openai",
                quality_evaluator_model="gpt-4.1-2025-04-14",
            )
        )

        self.assertEqual(result.status, SMOKE_STATUS_DRY_RUN)
        self.assertEqual(
            result.sanitized_result["thinking_modes"]["candidate_writer"],
            THINKING_MODE_DISABLED,
        )
        self.assertEqual(
            result.sanitized_result["thinking_modes"]["semantic_grounding"],
            THINKING_MODE_PROVIDER_DEFAULT,
        )

    def test_dry_run_can_construct_manual_mixed_provider_matrix(self) -> None:
        cases = (
            ("openai", "gpt-4.1-2025-04-14", "openai", "gpt-4.1-2025-04-14"),
            ("gemini", "gemini-3.6-flash", "openai", "gpt-4.1-2025-04-14"),
            ("openai", "gpt-4.1-2025-04-14", "gemini", "gemini-3.6-flash"),
            ("gemini", "gemini-3.6-flash", "gemini", "gemini-3.6-flash"),
            ("anthropic", "claude-sonnet-5", "openai", "gpt-4.1-2025-04-14"),
            ("openai", "gpt-4.1-2025-04-14", "anthropic", "claude-sonnet-5"),
            ("anthropic", "claude-sonnet-5", "anthropic", "claude-sonnet-5"),
        )

        for (
            candidate_provider,
            candidate_model,
            grounding_provider,
            grounding_model,
        ) in cases:
            with self.subTest(
                candidate_provider=candidate_provider,
                grounding_provider=grounding_provider,
            ):
                result = run_final_post_smoke(
                    FinalPostSmokeRunRequest(
                        input_path=FIXTURE_PATH,
                        candidate_provider=candidate_provider,
                        candidate_model=candidate_model,
                        semantic_grounding_provider=grounding_provider,
                        semantic_grounding_model=grounding_model,
                        quality_evaluator_provider="openai",
                        quality_evaluator_model="gpt-4.1-2025-04-14",
                    )
                )

                self.assertEqual(result.status, SMOKE_STATUS_DRY_RUN)
                self.assertEqual(
                    result.invocation_counts,
                    {
                        "candidate_writer": 0,
                        "semantic_grounding": 0,
                        "quality_evaluator": 0,
                        "repair_writer": 0,
                    },
                )
                self.assertEqual(
                    result.provider_models["candidate_writer"]["provider"],
                    candidate_provider,
                )
                self.assertEqual(
                    result.provider_models["semantic_grounding"]["provider"],
                    grounding_provider,
                )
                self.assertEqual(
                    result.provider_models["quality_evaluator"],
                    {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
                )

    def test_provider_model_mismatch_returns_config_error_before_provider_execution(
        self,
    ) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    candidate_provider="gemini",
                    candidate_model="gpt-4.1-2025-04-14",
                )
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertIn(
            "unsupported PostFlow final post role/provider/model",
            result.safe_failure_message,
        )

    def test_unsupported_provider_returns_config_error_before_provider_execution(
        self,
    ) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    semantic_grounding_provider="unknown",
                    semantic_grounding_model="model",
                )
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(
            result.safe_failure_message,
            "unsupported PostFlow final post role/provider/model: "
            "role=semantic_grounding provider=unknown model=model",
        )

    def test_controlled_repair_keeps_repair_writer_openai_only_before_execution(
        self,
    ) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_controlled_repair_attempt",
        ) as controlled:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    mode=SMOKE_MODE_CONTROLLED_REPAIR,
                    repair_provider="gemini",
                    repair_model="gemini-3.6-flash",
                )
            )

        controlled.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(
            result.safe_failure_message,
            "unsupported PostFlow final post role/provider/model: "
            "role=repair_writer provider=gemini model=gemini-3.6-flash",
        )

    def test_quality_evaluator_gemini_returns_config_error_before_execution(
        self,
    ) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    quality_evaluator_provider="gemini",
                    quality_evaluator_model="gemini-3.6-flash",
                )
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(
            result.safe_failure_message,
            "unsupported PostFlow final post role/provider/model: "
            "role=quality_evaluator provider=gemini model=gemini-3.6-flash",
        )
        self.assertEqual(result.invocation_counts["quality_evaluator"], 0)

    def test_quality_evaluator_anthropic_returns_config_error_before_execution(
        self,
    ) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    quality_evaluator_provider="anthropic",
                    quality_evaluator_model="claude-sonnet-5",
                )
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(
            result.safe_failure_message,
            "unsupported PostFlow final post role/provider/model: "
            "role=quality_evaluator provider=anthropic model=claude-sonnet-5",
        )
        self.assertEqual(result.invocation_counts["quality_evaluator"], 0)

    @override_settings(OPENAI_API_KEY="sk-test", GEMINI_API_KEY="")
    def test_allow_api_requires_gemini_key_before_provider_execution(self) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    allow_api=True,
                    candidate_provider="gemini",
                    candidate_model="gemini-3.6-flash",
                )
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(result.exit_code, EXIT_CONFIG_ERROR)
        self.assertIn("GEMINI_API_KEY", result.safe_failure_message)
        self.assertNotIn("sk-test", json.dumps(result.to_dict(), sort_keys=True))

    @override_settings(OPENAI_API_KEY="sk-test", ANTHROPIC_API_KEY="")
    def test_allow_api_requires_anthropic_key_before_provider_execution(self) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    allow_api=True,
                    candidate_provider="anthropic",
                    candidate_model="claude-sonnet-5",
                )
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(result.exit_code, EXIT_CONFIG_ERROR)
        self.assertIn("ANTHROPIC_API_KEY", result.safe_failure_message)
        self.assertNotIn("sk-test", json.dumps(result.to_dict(), sort_keys=True))

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_standalone_mode_delegates_once_to_public_api(self) -> None:
        fake_result = _standalone_result()

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=FIXTURE_PATH, allow_api=True)
            )

        standalone.assert_called_once()
        _, kwargs = standalone.call_args
        self.assertEqual(
            kwargs["selected_evidence_ids"],
            ("ev-remote-policy", "ev-inclusion", "ev-isolation"),
        )
        self.assertEqual(result.status, SMOKE_STATUS_COMPLETED)
        self.assertEqual(result.final_post_text, "Accepted smoke post.")
        self.assertEqual(result.invocation_counts["candidate_writer"], 1)
        self.assertEqual(result.invocation_counts["semantic_grounding"], 1)
        self.assertEqual(result.invocation_counts["quality_evaluator"], 1)
        self.assertEqual(result.invocation_counts["repair_writer"], 0)
        self.assertEqual(
            result.sanitized_result["quality_review"]["criterion_rationales"],
            {"human_voice": {"rationale": "Human voice is specific."}},
        )
        self.assertEqual(
            result.sanitized_result["thinking_modes"],
            {
                "candidate_writer": "provider_default",
                "semantic_grounding": "provider_default",
            },
        )

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_semantic_grounding_summary_uses_blocking_claim_ids_only(self) -> None:
        fake_result = _standalone_result()
        fake_result.semantic_grounding_state.grounding_review.blocking_claim_ids = (
            "c1",
            "c2",
        )

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=FIXTURE_PATH, allow_api=True)
            )

        semantic_summary = result.sanitized_result["semantic_grounding_review"]

        self.assertEqual(semantic_summary["blocking_claim_ids"], ["c1", "c2"])
        self.assertNotIn("failed_claim_ids", semantic_summary)

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_controlled_repair_mode_delegates_once_to_public_api(self) -> None:
        fake_result = _controlled_result(repair_executed=True)

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_controlled_repair_attempt",
            return_value=fake_result,
        ) as controlled:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    mode=SMOKE_MODE_CONTROLLED_REPAIR,
                    allow_api=True,
                )
            )

        controlled.assert_called_once()
        _, kwargs = controlled.call_args
        self.assertEqual(
            kwargs["selected_evidence_ids"],
            ("ev-remote-policy", "ev-inclusion", "ev-isolation"),
        )
        self.assertEqual(result.status, SMOKE_STATUS_COMPLETED)
        self.assertTrue(result.repair_executed)
        self.assertEqual(result.invocation_counts["candidate_writer"], 1)
        self.assertEqual(result.invocation_counts["semantic_grounding"], 2)
        self.assertEqual(result.invocation_counts["quality_evaluator"], 2)
        self.assertEqual(result.invocation_counts["repair_writer"], 1)
        self.assertEqual(result.final_post_text, "Accepted repaired smoke post.")

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_repair_expected_but_not_executed_is_scenario_mismatch(self) -> None:
        fake_result = _controlled_result(repair_executed=False)

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_controlled_repair_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    mode=SMOKE_MODE_CONTROLLED_REPAIR,
                    allow_api=True,
                    expect_repair=True,
                )
            )

        self.assertEqual(result.status, SMOKE_STATUS_SCENARIO_MISMATCH)
        self.assertEqual(result.exit_code, EXIT_SCENARIO_MISMATCH)
        self.assertEqual(result.safe_failure_code, "expected_repair_not_executed")

    def test_fixture_secret_fields_are_rejected_before_provider_execution(self) -> None:
        secret_fixture = _write_temp_fixture({"OPENAI_API_KEY": "sk-secret"})

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=secret_fixture, allow_api=True)
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(result.exit_code, EXIT_CONFIG_ERROR)
        self.assertNotIn("sk-secret", result.safe_failure_message)

    def test_invalid_json_fixture_returns_config_error_before_provider_execution(self) -> None:
        invalid_fixture = _write_raw_temp_fixture("{not valid json")

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=invalid_fixture, allow_api=True)
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(result.exit_code, EXIT_CONFIG_ERROR)
        self.assertIn("not valid JSON", result.safe_failure_message)

    def test_directory_input_returns_config_error_before_provider_execution(self) -> None:
        directory_input = FIXTURE_PATH.parent

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=directory_input, allow_api=True)
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(result.exit_code, EXIT_CONFIG_ERROR)
        self.assertIn("not a file", result.safe_failure_message)
        self.assertNotIn("Traceback", result.safe_failure_message)
        self.assertNotIn("PermissionError", result.safe_failure_message)

        command_output = _StringOutput()
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as command_standalone:
            with self.assertRaises(SystemExit) as exc:
                call_command(
                    "smoke_linkedin_final_post",
                    "--input",
                    str(directory_input),
                    "--allow-api",
                    stdout=command_output,
                )

        command_standalone.assert_not_called()
        self.assertEqual(exc.exception.code, EXIT_CONFIG_ERROR)
        self.assertIn(SMOKE_STATUS_CONFIG_ERROR, command_output.getvalue())
        self.assertNotIn("Traceback", command_output.getvalue())
        self.assertNotIn("PermissionError", command_output.getvalue())

    def test_malformed_selected_evidence_returns_config_error_before_provider_execution(
        self,
    ) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        del payload["post_brief"]["evidence_to_use"][0]["evidence_text"]
        malformed_fixture = _write_raw_temp_fixture(json.dumps(payload))

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=malformed_fixture, allow_api=True)
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(result.exit_code, EXIT_CONFIG_ERROR)
        self.assertIn("evidence_text", result.safe_failure_message)

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_raw_responses_are_hidden_by_default_and_provider_raw_is_never_saved(
        self,
    ) -> None:
        fake_result = _standalone_result()

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=FIXTURE_PATH, allow_api=True)
            )
            debug_result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    allow_api=True,
                    include_raw_responses=True,
                )
            )

        default_serialized = json.dumps(result.to_dict(), sort_keys=True)
        debug_serialized = json.dumps(debug_result.to_dict(), sort_keys=True)
        self.assertNotIn("raw_texts", default_serialized)
        self.assertIn("candidate raw text", debug_serialized)
        self.assertNotIn("raw_provider_response", debug_serialized)
        self.assertNotIn("secret-provider-metadata", debug_serialized)

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_empty_candidate_response_includes_safe_provider_metadata_on_failure(
        self,
    ) -> None:
        fake_result = _standalone_result(
            failure_code="candidate_writer_empty_response",
            failure_message="empty provider response",
            stage_metadata={
                "provider_response_metadata": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "stop_reason": "max_tokens",
                    "content_block_types": ["thinking"],
                    "input_tokens": 1234,
                    "output_tokens": 4000,
                    "thinking_tokens": 4000,
                    "headers": {"authorization": "Bearer secret"},
                    "raw_prompt": "prompt secret",
                },
                "empty_text_classification": "MAX_TOKENS_BEFORE_TEXT",
            },
            accepted_payload=None,
        )

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    allow_api=True,
                    include_raw_responses=True,
                )
            )

        serialized = json.dumps(result.to_dict(), sort_keys=True)
        failed_status = result.sanitized_result["stage_statuses"][0]
        self.assertEqual(
            failed_status["metadata"]["provider_response_metadata"],
            {
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "stop_reason": "max_tokens",
                "content_block_types": ["thinking"],
                "input_tokens": 1234,
                "output_tokens": 4000,
                "thinking_tokens": 4000,
            },
        )
        self.assertEqual(
            failed_status["metadata"]["empty_text_classification"],
            "MAX_TOKENS_BEFORE_TEXT",
        )
        self.assertNotIn("raw_provider_response", serialized)
        self.assertNotIn("authorization", serialized)
        self.assertNotIn("Bearer secret", serialized)
        self.assertNotIn("prompt secret", serialized)

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_adaptation_failure_sanitized_output_includes_safe_validation_detail(
        self,
    ) -> None:
        fake_result = _standalone_result(
            failure_code="candidate_writer_adaptation_failure",
            failure_message="parsed candidate does not satisfy FinalPostPayload structure.",
            accepted_payload=None,
            failure_stage="candidate_writer_adaptation",
            stage_metadata={
                "validation_detail": {
                    "field_path": "FinalPostPayload.post_text",
                    "field_name": "post_text",
                    "message": (
                        "FinalPostPayload.post_text must not exceed 1300 characters."
                    ),
                    "max_chars": 1300,
                    "actual_chars": 1332,
                    "excess_chars": 32,
                    "raw_prompt": "prompt secret",
                    "api_key": "sk-secret",
                }
            },
        )

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=FIXTURE_PATH, allow_api=True)
            )

        stage_status = result.sanitized_result["stage_statuses"][0]
        self.assertEqual(
            stage_status["metadata"],
            {
                "validation_detail": {
                    "field_path": "FinalPostPayload.post_text",
                    "field_name": "post_text",
                    "message": (
                        "FinalPostPayload.post_text must not exceed 1300 characters."
                    ),
                    "max_chars": 1300,
                    "actual_chars": 1332,
                    "excess_chars": 32,
                }
            },
        )
        serialized = json.dumps(result.to_dict(), sort_keys=True)
        self.assertNotIn("prompt secret", serialized)
        self.assertNotIn("sk-secret", serialized)

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_unaccepted_candidate_text_is_hidden_from_default_sanitized_outputs(
        self,
    ) -> None:
        output_dir = Path(settings.BASE_DIR) / "debug_outputs"
        output_dir.mkdir(exist_ok=True)
        fake_result = _standalone_result(
            failure_code="quality_rejected",
            failure_message="quality evaluator rejected candidate",
            accepted_payload=None,
            candidate_post_text="Rejected candidate smoke post.",
        )

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    allow_api=True,
                    save_output=True,
                    output_dir=output_dir,
                )
            )

        serialized = json.dumps(result.to_dict(), sort_keys=True)
        saved = Path(result.saved_output_path).read_text(encoding="utf-8")
        self.assertEqual(result.final_post_text, "")
        self.assertNotIn("Rejected candidate smoke post.", serialized)
        self.assertNotIn("Rejected candidate smoke post.", saved)

        command_output = _StringOutput()
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            with self.assertRaises(SystemExit) as exc:
                call_command(
                    "smoke_linkedin_final_post",
                    "--input",
                    str(FIXTURE_PATH),
                    "--allow-api",
                    stdout=command_output,
                )

        self.assertEqual(exc.exception.code, EXIT_EXECUTION_FAILURE)
        self.assertNotIn("Rejected candidate smoke post.", command_output.getvalue())

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_provider_failure_uses_stable_safe_output(self) -> None:
        fake_result = _standalone_result(
            failure_code="candidate_writer_provider_failure",
            failure_message="provider failed with sk-secret-value",
            accepted_payload=None,
        )

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=FIXTURE_PATH, allow_api=True)
            )

        self.assertEqual(result.status, SMOKE_STATUS_EXECUTION_FAILED)
        self.assertEqual(result.exit_code, EXIT_EXECUTION_FAILURE)
        self.assertEqual(result.safe_failure_message, "redacted provider/configuration message")

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_unexpected_orchestration_exception_uses_safe_technical_failure(
        self,
    ) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            side_effect=RuntimeError("provider exploded with sk-secret-value"),
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=FIXTURE_PATH, allow_api=True)
            )

        self.assertEqual(result.status, SMOKE_STATUS_EXECUTION_FAILED)
        self.assertEqual(result.exit_code, EXIT_EXECUTION_FAILURE)
        self.assertEqual(result.safe_failure_code, "smoke_orchestration_failure")
        self.assertEqual(result.safe_failure_message, "smoke orchestration failed")
        self.assertNotIn(
            "sk-secret-value",
            json.dumps(result.to_dict(), sort_keys=True),
        )

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_save_output_writes_sanitized_json_under_requested_debug_directory(
        self,
    ) -> None:
        output_dir = Path(settings.BASE_DIR) / "debug_outputs"
        output_dir.mkdir(exist_ok=True)
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=_standalone_result(),
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=FIXTURE_PATH,
                    allow_api=True,
                    save_output=True,
                    output_dir=output_dir,
                )
            )

        self.assertIsNotNone(result.saved_output_path)
        output_path = Path(result.saved_output_path)
        self.assertTrue(output_path.exists())
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], SMOKE_STATUS_COMPLETED)
        self.assertEqual(saved["final_post_text"], "Accepted smoke post.")
        self.assertNotIn(
            "raw_provider_response",
            json.dumps(saved, sort_keys=True),
        )

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_input_fixture_is_not_mutated_and_output_is_json_safe(self) -> None:
        before = FIXTURE_PATH.read_text(encoding="utf-8")

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=_standalone_result(),
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=FIXTURE_PATH, allow_api=True)
            )

        self.assertEqual(FIXTURE_PATH.read_text(encoding="utf-8"), before)
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)

    def test_module_does_not_import_production_runtime_or_django_models(self) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_final_post_smoke_runner))
        imported_modules = _imported_modules(tree)
        imported_symbols = _imported_symbols(tree)

        forbidden_modules = {
            "apps.packaging.models",
            "django.db",
            "services.packaging.generator",
        }
        forbidden_symbols = {
            "ContentPackage",
            "generate_content_package_for_digest",
        }

        self.assertTrue(forbidden_modules.isdisjoint(imported_modules))
        self.assertTrue(forbidden_symbols.isdisjoint(imported_symbols))

    def test_command_dry_run_prints_budget_and_skips_api(self) -> None:
        output = _StringOutput()

        call_command(
            "smoke_linkedin_final_post",
            "--input",
            str(FIXTURE_PATH),
            stdout=output,
        )

        text = output.getvalue()
        self.assertIn("=== INVOCATION BUDGET ===", text)
        self.assertIn("semantic_grounding", text)
        self.assertIn(SMOKE_STATUS_DRY_RUN, text)
        self.assertIn("production_runtime_wiring: False", text)

    def test_command_defaults_use_expanded_grounding_and_quality_token_budgets(
        self,
    ) -> None:
        output = _StringOutput()

        with patch.object(
            smoke_linkedin_final_post,
            "run_final_post_smoke",
            return_value=linkedin_post_final_post_smoke_runner.FinalPostSmokeRunResult(
                status=SMOKE_STATUS_COMPLETED,
                exit_code=EXIT_OK,
                mode=SMOKE_MODE_STANDALONE,
                input_path=str(FIXTURE_PATH),
                provider_models={},
                invocation_budget={},
                invocation_counts={},
                dry_run=True,
                repair_enabled=False,
                repair_executed=False,
                initial_outcome=None,
                final_outcome=None,
                accepted=True,
                final_post_text="",
                deterministic_gate_passed=True,
                quality_passed=True,
                safe_failure_code=None,
                safe_failure_message="",
                saved_output_path=None,
                sanitized_result={},
            ),
        ) as smoke:
            call_command(
                "smoke_linkedin_final_post",
                "--input",
                str(FIXTURE_PATH),
                stdout=output,
            )

        smoke.assert_called_once()
        request = smoke.call_args.args[0]
        self.assertEqual(request.semantic_grounding_max_output_tokens, 2400)
        self.assertEqual(request.quality_evaluator_max_output_tokens, 2400)

    def test_command_accepts_grounding_provider_model_and_token_options(
        self,
    ) -> None:
        output = _StringOutput()

        with patch.object(
            smoke_linkedin_final_post,
            "run_final_post_smoke",
            return_value=linkedin_post_final_post_smoke_runner.FinalPostSmokeRunResult(
                status=SMOKE_STATUS_DRY_RUN,
                exit_code=EXIT_OK,
                mode=SMOKE_MODE_STANDALONE,
                input_path=str(FIXTURE_PATH),
                provider_models={},
                invocation_budget={},
                invocation_counts={},
                dry_run=True,
                repair_enabled=False,
                repair_executed=False,
                initial_outcome=None,
                final_outcome=None,
                accepted=False,
                final_post_text="",
                deterministic_gate_passed=None,
                quality_passed=None,
                safe_failure_code=None,
                safe_failure_message="",
                saved_output_path=None,
                sanitized_result={},
            ),
        ) as smoke:
            call_command(
                "smoke_linkedin_final_post",
                "--input",
                str(FIXTURE_PATH),
                "--grounding-provider",
                "openai",
                "--grounding-model",
                "grounding-model",
                "--grounding-max-output-tokens",
                "2600",
                stdout=output,
            )

        request = smoke.call_args.args[0]
        self.assertEqual(request.semantic_grounding_provider, "openai")
        self.assertEqual(request.semantic_grounding_model, "grounding-model")
        self.assertEqual(request.semantic_grounding_max_output_tokens, 2600)


def _standalone_result(
    *,
    failure_code: str | None = None,
    failure_message: str = "",
    accepted_payload: dict | None = None,
    candidate_post_text: str | None = None,
    failure_stage: str | None = None,
    stage_metadata: dict | None = None,
) -> SimpleNamespace:
    failed_stage = failure_stage or "candidate_writer_execution"
    payload = (
        {"post_text": "Accepted smoke post."}
        if accepted_payload is None and failure_code is None
        else accepted_payload
    )
    completed_stage = (
        "candidate_writer_parse"
        if failed_stage == "candidate_writer_adaptation"
        else "candidate_writer_execution"
    )
    return SimpleNamespace(
        completed_stage="attempt_outcome" if failure_code is None else completed_stage,
        failure_stage=None if failure_code is None else failed_stage,
        failure_code=failure_code,
        failure_message=failure_message,
        stage_statuses=[
            SimpleNamespace(
                stage="candidate_writer_execution" if failure_code is None else failed_stage,
                status="succeeded" if failure_code is None else "failed",
                error_code=failure_code,
                error_message=failure_message,
                metadata=copy.deepcopy(stage_metadata),
            )
        ],
        candidate_writer_invocation_count=1,
        semantic_grounding_invocation_count=0 if failure_code else 1,
        quality_evaluator_invocation_count=0 if failure_code else 1,
        repair_invocation_count=0,
        deterministic_gate_output=SimpleNamespace(
            validation_passed=failure_code is None,
            diagnostics=SimpleNamespace(
                system_linkedin_ready=failure_code is None,
                deterministic_checks_passed=failure_code is None,
            ),
        ),
        quality_evaluation_state=SimpleNamespace(
            quality_review={
                "pass": failure_code is None,
                "total_score": 37,
                "criterion_rationales": {
                    "human_voice": {"rationale": "Human voice is specific."}
                },
            }
        ),
        semantic_grounding_state=SimpleNamespace(
            grounding_review=SimpleNamespace(
                passed=failure_code is None,
                failed_claim_ids=(),
                blocking_claim_ids=(),
                automatic_fail_reason="",
                requires_human_review=False,
            )
        ),
        final_attempt_outcome=SimpleNamespace(
            outcome="accepted" if failure_code is None else "not_ready",
            reason=failure_message,
            repair_required=False,
            decision=SimpleNamespace(
                action="accept" if failure_code is None else "not_ready",
                reason=failure_message,
                repair_type=None,
                needs_human_review=False,
            ),
            accepted_result=(
                SimpleNamespace(accepted_payload=payload) if payload is not None else None
            ),
        ),
        candidate_writer_output=SimpleNamespace(
            payload=(
                _full_payload(candidate_post_text or payload["post_text"])
                if candidate_post_text or payload
                else None
            )
        ),
        candidate_writer_raw_response=SimpleNamespace(
            raw_text="candidate raw text",
            raw_provider_response={"secret": "secret-provider-metadata"},
        ),
        semantic_grounding_raw_response=SimpleNamespace(
            raw_text="semantic grounding raw text",
            raw_provider_response={"secret": "secret-provider-metadata"},
        ),
        quality_evaluator_raw_response=SimpleNamespace(
            raw_text="quality raw text",
            raw_provider_response={"secret": "secret-provider-metadata"},
        ),
        request=SimpleNamespace(
            candidate_writer_thinking_mode="provider_default",
            semantic_grounding_provider="openai",
            semantic_grounding_model="gpt-4.1-2025-04-14",
        ),
    )


def _controlled_result(*, repair_executed: bool) -> SimpleNamespace:
    accepted_payload = (
        {"post_text": "Accepted repaired smoke post."}
        if repair_executed
        else {"post_text": "Accepted smoke post."}
    )
    return SimpleNamespace(
        initial_attempt_result=_standalone_result(
            accepted_payload=None if repair_executed else accepted_payload
        ),
        repair_eligibility=SimpleNamespace(
            status="eligible" if repair_executed else "ineligible",
            eligible=repair_executed,
            reason=(
                "initial attempt eligible for one editorial repair"
                if repair_executed
                else "initial outcome is accepted"
            ),
        ),
        repair_executed=repair_executed,
        accepted_payload=accepted_payload,
        terminal_outcome="accepted",
        terminal_reason="deterministic and quality gates passed",
        failure_stage=None,
        failure_code=None,
        failure_message="",
        candidate_writer_invocation_count=1,
        semantic_grounding_invocation_count=2 if repair_executed else 1,
        quality_evaluator_invocation_count=2 if repair_executed else 1,
        repair_invocation_count=1 if repair_executed else 0,
        repaired_deterministic_gate_output=(
            SimpleNamespace(
                validation_passed=True,
                diagnostics=SimpleNamespace(
                    system_linkedin_ready=True,
                    deterministic_checks_passed=True,
                ),
            )
            if repair_executed
            else None
        ),
        repaired_quality_evaluation_state=(
            SimpleNamespace(quality_review={"pass": True, "total_score": 37})
            if repair_executed
            else None
        ),
        repaired_semantic_grounding_state=(
            SimpleNamespace(
                grounding_review=SimpleNamespace(
                    passed=True,
                    failed_claim_ids=(),
                    blocking_claim_ids=(),
                    automatic_fail_reason="",
                    requires_human_review=False,
                )
            )
            if repair_executed
            else None
        ),
        repaired_candidate_output=(
            SimpleNamespace(payload=_full_payload("Accepted repaired smoke post."))
            if repair_executed
            else None
        ),
        repair_writer_raw_response=SimpleNamespace(raw_text="repair raw text"),
        repaired_semantic_grounding_raw_response=(
            SimpleNamespace(raw_text="repaired semantic grounding raw text")
            if repair_executed
            else None
        ),
        repaired_quality_evaluator_raw_response=SimpleNamespace(
            raw_text="repaired quality raw text"
        ),
    )


def _full_payload(post_text: str) -> dict:
    return {
        "post_text": post_text,
        "hook_variants": ["Hook 1", "Hook 2", "Hook 3"],
        "cta_variants": ["CTA 1", "CTA 2", "CTA 3"],
        "hashtags": ["#work"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }


def _write_temp_fixture(extra: dict) -> Path:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload.update(extra)
    return _write_raw_temp_fixture(json.dumps(payload))


def _write_raw_temp_fixture(content: str) -> Path:
    path = Path(settings.BASE_DIR) / "debug_outputs" / "test_smoke_fixture.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class _StringOutput:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, text: str = "", *args, **kwargs) -> None:
        self.parts.append(str(text))

    def getvalue(self) -> str:
        return "\n".join(self.parts)


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _imported_symbols(tree: ast.AST) -> set[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            symbols.update(
                alias.asname or alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            symbols.update(alias.asname or alias.name for alias in node.names)
    return symbols
