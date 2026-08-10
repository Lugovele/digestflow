from __future__ import annotations

import ast
import copy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.conf import settings
from django.core.management import call_command
from django.test import SimpleTestCase
from django.test import override_settings

from apps.packaging.management.commands import smoke_linkedin_final_post
from services.packaging.linkedin_post_candidate_writer_execution import (
    EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT,
)
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FAILURE_CANDIDATE_WRITER_ADAPTATION,
    STAGE_CANDIDATE_WRITER_ADAPTATION,
)
from services.packaging import linkedin_post_final_post_smoke_runner
from services.packaging.linkedin_post_pipeline import (
    EvidenceRelationship,
    build_authorial_voice_directive_from_evidence_relationship,
)
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


BASE_FIXTURE_PATH = (
    Path(settings.BASE_DIR)
    / "tests"
    / "fixtures"
    / "linkedin_post_smoke"
    / "remote_work_policy.json"
)


class FinalPostSmokeRunnerTests(SimpleTestCase):
    def setUp(self) -> None:
        self.fixture_path = _write_current_contract_fixture()

    def test_dry_run_does_not_invoke_standalone_api(self) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=self.fixture_path)
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_DRY_RUN)
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertEqual(result.invocation_counts["candidate_writer"], 0)
        self.assertIn(
            "selected_evidence_ids",
            result.sanitized_result,
        )

    def test_dry_run_reports_role_diagnostics_for_mixed_candidate_roles(self) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=self.fixture_path,
                    candidate_provider="gemini",
                    candidate_model="gemini-3.6-flash",
                    semantic_grounding_provider="anthropic",
                    semantic_grounding_model="claude-sonnet-5",
                    quality_evaluator_provider="openai",
                    quality_evaluator_model="gpt-4.1-2025-04-14",
                )
            )

        standalone.assert_not_called()
        diagnostics = result.sanitized_result["role_diagnostics"]
        self.assertEqual(result.status, SMOKE_STATUS_DRY_RUN)
        self.assertEqual(
            [diagnostic["role"] for diagnostic in diagnostics],
            ["candidate_writer", "semantic_grounding", "quality_evaluator"],
        )
        self.assertEqual(diagnostics[0]["provider"], "gemini")
        self.assertEqual(diagnostics[0]["model"], "gemini-3.6-flash")
        self.assertEqual(diagnostics[1]["provider"], "anthropic")
        self.assertEqual(diagnostics[1]["model"], "claude-sonnet-5")
        self.assertEqual(diagnostics[2]["provider"], "openai")
        self.assertTrue(
            all(
                diagnostic["validation_status"] == "valid"
                for diagnostic in diagnostics
            )
        )

    def test_dry_run_rejects_invalid_role_model_before_provider_execution(self) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=self.fixture_path,
                    candidate_provider="gemini",
                    candidate_model="gpt-4.1-2025-04-14",
                )
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(result.exit_code, EXIT_CONFIG_ERROR)
        self.assertIn("Candidate Writer provider/model mismatch", result.safe_failure_message)

    @override_settings(OPENAI_API_KEY="sk-test", GEMINI_API_KEY="")
    def test_live_smoke_rejects_missing_selected_provider_key_before_delegation(
        self,
    ) -> None:
        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=self.fixture_path,
                    allow_api=True,
                    candidate_provider="gemini",
                    candidate_model="gemini-3.6-flash",
                )
            )

        standalone.assert_not_called()
        self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
        self.assertEqual(result.exit_code, EXIT_CONFIG_ERROR)
        self.assertIn("GEMINI_API_KEY", result.safe_failure_message)

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_standalone_mode_delegates_once_to_public_api(self) -> None:
        fake_result = _standalone_result()

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ) as standalone:
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
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
            result.sanitized_result["candidate_payload"],
            {"post_text_length": len("Accepted smoke post.")},
        )
        self.assertEqual(
            result.sanitized_result["accepted_core_post"],
            {"post_text": "Accepted smoke post."},
        )
        publication_package = result.sanitized_result["publication_package"]
        self.assertEqual(publication_package["post_text"], "Accepted smoke post.")
        self.assertEqual(publication_package["hook_variants"], [])
        self.assertEqual(publication_package["cta_variants"], [])
        self.assertEqual(publication_package["hashtags"], [])
        self.assertEqual(publication_package["carousel_outline"], [])
        self.assertEqual(
            publication_package["quality_checks"],
            {
                "linkedin_ready": True,
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
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
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
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
                    input_path=self.fixture_path,
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
        self.assertEqual(
            result.sanitized_result["repaired_candidate_payload"],
            {"post_text_length": len("Accepted repaired smoke post.")},
        )
        self.assertEqual(
            result.sanitized_result["accepted_core_post"],
            {"post_text": "Accepted repaired smoke post."},
        )
        self.assertEqual(result.sanitized_result["publication_package"]["hashtags"], [])
        self.assertEqual(
            result.sanitized_result["publication_package"]["carousel_outline"],
            [],
        )

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
                    input_path=self.fixture_path,
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
        directory_input = self.fixture_path.parent

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
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
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
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
            )
            debug_result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=self.fixture_path,
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
                    input_path=self.fixture_path,
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
                    str(self.fixture_path),
                    "--allow-api",
                    stdout=command_output,
                )

        self.assertEqual(exc.exception.code, EXIT_EXECUTION_FAILURE)
        self.assertNotIn("Rejected candidate smoke post.", command_output.getvalue())

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_candidate_post_text_is_hidden_by_default_and_exposed_by_flag(
        self,
    ) -> None:
        fake_result = _standalone_result(
            candidate_post_text="Canonical CandidatePost smoke text."
        )

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            default_result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
            )
            exposed_result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=self.fixture_path,
                    allow_api=True,
                    include_candidate_post_text=True,
                )
            )

        self.assertEqual(
            default_result.sanitized_result["candidate_payload"],
            {"post_text_length": len("Canonical CandidatePost smoke text.")},
        )
        self.assertEqual(
            exposed_result.sanitized_result["candidate_payload"],
            {
                "post_text": "Canonical CandidatePost smoke text.",
                "post_text_length": len("Canonical CandidatePost smoke text."),
            },
        )

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
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
            )

        self.assertEqual(result.status, SMOKE_STATUS_EXECUTION_FAILED)
        self.assertEqual(result.exit_code, EXIT_EXECUTION_FAILURE)
        self.assertEqual(result.safe_failure_message, "redacted provider/configuration message")

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_empty_candidate_response_exposes_only_safe_provider_diagnostics(
        self,
    ) -> None:
        provider_metadata = {
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "stop_reason": "max_tokens",
            "content_block_types": [
                "thinking",
                {"secret": "raw content block"},
            ],
            "input_tokens": 6616,
            "output_tokens": 4000,
            "thinking_tokens": 4000,
            "raw_provider_response": {"secret": "raw payload"},
            "headers": {"x-api-key": "sk-secret"},
            "prompt": "secret prompt text",
        }
        fake_result = _standalone_result(
            failure_code="candidate_writer_empty_response",
            failure_message="empty provider response",
            accepted_payload=None,
            provider_response_metadata=provider_metadata,
            empty_text_classification=(
                EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT
            ),
        )

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(
                    input_path=self.fixture_path,
                    allow_api=True,
                    include_raw_responses=True,
                )
            )

        diagnostics = result.sanitized_result["provider_response_diagnostics"]
        self.assertEqual(
            diagnostics,
            {
                "candidate_writer": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "stop_reason": "max_tokens",
                    "content_block_types": ["thinking"],
                    "input_tokens": 6616,
                    "output_tokens": 4000,
                    "thinking_tokens": 4000,
                    "empty_text_classification": (
                        EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT
                    ),
                }
            },
        )
        serialized = json.dumps(result.to_dict(), sort_keys=True)
        self.assertNotIn("raw_provider_response", serialized)
        self.assertNotIn("raw payload", serialized)
        self.assertNotIn("raw content block", serialized)
        self.assertNotIn("x-api-key", serialized)
        self.assertNotIn("sk-secret", serialized)
        self.assertNotIn("secret prompt text", serialized)

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
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
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
    def test_smoke_output_exposes_only_safe_adaptation_details(self) -> None:
        fake_result = _standalone_result(
            failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
            failure_message="parsed candidate does not satisfy FinalPostPayload structure.",
        )
        fake_result.stage_statuses = [
            SimpleNamespace(
                stage=STAGE_CANDIDATE_WRITER_ADAPTATION,
                status="failed",
                error_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
                error_message=(
                    "parsed candidate does not satisfy FinalPostPayload structure."
                ),
                metadata={
                    "adaptation_error_code": "invalid_final_post_payload",
                    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS: {
                        "schema_version": "1.0",
                        "failure_stage": STAGE_CANDIDATE_WRITER_ADAPTATION,
                        "parser_error_code": None,
                        "adapter_error_code": "invalid_field_values",
                        "top_level_json_type": "dict",
                        "received_top_level_keys": ["post_text"],
                        "missing_required_fields": ["hook_variants"],
                        "unexpected_fields": [],
                        "invalid_field_names": ["post_text"],
                        "candidate_text_length": None,
                        "diagnostics_truncated": False,
                        "redacted_key_count": 0,
                    },
                    "raw_response": "secret raw provider response",
                    "safe_details": {
                        "post_text": {
                            "input_length": (
                                FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1
                            ),
                            "max_chars": FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
                        },
                        "raw_text": "secret rejected candidate content",
                    },
                },
            )
        ]

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
            )

        stage_metadata = result.sanitized_result["stage_statuses"][0]["metadata"]
        self.assertEqual(
            stage_metadata["adaptation_error_code"],
            "invalid_final_post_payload",
        )
        self.assertEqual(
            stage_metadata["safe_details"]["post_text"],
            {
                "input_length": FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1,
                "max_chars": FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
            },
        )
        self.assertEqual(
            stage_metadata[METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS][
                "adapter_error_code"
            ],
            "invalid_field_values",
        )
        serialized = json.dumps(result.to_dict(), sort_keys=True)
        self.assertNotIn("secret rejected candidate content", serialized)
        self.assertNotIn("secret raw provider response", serialized)

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_smoke_output_resanitizes_structural_diagnostics_metadata(self) -> None:
        fake_result = _standalone_result(
            failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
            failure_message="parsed candidate does not satisfy FinalPostPayload structure.",
        )
        fake_result.stage_statuses = [
            SimpleNamespace(
                stage=STAGE_CANDIDATE_WRITER_ADAPTATION,
                status="failed",
                error_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
                error_message=(
                    "parsed candidate does not satisfy FinalPostPayload structure."
                ),
                metadata={
                    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS: {
                        "schema_version": "1.0",
                        "failure_stage": STAGE_CANDIDATE_WRITER_ADAPTATION,
                        "parser_error_code": None,
                        "adapter_error_code": "missing_required_fields",
                        "top_level_json_type": "prompt: secret raw response text",
                        "received_top_level_keys": [
                            "post_text",
                            "api_key",
                            "secret raw response text",
                        ],
                        "missing_required_fields": ["hook_variants"],
                        "unexpected_fields": ["provider_payload", "debug_extra"],
                        "invalid_field_names": ["post_text"],
                        "candidate_text_length": None,
                        "diagnostics_truncated": False,
                        "redacted_key_count": 0,
                    },
                },
            )
        ]

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=fake_result,
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
            )

        diagnostics = result.sanitized_result["stage_statuses"][0]["metadata"][
            METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS
        ]
        self.assertEqual(diagnostics["received_top_level_keys"], ["post_text"])
        self.assertEqual(diagnostics["unexpected_fields"], ["debug_extra"])
        self.assertIsNone(diagnostics["top_level_json_type"])
        self.assertTrue(diagnostics["diagnostics_truncated"])
        serialized = json.dumps(result.to_dict(), sort_keys=True)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("secret raw response text", serialized)
        self.assertNotIn("provider_payload", serialized)

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
                    input_path=self.fixture_path,
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
        before = self.fixture_path.read_text(encoding="utf-8")

        with patch.object(
            linkedin_post_final_post_smoke_runner,
            "execute_final_post_standalone_attempt",
            return_value=_standalone_result(),
        ):
            result = run_final_post_smoke(
                FinalPostSmokeRunRequest(input_path=self.fixture_path, allow_api=True)
            )

        self.assertEqual(self.fixture_path.read_text(encoding="utf-8"), before)
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
            str(self.fixture_path),
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
                input_path=str(self.fixture_path),
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
                str(self.fixture_path),
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
                input_path=str(self.fixture_path),
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
                str(self.fixture_path),
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
    provider_response_metadata: dict | None = None,
    empty_text_classification: str | None = None,
) -> SimpleNamespace:
    payload = (
        {"post_text": "Accepted smoke post."}
        if accepted_payload is None and failure_code is None
        else accepted_payload
    )
    return SimpleNamespace(
        completed_stage="attempt_outcome" if failure_code is None else "candidate_writer_execution",
        failure_stage=None if failure_code is None else "candidate_writer_execution",
        failure_code=failure_code,
        failure_message=failure_message,
        stage_statuses=[
            SimpleNamespace(
                stage="candidate_writer_execution",
                status="succeeded" if failure_code is None else "failed",
                error_code=failure_code,
                error_message=failure_message,
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
            raw_text=(
                ""
                if failure_code == "candidate_writer_empty_response"
                else "candidate raw text"
            ),
            raw_provider_response={"secret": "secret-provider-metadata"},
            provider_response_metadata=copy.deepcopy(provider_response_metadata),
            empty_text_classification=empty_text_classification,
        ),
        semantic_grounding_raw_response=SimpleNamespace(
            raw_text="semantic grounding raw text",
            raw_provider_response={"secret": "secret-provider-metadata"},
        ),
        quality_evaluator_raw_response=SimpleNamespace(
            raw_text="quality raw text",
            raw_provider_response={"secret": "secret-provider-metadata"},
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
    payload = _current_contract_payload()
    payload.update(extra)
    return _write_raw_temp_fixture(json.dumps(payload))


def _write_current_contract_fixture() -> Path:
    return _write_raw_temp_fixture(json.dumps(_current_contract_payload()))


def _current_contract_payload() -> dict:
    payload = json.loads(BASE_FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["angle_decision"]["authorial_voice_directive"] = asdict(
        build_authorial_voice_directive_from_evidence_relationship(
            _evidence_relationship_from_fixture_payload(payload)
        )
    )
    return payload


def _evidence_relationship_from_fixture_payload(payload: dict) -> EvidenceRelationship:
    angle_decision = payload["angle_decision"]
    selected_evidence_ids = [
        item["evidence_id"] for item in payload["post_brief"]["evidence_to_use"]
    ]
    return EvidenceRelationship(
        relationship_type="contrast",
        left_label="operational clarity",
        right_label="human support",
        left_evidence_ids=selected_evidence_ids[:1],
        right_evidence_ids=selected_evidence_ids[1:],
        supporting_evidence_ids=selected_evidence_ids,
        qualifier="selected smoke fixture evidence",
        thesis=angle_decision["controlling_angle"],
        reader_problem=angle_decision["reader_problem"],
        author_position=angle_decision["author_position"],
        main_tension=angle_decision["main_tension"],
        score=5,
        priority=1,
    )


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
