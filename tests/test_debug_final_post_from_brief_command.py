from io import StringIO
import inspect
import importlib
import json
import sys
from types import ModuleType
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase, override_settings

from apps.packaging.management.commands import debug_final_post_from_brief
from apps.packaging.management.commands.debug_final_post_from_brief import (
    Command,
    _parse_json_response,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "linkedin_post_cases"
    / "topic_214_digest_128.json"
)


class DebugFinalPostFromBriefCommandTests(SimpleTestCase):
    def test_services_packaging_import_does_not_eagerly_import_generator(self) -> None:
        sys.modules.pop("services.packaging.generator", None)

        importlib.import_module("services.packaging")

        self.assertNotIn("services.packaging.generator", sys.modules)

    def test_services_packaging_lazy_wrapper_resolves_generator_function(self) -> None:
        import services.packaging

        fake_generator = ModuleType("services.packaging.generator")

        def fake_generate_content_package_for_digest(*args, **kwargs):
            return {
                "args": args,
                "kwargs": kwargs,
            }

        fake_generator.generate_content_package_for_digest = fake_generate_content_package_for_digest

        with patch.dict(sys.modules, {"services.packaging.generator": fake_generator}):
            result = services.packaging.generate_content_package_for_digest(
                "digest",
                author_profile={"role": "debug"},
            )

        self.assertEqual(result["args"], ("digest",))
        self.assertEqual(result["kwargs"], {"author_profile": {"role": "debug"}})

    def test_command_module_does_not_import_packaging_generator(self) -> None:
        source = inspect.getsource(debug_final_post_from_brief)

        self.assertNotIn("services.packaging.generator", source)

    def test_command_disables_system_checks_for_lightweight_dry_run(self) -> None:
        self.assertEqual(Command.requires_system_checks, [])


    @override_settings(POSTFLOW_POST_MODEL="post-model-debug")
    @patch("apps.packaging.management.commands.debug_final_post_from_brief.OpenAIClient")
    def test_dry_run_does_not_call_openai_client(self, mock_openai_client) -> None:
        output = StringIO()

        call_command(
            "debug_final_post_from_brief",
            "--fixture",
            str(FIXTURE_PATH),
            stdout=output,
        )

        mock_openai_client.assert_not_called()
        self.assertIn("API call skipped", output.getvalue())

    @override_settings(POSTFLOW_POST_MODEL="post-model-debug")
    @patch("apps.packaging.management.commands.debug_final_post_from_brief.OpenAIClient")
    def test_dry_run_reports_model_and_selected_evidence_ids(self, mock_openai_client) -> None:
        output = StringIO()

        call_command(
            "debug_final_post_from_brief",
            "--fixture",
            str(FIXTURE_PATH),
            stdout=output,
        )

        command_output = output.getvalue()
        mock_openai_client.assert_not_called()
        self.assertIn("model: post-model-debug", command_output)
        self.assertIn("selected_evidence_ids", command_output)
        self.assertIn("a0-summary", command_output)

    @override_settings(
        OPENAI_API_KEY="sk-test",
        POSTFLOW_POST_MODEL="post-model-debug",
        POSTFLOW_POST_PROVIDER="openai",
    )
    @patch("apps.packaging.management.commands.debug_final_post_from_brief.OpenAIClient")
    def test_api_output_path_prints_deterministic_diagnostics(self, mock_openai_client) -> None:
        output = StringIO()
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(_model_payload_with_diagnostics_failures())
        )

        call_command(
            "debug_final_post_from_brief",
            "--fixture",
            str(FIXTURE_PATH),
            "--allow-api",
            stdout=output,
        )

        command_output = output.getvalue()
        self.assertIn("=== DETERMINISTIC DIAGNOSTICS ===", command_output)
        self.assertIn("system_linkedin_ready: false", command_output)
        self.assertIn("deterministic_checks_passed: false", command_output)
        self.assertIn("missing_quality_check_keys", command_output)
        self.assertIn("evidence_id_leaks", command_output)
        self.assertIn("scaffold_phrase_leaks", command_output)

    @override_settings(
        OPENAI_API_KEY="sk-test",
        POSTFLOW_POST_MODEL="post-model-debug",
        POSTFLOW_POST_PROVIDER="openai",
    )
    @patch("apps.packaging.management.commands.debug_final_post_from_brief.OpenAIClient")
    @patch("apps.packaging.management.commands.debug_final_post_from_brief.diagnose_final_post_payload")
    def test_diagnostics_receive_selected_evidence_ids(
        self,
        mock_diagnose_final_post_payload,
        mock_openai_client,
    ) -> None:
        output = StringIO()
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(_valid_model_payload())
        )
        mock_diagnose_final_post_payload.return_value = SimpleNamespace(
            system_linkedin_ready=True,
            deterministic_checks_passed=True,
            model_claimed_linkedin_ready=True,
            missing_quality_check_keys=[],
            non_boolean_quality_check_keys=[],
            evidence_id_leaks=[],
            scaffold_phrase_leaks=[],
            source_summary_phrase_leaks=[],
            repair_reasons=[],
            to_dict=lambda: {"system_linkedin_ready": True},
        )

        call_command(
            "debug_final_post_from_brief",
            "--fixture",
            str(FIXTURE_PATH),
            "--allow-api",
            stdout=output,
        )

        self.assertEqual(
            mock_diagnose_final_post_payload.call_args.kwargs["selected_evidence_ids"],
            ["a0-summary", "a1-kp0", "a2-summary"],
        )

    @override_settings(
        OPENAI_API_KEY="sk-test",
        POSTFLOW_POST_MODEL="post-model-debug",
        POSTFLOW_POST_PROVIDER="openai",
    )
    @patch("apps.packaging.management.commands.debug_final_post_from_brief._save_debug_output")
    @patch("apps.packaging.management.commands.debug_final_post_from_brief.OpenAIClient")
    def test_save_output_includes_diagnostics(self, mock_openai_client, mock_save_debug_output) -> None:
        output = StringIO()
        mock_save_debug_output.return_value = Path("debug_outputs/final_post_model_runs/test.json")
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(_model_payload_with_diagnostics_failures())
        )

        call_command(
            "debug_final_post_from_brief",
            "--fixture",
            str(FIXTURE_PATH),
            "--allow-api",
            "--save-output",
            stdout=output,
        )

        saved_payload = mock_save_debug_output.call_args.args[0]
        self.assertIn("diagnostics", saved_payload)
        self.assertFalse(saved_payload["diagnostics"]["system_linkedin_ready"])
        self.assertIn("missing_quality_checks", saved_payload["diagnostics"]["repair_reasons"])

    def test_local_json_parser_extracts_fenced_json_object(self) -> None:
        payload = _parse_json_response(
            '```json\n{"post_text": "Done", "hashtags": ["#AI"]}\n```'
        )

        self.assertEqual(payload["post_text"], "Done")
        self.assertEqual(payload["hashtags"], ["#AI"])


def _valid_model_payload() -> dict:
    return {
        "post_text": "A clear final post about how work is changing.",
        "hook_variants": ["Hook one", "Hook two", "Hook three"],
        "cta_variants": ["CTA one", "CTA two", "CTA three"],
        "hashtags": ["#FutureOfWork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }


def _model_payload_with_diagnostics_failures() -> dict:
    payload = _valid_model_payload()
    payload.update(
        {
            "post_text": "This selected evidence cites a0-summary directly.",
            "hook_variants": ["separate signals", "Hook two", "Hook three"],
            "quality_checks": {"linkedin_ready": True},
        }
    )
    return payload
