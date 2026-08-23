from unittest.mock import patch

from django.test import SimpleTestCase

from config.settings import (
    _postflow_model_setting,
    _postflow_provider_setting,
    _postflow_role_model_setting,
    _postflow_role_provider_setting,
)


class PostFlowModelSettingsTests(SimpleTestCase):
    def test_stage_model_setting_prefers_explicit_postflow_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_MODEL": "legacy-shared-model",
                "POSTFLOW_RESEARCH_MODEL": "stage-specific-model",
            },
        ):
            self.assertEqual(
                _postflow_model_setting(
                    "POSTFLOW_RESEARCH_MODEL",
                    "default-stage-model",
                ),
                "stage-specific-model",
            )

    def test_stage_model_setting_falls_back_to_legacy_openai_model(self) -> None:
        with patch.dict(
            "os.environ",
            {"OPENAI_MODEL": "legacy-shared-model"},
            clear=True,
        ):
            self.assertEqual(
                _postflow_model_setting(
                    "POSTFLOW_RESEARCH_MODEL",
                    "default-stage-model",
                ),
                "legacy-shared-model",
            )

    def test_stage_model_setting_uses_stage_default_without_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                _postflow_model_setting(
                    "POSTFLOW_POST_MODEL",
                    "gpt-4.1-2025-04-14",
                ),
                "gpt-4.1-2025-04-14",
            )

    def test_stage_provider_setting_defaults_to_openai(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                _postflow_provider_setting("POSTFLOW_POST_PROVIDER"),
                "openai",
            )

    def test_stage_provider_setting_normalizes_configured_provider(self) -> None:
        with patch.dict("os.environ", {"POSTFLOW_POST_PROVIDER": " OpenAI "}):
            self.assertEqual(
                _postflow_provider_setting("POSTFLOW_POST_PROVIDER"),
                "openai",
            )

    def test_candidate_writer_role_settings_default_to_frozen_claude(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                _postflow_role_provider_setting(
                    "POSTFLOW_CANDIDATE_WRITER_PROVIDER",
                    "anthropic",
                ),
                "anthropic",
            )
            self.assertEqual(
                _postflow_role_model_setting(
                    "POSTFLOW_CANDIDATE_WRITER_MODEL",
                    "claude-sonnet-5",
                ),
                "claude-sonnet-5",
            )

    def test_quality_evaluator_role_settings_default_to_frozen_gpt41(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                _postflow_role_provider_setting(
                    "POSTFLOW_QUALITY_EVALUATOR_PROVIDER",
                    "openai",
                ),
                "openai",
            )
            self.assertEqual(
                _postflow_role_model_setting(
                    "POSTFLOW_QUALITY_EVALUATOR_MODEL",
                    "gpt-4.1-2025-04-14",
                ),
                "gpt-4.1-2025-04-14",
            )

    def test_role_specific_settings_ignore_legacy_shared_post_defaults(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POSTFLOW_POST_PROVIDER": "openai",
                "POSTFLOW_POST_MODEL": "legacy-shared-post-model",
            },
            clear=True,
        ):
            self.assertEqual(
                _postflow_role_provider_setting(
                    "POSTFLOW_CANDIDATE_WRITER_PROVIDER",
                    "anthropic",
                ),
                "anthropic",
            )
            self.assertEqual(
                _postflow_role_model_setting(
                    "POSTFLOW_CANDIDATE_WRITER_MODEL",
                    "claude-sonnet-5",
                ),
                "claude-sonnet-5",
            )
            self.assertEqual(
                _postflow_role_provider_setting(
                    "POSTFLOW_QUALITY_EVALUATOR_PROVIDER",
                    "openai",
                ),
                "openai",
            )
            self.assertEqual(
                _postflow_role_model_setting(
                    "POSTFLOW_QUALITY_EVALUATOR_MODEL",
                    "gpt-4.1-2025-04-14",
                ),
                "gpt-4.1-2025-04-14",
            )

    def test_role_specific_settings_can_be_configured_independently(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POSTFLOW_CANDIDATE_WRITER_PROVIDER": " Anthropic ",
                "POSTFLOW_CANDIDATE_WRITER_MODEL": "claude-sonnet-5",
                "POSTFLOW_QUALITY_EVALUATOR_PROVIDER": " OpenAI ",
                "POSTFLOW_QUALITY_EVALUATOR_MODEL": "gpt-4.1-2025-04-14",
            },
            clear=True,
        ):
            self.assertEqual(
                _postflow_role_provider_setting(
                    "POSTFLOW_CANDIDATE_WRITER_PROVIDER",
                    "anthropic",
                ),
                "anthropic",
            )
            self.assertEqual(
                _postflow_role_model_setting(
                    "POSTFLOW_CANDIDATE_WRITER_MODEL",
                    "unused-default",
                ),
                "claude-sonnet-5",
            )
            self.assertEqual(
                _postflow_role_provider_setting(
                    "POSTFLOW_QUALITY_EVALUATOR_PROVIDER",
                    "openai",
                ),
                "openai",
            )
            self.assertEqual(
                _postflow_role_model_setting(
                    "POSTFLOW_QUALITY_EVALUATOR_MODEL",
                    "unused-default",
                ),
                "gpt-4.1-2025-04-14",
            )
