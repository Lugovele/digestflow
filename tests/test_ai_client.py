from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from django.test import override_settings

from apps.ai.client import (
    AI_PROVIDER_GEMINI,
    GEMINI_OPENAI_COMPATIBLE_BASE_URL,
    OpenAIClient,
    build_ai_client,
    get_ai_client_configuration_error,
    get_ai_provider_config,
    get_ai_provider_model_error,
    _extract_usage,
    estimate_cost_usd,
)


class EstimateCostUsdTests(SimpleTestCase):
    def test_estimated_cost_is_calculated_from_prompt_and_completion_tokens(self):
        cost = estimate_cost_usd(prompt_tokens=1000, completion_tokens=1000)

        self.assertEqual(cost, 0.02)

    def test_estimated_cost_is_none_when_tokens_are_missing(self):
        self.assertIsNone(estimate_cost_usd(prompt_tokens=None, completion_tokens=1000))
        self.assertIsNone(estimate_cost_usd(prompt_tokens=1000, completion_tokens=None))


class ExtractUsageTests(SimpleTestCase):
    def test_total_tokens_is_taken_from_response_usage_when_available(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=120, output_tokens=30, total_tokens=150)
        )

        usage = _extract_usage(response, {"usage": {}})

        self.assertEqual(
            usage,
            {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            },
        )

    def test_total_tokens_is_computed_when_missing_but_prompt_and_completion_exist(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=80, output_tokens=20, total_tokens=None)
        )

        usage = _extract_usage(response, {"usage": {}})

        self.assertEqual(usage["prompt_tokens"], 80)
        self.assertEqual(usage["completion_tokens"], 20)
        self.assertEqual(usage["total_tokens"], 100)

    def test_usage_falls_back_to_raw_usage_keys(self):
        response = SimpleNamespace(usage=None)
        raw = {
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
            }
        }

        usage = _extract_usage(response, raw)

        self.assertEqual(
            usage,
            {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
            },
        )


class OpenAIClientGenerateTextTests(SimpleTestCase):
    @override_settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="test-model",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_json_mode_failure_falls_back_for_legacy_callers_by_default(
        self,
        mock_openai,
    ):
        calls = []

        def create_response(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("json mode unavailable")
            return SimpleNamespace(
                output_text="fallback text",
                model_dump=lambda: {"id": "fallback"},
                usage=SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3),
            )

        mock_openai.return_value.responses.create.side_effect = create_response

        response = OpenAIClient().generate_text("Prompt", json_mode=True)

        self.assertEqual(response.text, "fallback text")
        self.assertEqual(len(calls), 2)
        self.assertIn("text", calls[0])
        self.assertNotIn("text", calls[1])

    @override_settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="test-model",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_json_mode_fallback_can_be_disabled(self, mock_openai):
        mock_openai.return_value.responses.create.side_effect = RuntimeError(
            "json mode unavailable"
        )

        with self.assertRaisesRegex(RuntimeError, "json mode unavailable"):
            OpenAIClient().generate_text(
                "Prompt",
                json_mode=True,
                allow_json_mode_fallback=False,
            )

        mock_openai.return_value.responses.create.assert_called_once()


class AIProviderConfigTests(SimpleTestCase):
    @override_settings(GEMINI_API_KEY="gemini-test-key")
    def test_gemini_provider_resolves_api_key_and_base_url(self):
        config = get_ai_provider_config("gemini")

        self.assertEqual(config.provider, AI_PROVIDER_GEMINI)
        self.assertEqual(config.api_key, "gemini-test-key")
        self.assertEqual(config.base_url, GEMINI_OPENAI_COMPATIBLE_BASE_URL)

    @override_settings(
        GEMINI_API_KEY="gemini-test-key",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_build_ai_client_selects_gemini_base_url_and_model(self, mock_openai):
        client = build_ai_client("gemini", "gemini-3.6-flash")

        self.assertEqual(client.provider, "gemini")
        self.assertEqual(client.model, "gemini-3.6-flash")
        mock_openai.assert_called_once_with(
            api_key="gemini-test-key",
            timeout=30,
            base_url=GEMINI_OPENAI_COMPATIBLE_BASE_URL,
        )

    @override_settings(
        GEMINI_API_KEY="gemini-test-key",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_gemini_generation_uses_chat_completions_not_responses(self, mock_openai):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Gemini text"))],
            model_dump=lambda: {"id": "gemini-chat"},
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )
        mock_openai.return_value.chat.completions.create.return_value = response
        client = build_ai_client("gemini", "gemini-3.6-flash")

        result = client.generate_text("Prompt", max_output_tokens=300, json_mode=False)

        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            model="gemini-3.6-flash",
            messages=[{"role": "user", "content": "Prompt"}],
            max_tokens=300,
        )
        mock_openai.return_value.responses.create.assert_not_called()
        self.assertEqual(result.text, "Gemini text")
        self.assertEqual(result.raw, {"id": "gemini-chat"})

    @override_settings(
        GEMINI_API_KEY="gemini-test-key",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_gemini_json_mode_uses_chat_response_format_once(self, mock_openai):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"pass": true}'))],
            model_dump=lambda: {"id": "gemini-json"},
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )
        mock_openai.return_value.chat.completions.create.return_value = response
        client = build_ai_client("gemini", "gemini-3.6-flash")

        result = client.generate_text(
            "Prompt",
            max_output_tokens=300,
            json_mode=True,
            allow_json_mode_fallback=False,
        )

        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            model="gemini-3.6-flash",
            messages=[{"role": "user", "content": "Prompt"}],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        mock_openai.return_value.responses.create.assert_not_called()
        self.assertEqual(result.text, '{"pass": true}')

    @override_settings(OPENAI_API_KEY="openai-test-key", OPENAI_TIMEOUT_SECONDS=30)
    @patch("apps.ai.client.OpenAI")
    def test_build_ai_client_selects_openai_without_base_url(self, mock_openai):
        client = build_ai_client("openai", "gpt-4.1-2025-04-14")

        self.assertEqual(client.provider, "openai")
        self.assertEqual(client.model, "gpt-4.1-2025-04-14")
        mock_openai.assert_called_once_with(
            api_key="openai-test-key",
            timeout=30,
        )

    @override_settings(GEMINI_API_KEY="gemini-test-key", OPENAI_TIMEOUT_SECONDS=30)
    @patch("apps.ai.client.OpenAI")
    def test_build_ai_client_strips_model_before_construction(self, mock_openai):
        client = build_ai_client("gemini", " gemini-3.6-flash ")

        self.assertEqual(client.model, "gemini-3.6-flash")

    @override_settings(GEMINI_API_KEY="")
    def test_missing_gemini_key_returns_sanitized_configuration_error(self):
        error = get_ai_client_configuration_error(
            "gemini",
            "gemini-3.6-flash",
            stage_name="AI",
        )

        self.assertEqual(
            error,
            "GEMINI_API_KEY must be configured with a real key before "
            "AI provider execution",
        )
        self.assertNotIn("gemini-test-key", error)

    def test_unsupported_provider_is_rejected_before_client_construction(self):
        self.assertEqual(
            get_ai_provider_model_error(
                "unknown",
                "model",
                stage_name="AI",
            ),
            "unsupported AI provider: unknown",
        )

    def test_gemini_provider_rejects_openai_model_before_invocation(self):
        error = get_ai_provider_model_error(
            "gemini",
            "gpt-4.1-2025-04-14",
            stage_name="AI",
        )

        self.assertIn("AI provider/model mismatch", error)
        self.assertIn("gemini-3.6-flash", error)

    def test_gemini_provider_rejects_obsolete_gemini_models_before_invocation(self):
        for model in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview"):
            with self.subTest(model=model):
                error = get_ai_provider_model_error(
                    "gemini",
                    model,
                    stage_name="AI",
                )

                self.assertIn("AI provider/model mismatch", error)
                self.assertIn("gemini-3.6-flash", error)

    def test_openai_provider_rejects_gemini_model_before_invocation(self):
        self.assertEqual(
            get_ai_provider_model_error(
                "openai",
                "gemini-3.6-flash",
                stage_name="AI",
            ),
            "AI provider/model mismatch: provider openai cannot use a Gemini model",
        )
