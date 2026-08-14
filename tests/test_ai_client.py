import json
import traceback
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from django.test import SimpleTestCase
from django.test import override_settings

from apps.ai.client import (
    AI_PROVIDER_ANTHROPIC,
    AI_PROVIDER_GEMINI,
    AI_THINKING_MODE_DISABLED,
    AI_THINKING_MODE_PROVIDER_DEFAULT,
    ANTHROPIC_API_VERSION,
    ANTHROPIC_MESSAGES_ENDPOINT,
    GEMINI_OPENAI_COMPATIBLE_BASE_URL,
    OpenAIClient,
    build_ai_client,
    get_ai_client_configuration_error,
    get_ai_provider_config,
    get_ai_provider_model_error,
    get_ai_provider_thinking_mode_error,
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

    @override_settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4.1-2025-04-14",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_openai_responses_metadata_records_normal_completion(self, mock_openai):
        mock_openai.return_value.responses.create.return_value = SimpleNamespace(
            output_text="completed text",
            model_dump=lambda: {
                "id": "resp_completed",
                "model": "gpt-4.1-2025-04-14",
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
            usage=SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7),
        )

        response = OpenAIClient().generate_text("Prompt", max_output_tokens=500)

        self.assertEqual(response.text, "completed text")
        self.assertEqual(
            response.provider_response_metadata,
            {
                "provider": "openai",
                "model": "gpt-4.1-2025-04-14",
                "provider_finish_reason": "completed",
                "provider_stop_reason": None,
                "provider_max_output_tokens": 500,
                "provider_reported_output_tokens": 4,
                "provider_output_limit_reached": False,
                "provider_prompt_tokens": 3,
                "provider_visible_output_tokens": 4,
                "provider_total_tokens": None,
                "provider_hidden_output_tokens": None,
                "provider_combined_output_tokens": None,
                "provider_output_budget_utilization_percent": None,
                "provider_reasoning_tokens": None,
                "provider_thinking_tokens": None,
            },
        )

    @override_settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4.1-2025-04-14",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_openai_responses_metadata_detects_max_output_tokens(self, mock_openai):
        mock_openai.return_value.responses.create.return_value = SimpleNamespace(
            output_text='{"post_text":"partial"}',
            model_dump=lambda: {
                "id": "resp_incomplete",
                "model": "gpt-4.1-2025-04-14",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 8, "output_tokens": 500},
            },
            usage=SimpleNamespace(input_tokens=8, output_tokens=500, total_tokens=508),
        )

        response = OpenAIClient().generate_text("Prompt", max_output_tokens=500)

        self.assertEqual(
            response.provider_response_metadata,
            {
                "provider": "openai",
                "model": "gpt-4.1-2025-04-14",
                "provider_finish_reason": "max_output_tokens",
                "provider_stop_reason": None,
                "provider_max_output_tokens": 500,
                "provider_reported_output_tokens": 500,
                "provider_output_limit_reached": True,
                "provider_prompt_tokens": 8,
                "provider_visible_output_tokens": 500,
                "provider_total_tokens": None,
                "provider_hidden_output_tokens": None,
                "provider_combined_output_tokens": None,
                "provider_output_budget_utilization_percent": None,
                "provider_reasoning_tokens": None,
                "provider_thinking_tokens": None,
            },
        )


class AIProviderConfigTests(SimpleTestCase):
    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    def test_anthropic_provider_resolves_api_key(self):
        config = get_ai_provider_config("anthropic")

        self.assertEqual(config.provider, AI_PROVIDER_ANTHROPIC)
        self.assertEqual(config.api_key, "anthropic-test-key")
        self.assertIsNone(config.base_url)

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
            model_dump=lambda: {
                "id": "gemini-json",
                "model": "gemini-3.6-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"pass": true}'},
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
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
        self.assertEqual(
            result.provider_response_metadata,
            {
                "provider": "gemini",
                "model": "gemini-3.6-flash",
                "choices_count": 1,
                "finish_reasons": ["stop"],
                "provider_finish_reason": "stop",
                "provider_stop_reason": None,
                "provider_max_output_tokens": 300,
                "provider_reported_output_tokens": 3,
                "provider_output_limit_reached": False,
                "provider_prompt_tokens": 2,
                "provider_visible_output_tokens": 3,
                "provider_total_tokens": 5,
                "provider_hidden_output_tokens": 0,
                "provider_combined_output_tokens": 3,
                "provider_output_budget_utilization_percent": 1.0,
                "provider_reasoning_tokens": None,
                "provider_thinking_tokens": None,
                "message_content_types": ["str"],
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        )

    @override_settings(
        GEMINI_API_KEY="gemini-test-key",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_gemini_metadata_accounts_for_hidden_output_tokens(self, mock_openai):
        examples = (
            (2067, 70, 3863, 1726, 1796, 99.78),
            (2079, 69, 3875, 1727, 1796, 99.78),
            (2185, 219, 3934, 1530, 1749, 97.17),
        )

        for (
            prompt_tokens,
            visible_tokens,
            total_tokens,
            hidden_tokens,
            combined_tokens,
            utilization,
        ) in examples:
            with self.subTest(total_tokens=total_tokens):
                mock_openai.reset_mock()
                mock_openai.return_value.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"post_text":"x"}'))],
                    model_dump=lambda: {
                        "model": "gemini-3.6-flash",
                        "choices": [{"finish_reason": "length", "message": {"content": "x"}}],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": visible_tokens,
                            "total_tokens": total_tokens,
                        },
                    },
                    usage=SimpleNamespace(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=visible_tokens,
                        total_tokens=total_tokens,
                    ),
                )
                client = build_ai_client("gemini", "gemini-3.6-flash")

                response = client.generate_text(
                    "Prompt",
                    max_output_tokens=1800,
                    json_mode=False,
                )

                metadata = response.provider_response_metadata
                self.assertEqual(metadata["provider_visible_output_tokens"], visible_tokens)
                self.assertEqual(metadata["provider_hidden_output_tokens"], hidden_tokens)
                self.assertEqual(metadata["provider_combined_output_tokens"], combined_tokens)
                self.assertEqual(
                    metadata["provider_output_budget_utilization_percent"],
                    utilization,
                )

    @override_settings(
        GEMINI_API_KEY="gemini-test-key",
        OPENAI_TIMEOUT_SECONDS=30,
    )
    @patch("apps.ai.client.OpenAI")
    def test_gemini_metadata_leaves_hidden_accounting_null_when_not_derivable(
        self,
        mock_openai,
    ):
        cases = (
            ({"prompt_tokens": 2, "completion_tokens": 3}, 1800),
            ({"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 15}, 1800),
            ({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, 0),
        )

        for usage, max_output_tokens in cases:
            with self.subTest(usage=usage, max_output_tokens=max_output_tokens):
                mock_openai.reset_mock()
                mock_openai.return_value.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Gemini"))],
                    model_dump=lambda: {
                        "model": "gemini-3.6-flash",
                        "choices": [{"finish_reason": "stop", "message": {"content": "Gemini"}}],
                        "usage": dict(usage),
                    },
                    usage=SimpleNamespace(**usage),
                )
                client = build_ai_client("gemini", "gemini-3.6-flash")

                response = client.generate_text(
                    "Prompt",
                    max_output_tokens=max_output_tokens,
                )

                metadata = response.provider_response_metadata
                if usage.get("total_tokens", 0) - usage.get("prompt_tokens", 0) - usage.get("completion_tokens", 0) < 0:
                    self.assertIsNone(metadata["provider_hidden_output_tokens"])
                if max_output_tokens == 0:
                    self.assertIsNone(
                        metadata["provider_output_budget_utilization_percent"]
                    )

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


    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key", ANTHROPIC_TIMEOUT_SECONDS=17)
    @patch("apps.ai.client.urlopen")
    def test_build_ai_client_selects_anthropic_native_client(self, mock_urlopen):
        mock_urlopen.return_value = _AnthropicResponse(
            {"content": [{"type": "text", "text": "Claude text"}], "usage": {}}
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        result = client.generate_text("Prompt")

        self.assertEqual(client.provider, "anthropic")
        self.assertEqual(client.model, "claude-sonnet-5")
        self.assertEqual(result.text, "Claude text")
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 17)

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key", ANTHROPIC_TIMEOUT_SECONDS=17)
    @patch("apps.ai.client.urlopen")
    def test_anthropic_generation_uses_native_messages_endpoint_headers_and_body(
        self,
        mock_urlopen,
    ):
        mock_urlopen.return_value = _AnthropicResponse(
            {
                "content": [{"type": "text", "text": "Claude text"}],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        result = client.generate_text("Prompt", max_output_tokens=400, json_mode=False)

        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, ANTHROPIC_MESSAGES_ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["X-api-key"], "anthropic-test-key")
        self.assertEqual(request.headers["Anthropic-version"], ANTHROPIC_API_VERSION)
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(body["model"], "claude-sonnet-5")
        self.assertEqual(body["max_tokens"], 400)
        self.assertEqual(body["messages"], [{"role": "user", "content": "Prompt"}])
        self.assertNotIn("thinking", body)
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 17)
        self.assertEqual(result.text, "Claude text")
        self.assertEqual(
            result.usage,
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )
        self.assertEqual(
            result.provider_response_metadata,
            {
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "stop_reason": None,
                "provider_stop_reason": None,
                "provider_finish_reason": None,
                "provider_max_output_tokens": 400,
                "provider_reported_output_tokens": 3,
                "provider_output_limit_reached": None,
                "provider_prompt_tokens": 2,
                "provider_visible_output_tokens": 3,
                "provider_total_tokens": 5,
                "provider_hidden_output_tokens": 0,
                "provider_combined_output_tokens": 3,
                "provider_output_budget_utilization_percent": 0.75,
                "provider_reasoning_tokens": None,
                "provider_thinking_tokens": None,
                "content_block_types": ["text"],
                "input_tokens": 2,
                "output_tokens": 3,
                "thinking_tokens": None,
            },
        )

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_generation_can_disable_thinking_without_changing_max_tokens(
        self,
        mock_urlopen,
    ):
        mock_urlopen.return_value = _AnthropicResponse(
            {"content": [{"type": "text", "text": "Claude text"}], "usage": {}}
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        client.generate_text(
            "Prompt",
            max_output_tokens=4000,
            thinking_mode=AI_THINKING_MODE_DISABLED,
        )

        body = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(body["max_tokens"], 4000)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        mock_urlopen.assert_called_once()

    @override_settings(OPENAI_API_KEY="openai-test-key", OPENAI_TIMEOUT_SECONDS=30)
    @patch("apps.ai.client.OpenAI")
    def test_openai_rejects_non_default_thinking_mode_before_invocation(
        self,
        mock_openai,
    ):
        client = build_ai_client("openai", "gpt-4.1-2025-04-14")

        with self.assertRaisesRegex(ValueError, "unsupported AI thinking_mode"):
            client.generate_text("Prompt", thinking_mode=AI_THINKING_MODE_DISABLED)

        mock_openai.return_value.responses.create.assert_not_called()
        mock_openai.return_value.chat.completions.create.assert_not_called()

    @override_settings(GEMINI_API_KEY="gemini-test-key", OPENAI_TIMEOUT_SECONDS=30)
    @patch("apps.ai.client.OpenAI")
    def test_gemini_rejects_non_default_thinking_mode_before_invocation(
        self,
        mock_openai,
    ):
        client = build_ai_client("gemini", "gemini-3.6-flash")

        with self.assertRaisesRegex(ValueError, "unsupported AI thinking_mode"):
            client.generate_text("Prompt", thinking_mode=AI_THINKING_MODE_DISABLED)

        mock_openai.return_value.responses.create.assert_not_called()
        mock_openai.return_value.chat.completions.create.assert_not_called()

    def test_provider_default_thinking_mode_is_always_supported(self):
        for provider in ("openai", "gemini", "anthropic"):
            with self.subTest(provider=provider):
                self.assertIsNone(
                    get_ai_provider_thinking_mode_error(
                        provider=provider,
                        thinking_mode=AI_THINKING_MODE_PROVIDER_DEFAULT,
                    )
                )

    def test_unknown_thinking_mode_is_rejected(self):
        self.assertEqual(
            get_ai_provider_thinking_mode_error(
                provider="anthropic",
                thinking_mode="turbo-think",
            ),
            "unsupported AI thinking_mode: turbo-think",
        )

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_json_mode_uses_system_json_instruction(self, mock_urlopen):
        mock_urlopen.return_value = _AnthropicResponse(
            {"content": [{"type": "text", "text": '{"pass": true}'}], "usage": {}}
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        result = client.generate_text(
            "Prompt",
            max_output_tokens=400,
            json_mode=True,
            allow_json_mode_fallback=False,
        )

        body = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn("Return only valid JSON", body["system"])
        self.assertEqual(result.text, '{"pass": true}')
        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_multiple_text_blocks_are_concatenated_in_order(
        self,
        mock_urlopen,
    ):
        mock_urlopen.return_value = _AnthropicResponse(
            {
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": " second"},
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        result = client.generate_text("Prompt")

        self.assertEqual(result.text, "first second")

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

    @override_settings(ANTHROPIC_API_KEY="")
    def test_missing_anthropic_key_returns_sanitized_configuration_error(self):
        error = get_ai_client_configuration_error(
            "anthropic",
            "claude-sonnet-5",
            stage_name="AI",
        )

        self.assertEqual(
            error,
            "ANTHROPIC_API_KEY must be configured with a real key before "
            "AI provider execution",
        )
        self.assertNotIn("anthropic-test-key", error)

    def test_unsupported_provider_is_rejected_before_client_construction(self):
        self.assertEqual(
            get_ai_provider_model_error(
                "unknown",
                "model",
                stage_name="AI",
            ),
            "unsupported AI provider: unknown",
        )

    def test_anthropic_provider_rejects_unsupported_claude_models(self):
        error = get_ai_provider_model_error(
            "anthropic",
            "claude-3-5-sonnet",
            stage_name="AI",
        )

        self.assertIn("AI provider/model mismatch", error)
        self.assertIn("claude-sonnet-5", error)

    def test_openai_provider_rejects_anthropic_model_before_invocation(self):
        self.assertEqual(
            get_ai_provider_model_error(
                "openai",
                "claude-sonnet-5",
                stage_name="AI",
            ),
            "AI provider/model mismatch: provider openai cannot use "
            "an Anthropic model",
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

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_http_error_is_sanitized_and_not_retried(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            ANTHROPIC_MESSAGES_ENDPOINT,
            429,
            "secret anthropic provider body",
            {},
            None,
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        with self.assertRaisesRegex(RuntimeError, "anthropic provider request failed") as cm:
            client.generate_text("Prompt", json_mode=True)

        formatted_traceback = "".join(
            traceback.format_exception(
                type(cm.exception),
                cm.exception,
                cm.exception.__traceback__,
            )
        )
        self.assertNotIn("secret anthropic provider body", str(cm.exception))
        self.assertEqual(cm.exception.status_code, 429)
        self.assertEqual(cm.exception.code, "http_429")
        self.assertIsNone(cm.exception.__cause__)
        self.assertTrue(cm.exception.__suppress_context__)
        self.assertNotIn("secret anthropic provider body", formatted_traceback)
        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_network_error_is_sanitized_and_not_retried(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("secret timeout details")
        client = build_ai_client("anthropic", "claude-sonnet-5")

        with self.assertRaisesRegex(RuntimeError, "anthropic provider request failed") as cm:
            client.generate_text("Prompt")

        self.assertNotIn("secret timeout details", str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)
        self.assertTrue(cm.exception.__suppress_context__)
        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_invalid_json_error_does_not_chain_provider_body(
        self,
        mock_urlopen,
    ):
        mock_urlopen.return_value = _AnthropicResponse(
            b'{"secret": "anthropic provider raw response"'
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        with self.assertRaisesRegex(
            RuntimeError,
            "anthropic provider response was not valid JSON",
        ) as cm:
            client.generate_text("Prompt")

        formatted_traceback = "".join(
            traceback.format_exception(
                type(cm.exception),
                cm.exception,
                cm.exception.__traceback__,
            )
        )
        self.assertNotIn("anthropic provider raw response", str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)
        self.assertTrue(cm.exception.__suppress_context__)
        self.assertNotIn("anthropic provider raw response", formatted_traceback)
        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_invalid_shape_error_is_sanitized(self, mock_urlopen):
        mock_urlopen.return_value = _AnthropicResponse({"content": {"type": "text"}})
        client = build_ai_client("anthropic", "claude-sonnet-5")

        with self.assertRaisesRegex(
            RuntimeError,
            "anthropic provider response shape was invalid",
        ):
            client.generate_text("Prompt")

        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_malformed_content_block_shape_is_sanitized(self, mock_urlopen):
        mock_urlopen.return_value = _AnthropicResponse(
            {"content": [{"text": "missing type"}]}
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        with self.assertRaisesRegex(
            RuntimeError,
            "anthropic provider response shape was invalid",
        ):
            client.generate_text("Prompt")

        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_malformed_text_block_shape_is_sanitized(self, mock_urlopen):
        invalid_text_values = (None, 123, True, {"text": "secret"})

        for invalid_text in invalid_text_values:
            with self.subTest(invalid_text=invalid_text):
                mock_urlopen.reset_mock()
                mock_urlopen.return_value = _AnthropicResponse(
                    {"content": [{"type": "text", "text": invalid_text}]}
                )
                client = build_ai_client("anthropic", "claude-sonnet-5")

                with self.assertRaisesRegex(
                    RuntimeError,
                    "anthropic provider response shape was invalid",
                ):
                    client.generate_text("Prompt")

                mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_malformed_usage_shape_is_sanitized(self, mock_urlopen):
        mock_urlopen.return_value = _AnthropicResponse(
            {
                "content": [{"type": "text", "text": "Claude text"}],
                "usage": {"input_tokens": 2, "output_tokens": "secret"},
            }
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        with self.assertRaisesRegex(
            RuntimeError,
            "anthropic provider response shape was invalid",
        ):
            client.generate_text("Prompt")

        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_metadata_bounds_content_block_type_values(self, mock_urlopen):
        long_block_type = "custom-" + ("secret-" * 40)
        mock_urlopen.return_value = _AnthropicResponse(
            {
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "content": [{"type": long_block_type}],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        result = client.generate_text("Prompt")

        block_type = result.provider_response_metadata["content_block_types"][0]
        self.assertEqual(block_type, long_block_type[:120])
        self.assertLessEqual(len(block_type), 120)
        self.assertNotIn("secret-" * 25, json.dumps(result.provider_response_metadata))

        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_non_text_valid_response_returns_empty_text_with_safe_metadata(
        self,
        mock_urlopen,
    ):
        mock_urlopen.return_value = _AnthropicResponse(
            {
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "content": [{"type": "tool_use", "name": "lookup"}],
                "usage": {"input_tokens": 7, "output_tokens": 0},
            }
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        result = client.generate_text("Prompt")

        self.assertEqual(result.text, "")
        self.assertEqual(result.raw, {})
        self.assertEqual(
            result.usage,
            {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7},
        )
        self.assertEqual(
            result.provider_response_metadata,
            {
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "provider_stop_reason": "end_turn",
                "provider_finish_reason": None,
                "provider_max_output_tokens": 1200,
                "provider_reported_output_tokens": 0,
                "provider_output_limit_reached": False,
                "provider_prompt_tokens": 7,
                "provider_visible_output_tokens": 0,
                "provider_total_tokens": 7,
                "provider_hidden_output_tokens": 0,
                "provider_combined_output_tokens": 0,
                "provider_output_budget_utilization_percent": 0.0,
                "provider_reasoning_tokens": None,
                "provider_thinking_tokens": None,
                "content_block_types": ["tool_use"],
                "input_tokens": 7,
                "output_tokens": 0,
                "thinking_tokens": None,
            },
        )
        serialized = json.dumps(result.provider_response_metadata, sort_keys=True)
        self.assertNotIn("lookup", serialized)
        self.assertNotIn("Prompt", serialized)
        self.assertNotIn("x-api-key", serialized)

        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_thinking_only_valid_response_returns_empty_text_with_safe_metadata(
        self,
        mock_urlopen,
    ):
        mock_urlopen.return_value = _AnthropicResponse(
            {
                "model": "claude-sonnet-5",
                "stop_reason": "max_tokens",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "secret provider thinking text",
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 13,
                    "thinking_tokens": 13,
                },
            }
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        result = client.generate_text("Prompt")

        self.assertEqual(result.text, "")
        self.assertEqual(result.raw, {})
        self.assertEqual(
            result.provider_response_metadata,
            {
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "stop_reason": "max_tokens",
                "provider_stop_reason": "max_tokens",
                "provider_finish_reason": None,
                "provider_max_output_tokens": 1200,
                "provider_reported_output_tokens": 13,
                "provider_output_limit_reached": True,
                "provider_prompt_tokens": 11,
                "provider_visible_output_tokens": 13,
                "provider_total_tokens": 24,
                "provider_hidden_output_tokens": 0,
                "provider_combined_output_tokens": 13,
                "provider_output_budget_utilization_percent": 1.08,
                "provider_reasoning_tokens": None,
                "provider_thinking_tokens": 13,
                "content_block_types": ["thinking"],
                "input_tokens": 11,
                "output_tokens": 13,
                "thinking_tokens": 13,
            },
        )
        serialized = json.dumps(result.provider_response_metadata, sort_keys=True)
        self.assertNotIn("secret provider thinking text", serialized)
        self.assertNotIn("Prompt", serialized)

        mock_urlopen.assert_called_once()

    @override_settings(ANTHROPIC_API_KEY="anthropic-test-key")
    @patch("apps.ai.client.urlopen")
    def test_anthropic_empty_content_valid_response_returns_empty_text_with_metadata(
        self,
        mock_urlopen,
    ):
        mock_urlopen.return_value = _AnthropicResponse(
            {
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 0},
            }
        )
        client = build_ai_client("anthropic", "claude-sonnet-5")

        result = client.generate_text("Prompt")

        self.assertEqual(result.text, "")
        self.assertEqual(result.raw, {})
        self.assertEqual(result.provider_response_metadata["content_block_types"], [])
        self.assertEqual(result.provider_response_metadata["stop_reason"], "end_turn")

        mock_urlopen.assert_called_once()



class _AnthropicResponse:
    def __init__(self, payload: dict | bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")
