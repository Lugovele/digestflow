from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from django.test import override_settings

from apps.ai.client import OpenAIClient, _extract_usage, estimate_cost_usd


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
