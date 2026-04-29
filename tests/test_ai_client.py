from types import SimpleNamespace

from django.test import SimpleTestCase

from apps.ai.client import _extract_usage, estimate_cost_usd


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
