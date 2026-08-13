from __future__ import annotations

from pathlib import Path

from django.test import SimpleTestCase


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "linkedin"
    / "final_post_semantic_grounding_evaluator.txt"
)


def _prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


class LinkedInPostSemanticGroundingPromptContractTests(SimpleTestCase):
    def test_prompt_treats_rendered_rules_as_authoritative(self) -> None:
        prompt = _prompt_text()

        self.assertIn("SEMANTIC_GROUNDING_RULES_JSON", prompt)
        self.assertIn("authoritative source", prompt)
        self.assertIn("Do not override or reinterpret those rules", prompt)

    def test_prompt_does_not_keep_independent_status_severity_matrix(self) -> None:
        prompt = _prompt_text()

        self.assertNotIn("Allowed support_status values:", prompt)
        self.assertNotIn("Allowed severity values:", prompt)
        self.assertNotIn("Set pass to false when any major or blocking claim", prompt)

    def test_prompt_uses_structured_repair_instruction_schema(self) -> None:
        prompt = _prompt_text()

        self.assertIn('"repair_instructions": [', prompt)
        self.assertIn('"claim_id": "c1"', prompt)
        self.assertIn('"instruction": "Remove or qualify unsupported strengthened claims."', prompt)

    def test_prompt_preserves_strict_json_only_output(self) -> None:
        prompt = _prompt_text()

        self.assertIn("Return only JSON compatible with this shape:", prompt)

    def test_prompt_allows_source_bounded_authorial_synthesis(self) -> None:
        text = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("Source-bounded authorial synthesis is allowed", text)
        self.assertIn("Do not mark authorial synthesis", text)
        self.assertIn("exact wording is absent", text)
        self.assertIn("block invented facts", text)
        self.assertIn("causal mechanisms", text)

    def test_prompt_requires_mixed_claim_splitting(self) -> None:
        text = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("Split mixed rhetorical/authorial framing", text)
        self.assertIn("factual or causal assertions", text)

    def test_prompt_requires_rhetorical_setup_to_use_local_discourse_context(self) -> None:
        text = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("Evaluate rhetorical setup in local discourse context", text)
        self.assertIn("immediately rejects or qualifies it", text)
        self.assertIn("rejected proposition", text)
        self.assertIn("embedded factual, metric, predictive, causal", text)


    def test_prompt_treats_forecast_wording_as_projection_qualifier(self) -> None:
        text = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("treat forecast", text)
        self.assertIn("projection qualifiers", text)
        self.assertIn("forecast is aggressive", text)
        self.assertIn("guaranteed outcome", text)

    def test_prompt_does_not_show_legacy_string_repair_instruction_schema(self) -> None:
        prompt = _prompt_text()

        self.assertNotIn(
            '"repair_instructions": ["Remove or qualify unsupported strengthened claims."]',
            prompt,
        )
