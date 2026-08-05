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
    def test_prompt_uses_structured_repair_instruction_schema(self) -> None:
        prompt = _prompt_text()

        self.assertIn('"repair_instructions": [', prompt)
        self.assertIn('"claim_id": "c1"', prompt)
        self.assertIn('"instruction": "Remove or qualify unsupported strengthened claims."', prompt)

    def test_prompt_does_not_show_legacy_string_repair_instruction_schema(self) -> None:
        prompt = _prompt_text()

        self.assertNotIn(
            '"repair_instructions": ["Remove or qualify unsupported strengthened claims."]',
            prompt,
        )
