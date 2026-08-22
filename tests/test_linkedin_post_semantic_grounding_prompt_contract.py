from __future__ import annotations

from pathlib import Path

from django.test import SimpleTestCase


PROMPT_PATH = Path("prompts/linkedin/final_post_semantic_grounding_evaluator.txt")


class LinkedInPostSemanticGroundingPromptContractTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
        cls.prompt_lower = cls.prompt_text.lower()

    def test_prompt_uses_rendered_semantic_grounding_rules_as_authority(self) -> None:
        self.assertIn("semantic_grounding_rules_json", self.prompt_lower)
        self.assertIn("authoritative decision contract", self.prompt_lower)
        self.assertIn("do not invent alternate", self.prompt_lower)

    def test_prompt_does_not_duplicate_complete_status_taxonomy(self) -> None:
        for heading in (
            "allowed support_status values",
            "allowed severity values",
        ):
            with self.subTest(heading=heading):
                self.assertNotIn(heading, self.prompt_lower)

    def test_prompt_requires_claim_order_failed_ids_consistency(self) -> None:
        self.assertIn("failed_claim_ids", self.prompt_text)
        self.assertIn("exactly matches the derived", self.prompt_lower)
        self.assertIn("claim_reviews order", self.prompt_lower)

    def test_prompt_describes_advisory_and_blocking_partial_support(self) -> None:
        self.assertIn("partially_supported with minor severity is advisory", self.prompt_lower)
        self.assertIn("partially_supported with major severity is blocking", self.prompt_lower)

    def test_prompt_requires_claim_addressed_repair_instructions(self) -> None:
        self.assertIn("repairable=true", self.prompt_lower)
        self.assertIn("repair_hint", self.prompt_text)
        self.assertIn("claim id followed by a", self.prompt_lower)
        self.assertIn('"c1:', self.prompt_text)

    def test_prompt_remains_semantic_not_editorial(self) -> None:
        self.assertIn("not editorial quality scoring", self.prompt_lower)
        self.assertIn("use only selected evidence", self.prompt_lower)
        self.assertNotIn("human voice score", self.prompt_lower)
