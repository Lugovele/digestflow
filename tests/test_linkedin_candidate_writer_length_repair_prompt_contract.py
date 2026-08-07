from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from services.packaging.linkedin_post_candidate_writer_length_repair import (
    render_candidate_writer_length_repair_prompt_input,
)


PROMPT_PATH = (
    Path(settings.BASE_DIR)
    / "prompts"
    / "linkedin"
    / "candidate_writer_length_repair.txt"
)


class CandidateWriterLengthRepairPromptContractTests(SimpleTestCase):
    def test_prompt_is_length_only_and_only_changes_post_text(self) -> None:
        prompt = _prompt()

        self.assertIn("length-only repair", prompt)
        self.assertIn("rewrite only post_text", prompt)
        self.assertIn("do not modify hook_variants", prompt)
        self.assertIn("cta_variants", prompt)
        self.assertIn("hashtags", prompt)
        self.assertIn("quality_checks", prompt)
        self.assertIn("carousel_outline", prompt)

    def test_prompt_requires_canonical_maximum_and_target_range(self) -> None:
        prompt = _prompt()

        self.assertIn("length_constraints_json.hard_max_chars", prompt)
        self.assertIn("no longer than", prompt)
        self.assertIn("length_constraints_json.prompt_target_min_chars", prompt)
        self.assertIn("length_constraints_json.prompt_target_max_chars", prompt)
        self.assertNotIn("1300", prompt)

    def test_prompt_preserves_meaning_grounding_voice_and_personal_presence(self) -> None:
        prompt = _prompt()

        for phrase in (
            "central argument",
            "evidence-backed factual claims",
            "qualifications and uncertainty",
            "attribution",
            "approved authorial observation",
            "rejected reading",
            "why the distinction matters",
            "explicit personal presence",
            "cta intent",
        ):
            self.assertIn(phrase, prompt)

    def test_prompt_forbids_new_facts_and_traceability_leaks(self) -> None:
        prompt = _prompt()

        self.assertIn("do not add facts", prompt)
        self.assertIn("metrics", prompt)
        self.assertIn("examples", prompt)
        self.assertIn("evidence ids", prompt)
        self.assertIn("author biography", prompt)

    def test_prompt_requires_minimal_strict_json_response(self) -> None:
        prompt = _prompt()

        self.assertIn("return exactly one strict json object", prompt)
        self.assertIn("exactly one key", prompt)
        self.assertIn('"post_text"', prompt)
        self.assertIn("any extra json keys", prompt)
        self.assertIn("do not return markdown fences", prompt)
        self.assertIn("plain text", prompt)
        self.assertIn("full", prompt)
        self.assertIn("finalpostpayload", prompt)

    def test_prompt_does_not_request_chain_of_thought_or_retry(self) -> None:
        prompt = _prompt()

        self.assertIn("do not return", prompt)
        self.assertIn("chain-of-thought", prompt)
        self.assertNotIn("try again", prompt)
        self.assertNotIn("retry", prompt)
        self.assertNotIn("fallback", prompt)
        self.assertNotIn("json repair", prompt)

    def test_rendered_input_includes_required_repair_context(self) -> None:
        render = render_candidate_writer_length_repair_prompt_input(
            parsed_candidate=_candidate_payload(post_text="A" * 1301),
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
        )

        for heading in (
            "POST_BRIEF_JSON",
            "ANGLE_DECISION_JSON",
            "AUTHORIAL_VOICE_DIRECTIVE_JSON",
            "SELECTED_EVIDENCE_JSON",
            "PRESERVATION_RULES_JSON",
        ):
            self.assertIn(heading, render.input_text)
        self.assertIn("Remote work policies need clear expectations.", render.input_text)
        self.assertIn("Policy clarity is not the same as support.", render.input_text)


def _prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").lower()


def _candidate_payload(*, post_text: object) -> dict:
    return {
        "post_text": post_text,
        "hook_variants": [
            "A practical remote work policy starts here.",
            "Remote policy is not just a document.",
            "Hybrid work needs clearer operating habits.",
        ],
        "cta_variants": [
            "What would you clarify first in a remote policy?",
            "Where does your team still need shared expectations?",
            "What makes hybrid work sustainable in your organization?",
        ],
        "hashtags": ["#remotework", "#futureofwork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }


def _post_brief() -> dict:
    return {
        "core_point": "Clear remote work policies need human operating habits.",
        "evidence_to_use": [
            {
                "evidence_id": "ev-1",
                "evidence_text": "Remote work policies need clear expectations.",
                "role_in_post": "proof",
            }
        ],
    }


def _angle_decision() -> dict:
    return {
        "controlling_angle": "Remote policies need clarity and inclusion.",
        "authorial_voice_directive": {
            "authorial_observation": "Policy clarity is not the same as support.",
            "rejected_reading": "Do not treat documentation as the whole answer.",
            "why_distinction_matters": "Remote work needs both clarity and care.",
            "personal_presence_requirement": "explicit_author_owned_statement_required",
            "first_person_policy": "allowed_not_required",
            "forbidden_author_claims": ["personal experience"],
        },
    }
