from pathlib import Path

from django.test import SimpleTestCase

from services.packaging.linkedin_post_prompt_registry import (
    PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF,
    get_prompt_contract,
)


def _prompt_path() -> Path:
    contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)
    return Path(__file__).resolve().parents[1] / contract.prompt_path


def _prompt_text() -> str:
    return _prompt_path().read_text(encoding="utf-8")


def _normalized_prompt_text() -> str:
    return _prompt_text().lower()


def _assert_contains_all(test_case: SimpleTestCase, text: str, expected: list[str]) -> None:
    for phrase in expected:
        test_case.assertIn(phrase.lower(), text)


class LinkedInCandidateWriterPromptContractTests(SimpleTestCase):
    def test_candidate_writer_prompt_declares_input_boundaries(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "PostBrief",
                "AngleDecision",
                "selected evidence",
                "Use only selected evidence from PostBrief.evidence_to_use",
            ],
        )

    def test_candidate_writer_prompt_declares_source_of_truth_rules(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "PostBrief as the source of truth",
                "PostBrief is the source of truth for exact selected evidence",
                "Exact evidence text is preserved upstream through PostBrief",
                "Use only selected evidence from PostBrief.evidence_to_use",
                "Do not use non-selected evidence",
            ],
        )

    def test_candidate_writer_prompt_preserves_controlling_angle(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Preserve AngleDecision.controlling_angle",
            ],
        )

    def test_candidate_writer_prompt_uses_authorial_voice_directive(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "AngleDecision.authorial_voice_directive",
                "source of truth",
                "authorial judgment",
                "what the author notices",
                "what reading",
                "rejects",
                "why the distinction matters",
                "First person is allowed but not required",
            ],
        )

    def test_candidate_writer_prompt_uses_personal_presence_instruction(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "personal_presence_instruction",
                "resolved policy",
                "bounded personal-presence requirement",
                "explicit personal presence",
            ],
        )

    def test_candidate_writer_prompt_requires_one_author_owned_interpretation_when_required(
        self,
    ):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "requires an explicit author-owned interpretive statement",
                "exactly one naturally integrated",
                "visibly assigns the interpretation to the author",
                "authorial_observation",
                "rejected_reading",
                "why_distinction_matters",
                "AngleDecision.controlling_angle",
            ],
        )

    def test_candidate_writer_prompt_rejects_impersonal_or_generic_presence(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "An impersonal editorial judgment is not sufficient",
                "Do not mechanically prepend I think",
                "in my opinion",
                "personally",
                "Do not require the post to open with first person",
                "Do not include more than one explicit ownership statement",
            ],
        )

    def test_candidate_writer_prompt_forbids_fake_author_claims(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Do not invent biography",
                "personal experience",
                "professional authority",
                "direct",
                "market exposure",
                "client stories",
                "customer stories",
                "emotional reactions",
            ],
        )

    def test_candidate_writer_prompt_declares_factuality_and_no_invention_rules(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Do not invent facts",
                "metrics",
                "examples",
                "company names",
                "dates",
                "claims",
                "Do not infer causality",
            ],
        )

    def test_candidate_writer_prompt_requires_human_facing_finished_post(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Write a finished LinkedIn post",
                "research summary",
                "evidence report",
                "generic scaffold language",
                "do not simply repeat it mechanically",
                "Convert scaffold-like brief instructions",
                "clear human-facing editorial angle",
            ],
        )

    def test_candidate_writer_prompt_declares_post_text_length_contract(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "final_post_payload_constraints_json",
                "post_text must be 1300 characters or fewer",
                "1100-1250 characters",
                "hard maximum",
                "do not output a character count field",
            ],
        )

    def test_candidate_writer_prompt_requires_final_length_self_check(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Before returning JSON",
                "verify post_text character count is 1300 characters or fewer",
                "If post_text is too long, shorten it before output",
            ],
        )

    def test_candidate_writer_prompt_forbids_evidence_ids_in_human_facing_text(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Selected evidence IDs are input-side traceability references only",
                "FinalPostPayload currently has no traceability field",
                "do not output evidence IDs",
                "Do not include evidence IDs in post_text, hook_variants, or cta_variants",
            ],
        )

    def test_candidate_writer_prompt_declares_final_post_payload_json_contract(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Return only valid JSON",
                "FinalPostPayload schema",
                "post_text",
                "hook_variants",
                "cta_variants",
                "hashtags",
                "quality_checks",
                "carousel_outline",
                "linkedin_ready",
                "uses_only_provided_facts",
                "has_clear_point_of_view",
            ],
        )

    def test_candidate_writer_prompt_requires_one_strict_json_object(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Return one valid JSON object only",
                "Do not include prose before or after the JSON",
                "valid JSON double-quoted keys",
                "double-quoted string values",
                "strict JSON parser",
                "without preprocessing",
            ],
        )

    def test_candidate_writer_prompt_forbids_json_syntax_that_breaks_strict_parsing(
        self,
    ):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Do not wrap the JSON in markdown fences",
                "Never place literal physical line breaks inside a quoted JSON string",
                "Do not use trailing commas",
                "Do not use comments",
                "Do not use single-quoted JSON",
            ],
        )

    def test_candidate_writer_prompt_requires_escaped_newlines_in_post_text(self):
        prompt = _prompt_text()

        self.assertIn("All line breaks inside JSON string values must be escaped", prompt)
        self.assertIn(r"\n characters", prompt)
        self.assertIn(
            r"If post_text needs paragraph breaks, encode them inside the JSON string as \n\n",
            prompt,
        )

    def test_candidate_writer_prompt_forbids_payload_debug_and_runtime_fields(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "debug fields",
                "repair fields",
                "runtime fields",
                "provider fields",
                "token fields",
                "cost fields",
                "persistence fields",
                "package fields",
            ],
        )
