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
                "CANDIDATE_POST_LENGTH_INSTRUCTION",
                "CANDIDATE_POST_CONSTRAINTS_JSON",
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
                "AuthorialVoiceDirective",
                "authorial_observation",
                "rejected_reading",
                "why_distinction_matters",
                "personal_presence_requirement",
                "first_person_policy",
                "forbidden_author_claims",
                "author-owned judgment",
            ],
        )

    def test_candidate_writer_prompt_uses_personal_presence_instruction(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "PERSONAL_PRESENCE_INSTRUCTION",
                "bounded personal-presence requirement",
                "explicit author-owned interpretive statement",
                "exactly one naturally integrated",
                "visibly assigns the interpretation to the author",
            ],
        )

    def test_candidate_writer_prompt_rejects_weak_personal_presence(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "An impersonal editorial judgment is not sufficient",
                "First-person grammar alone is not sufficient",
                "I think this is interesting is not a qualifying statement",
                "Do not mechanically prepend I think",
                "Do not include more than one explicit ownership statement",
            ],
        )

    def test_candidate_writer_prompt_forbids_invented_author_context(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Do not invent personal experience",
                "professional authority",
                "direct market exposure",
                "client or customer stories",
                "invented emotional reaction",
                "biographical claims",
                "personal context not supplied by the input",
            ],
        )

    def test_candidate_writer_prompt_allows_but_does_not_force_first_person(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "allowed_not_required",
                "first person is allowed when natural",
                "must not be forced",
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

    def test_candidate_writer_prompt_forbids_evidence_ids_in_human_facing_text(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Selected evidence IDs are input-side traceability references only",
                "CandidatePost currently has no traceability field",
                "do not output evidence IDs",
                "Do not include evidence IDs in post_text",
            ],
        )

    def test_candidate_writer_prompt_declares_candidate_post_json_contract(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Return only valid JSON",
                "CandidatePost contract",
                "exactly one content field",
                "post_text",
                "Do not include hook_variants",
                "cta_variants",
                "hashtags",
                "quality_checks",
                "carousel_outline",
            ],
        )

    def test_candidate_writer_prompt_requires_one_strict_json_object(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Return one valid JSON object only",
                "Do not include prose before or after the JSON object",
                "double-quoted keys",
                "double-quoted string values",
                "strict JSON",
                "without preprocessing or repair",
            ],
        )

    def test_candidate_writer_prompt_forbids_non_strict_json_syntax(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Do not wrap the JSON object in markdown code fences",
                "Do not use single-quoted pseudo-JSON",
                "Do not include comments",
                "Do not use trailing commas",
            ],
        )

    def test_candidate_writer_prompt_requires_escaped_json_string_newlines(self):
        prompt = _prompt_text()

        self.assertIn("All line breaks inside JSON string values must be encoded", prompt)
        self.assertIn(r"\n", prompt)
        self.assertIn(
            r"If post_text needs paragraph breaks, encode them inside the JSON string as \n\n",
            prompt,
        )
        self.assertIn(
            "Never place a literal physical newline inside a quoted JSON string",
            prompt,
        )

    def test_candidate_writer_prompt_uses_canonical_payload_constraints(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "CANDIDATE_POST_LENGTH_INSTRUCTION",
                "CANDIDATE_POST_CONSTRAINTS_JSON",
                "canonical structural contract",
                "post_text",
                "direct numeric target length",
                "hard maximum",
            ],
        )

    def test_candidate_writer_prompt_does_not_duplicate_numeric_payload_limits(self):
        prompt = _normalized_prompt_text()

        self.assertNotIn("1300", prompt)
        self.assertNotIn("1200", prompt)
        self.assertNotIn("1100", prompt)
        self.assertIn("candidate_post_constraints_json", prompt)

    def test_candidate_writer_prompt_uses_rendered_length_instruction(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "CANDIDATE_POST_LENGTH_INSTRUCTION",
                "direct numeric target length",
                "hard maximum",
                "rendered from the canonical CandidatePost constraints",
                "Do not treat CANDIDATE_POST_CONSTRAINTS_JSON as the writing target",
                "canonical structural contract",
            ],
        )

    def test_candidate_writer_prompt_is_role_level_not_provider_specific(self):
        prompt = _normalized_prompt_text()

        _assert_contains_all(
            self,
            prompt,
            [
                "Candidate Writer prompt contract",
                "role-level and provider-neutral",
                "same prompt contract applies",
                "approved Candidate Writer provider or model",
            ],
        )
        self.assertNotIn("gpt-4.1", prompt)
        self.assertNotIn("gpt-4o-mini", prompt)
        self.assertNotIn("claude", prompt)
        self.assertNotIn("gemini", prompt)

    def test_candidate_writer_prompt_does_not_introduce_length_repair_or_compression(self):
        prompt = _normalized_prompt_text()

        for forbidden in (
            "compress",
            "truncate",
            "trunca" + "tion",
            "com" + "pressor",
            "second writer " + "call",
            "shorten to fit",
            "retry",
            "fall" + "back",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, prompt)

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
