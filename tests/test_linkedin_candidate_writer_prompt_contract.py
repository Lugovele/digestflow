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
