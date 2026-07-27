from __future__ import annotations

import copy
import inspect

from django.test import SimpleTestCase

from services.packaging import linkedin_post_deterministic_gate
from services.packaging.linkedin_post_deterministic_gate import (
    run_final_post_deterministic_gate,
)
from services.packaging.linkedin_post_flow_handoffs import (
    CandidateWriterOutput,
    DeterministicGateOutput,
)


class LinkedInPostDeterministicGateTests(SimpleTestCase):
    def test_valid_candidate_payload_returns_deterministic_gate_output(self) -> None:
        output = run_final_post_deterministic_gate(
            _valid_payload(),
            selected_evidence_ids=("a0-summary", "a1-kp0"),
        )

        self.assertIsInstance(output, DeterministicGateOutput)

    def test_valid_candidate_payload_has_validation_passed_true(self) -> None:
        output = run_final_post_deterministic_gate(
            _valid_payload(),
            selected_evidence_ids=("a0-summary",),
        )

        self.assertTrue(output.validation_passed)
        self.assertEqual(output.validation_error, "")
        self.assertTrue(output.diagnostics.schema_validation_passed)

    def test_invalid_schema_returns_validation_passed_false(self) -> None:
        payload = _valid_payload()
        payload.pop("post_text")

        output = run_final_post_deterministic_gate(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertFalse(output.validation_passed)
        self.assertFalse(output.diagnostics.schema_validation_passed)

    def test_validation_error_is_captured_as_text(self) -> None:
        payload = _valid_payload()
        payload["hook_variants"] = ["Only one hook"]

        output = run_final_post_deterministic_gate(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertFalse(output.validation_passed)
        self.assertIn("FinalPostPayload.hook_variants", output.validation_error)

    def test_schema_validation_failure_is_in_diagnostics_repair_reasons(self) -> None:
        payload = _valid_payload()
        payload["quality_checks"] = {"linkedin_ready": True}

        output = run_final_post_deterministic_gate(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertIn("schema_validation_failed", output.diagnostics.repair_reasons)

    def test_evidence_id_leaks_are_reported(self) -> None:
        payload = _valid_payload(
            post_text="This post leaks a0-summary in human-facing text.",
        )

        output = run_final_post_deterministic_gate(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertEqual(len(output.diagnostics.evidence_id_leaks), 1)
        self.assertEqual(output.diagnostics.evidence_id_leaks[0].phrase, "a0-summary")

    def test_scaffold_phrase_leaks_are_reported(self) -> None:
        payload = _valid_payload(
            hook_variants=[
                "Separate signals before making broad claims.",
                "Second hook",
                "Third hook",
            ],
        )

        output = run_final_post_deterministic_gate(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertEqual(len(output.diagnostics.scaffold_phrase_leaks), 1)
        self.assertEqual(
            output.diagnostics.scaffold_phrase_leaks[0].phrase,
            "separate signals",
        )

    def test_source_summary_phrase_leaks_are_reported(self) -> None:
        payload = _valid_payload(
            cta_variants=[
                "According to one source, this matters.",
                "Second CTA",
                "Third CTA",
            ],
        )

        output = run_final_post_deterministic_gate(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertEqual(len(output.diagnostics.source_summary_phrase_leaks), 1)
        self.assertEqual(
            output.diagnostics.source_summary_phrase_leaks[0].phrase,
            "according to one source",
        )

    def test_original_payload_remains_unchanged(self) -> None:
        payload = _valid_payload()
        before = copy.deepcopy(payload)

        output = run_final_post_deterministic_gate(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertEqual(payload, before)
        self.assertIs(output.payload, payload)

    def test_accepts_candidate_writer_output_as_input(self) -> None:
        candidate_output = CandidateWriterOutput(
            payload=_valid_payload(),
            raw_output=None,
            provider=None,
            model=None,
            prompt_name=None,
            prompt_version=None,
            token_usage=None,
            cost_metadata=None,
        )

        output = run_final_post_deterministic_gate(
            candidate_output,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertTrue(output.validation_passed)
        self.assertIs(output.payload, candidate_output.payload)

    def test_selected_evidence_ids_are_preserved(self) -> None:
        output = run_final_post_deterministic_gate(
            _valid_payload(),
            selected_evidence_ids=("a0-summary", "a1-kp0"),
        )

        self.assertEqual(output.selected_evidence_ids, ("a0-summary", "a1-kp0"))

    def test_gate_does_not_import_execution_or_orchestration_layers(self) -> None:
        source = inspect.getsource(linkedin_post_deterministic_gate)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
        self.assertNotIn("linkedin_post_prompt_registry", source)
        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)
        self.assertNotIn("QualityReviewResult", source)
        self.assertNotIn("TargetedRepairPlan", source)
        self.assertNotIn("FinalPostDecision", source)

    def test_gate_does_not_reference_raw_articles(self) -> None:
        source = inspect.getsource(linkedin_post_deterministic_gate).lower()

        self.assertNotIn("raw_article", source)
        self.assertNotIn("articles", source)


def _valid_payload(**overrides) -> dict:
    payload = {
        "post_text": "Clear final post text for LinkedIn.",
        "hook_variants": ["First hook", "Second hook", "Third hook"],
        "cta_variants": ["First CTA", "Second CTA", "Third CTA"],
        "hashtags": ["#FutureOfWork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }
    payload.update(overrides)
    return payload
