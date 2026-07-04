from django.test import SimpleTestCase

from services.packaging.linkedin_final_post_diagnostics import (
    TextLeak,
    diagnose_final_post_payload,
)


class LinkedInFinalPostDiagnosticsTests(SimpleTestCase):
    def test_missing_quality_check_keys_are_reported(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(quality_checks={"linkedin_ready": True}),
            selected_evidence_ids=[],
            schema_validation_passed=True,
        )

        self.assertEqual(
            diagnostics.missing_quality_check_keys,
            ["has_clear_point_of_view", "uses_only_provided_facts"],
        )
        self.assertIn("missing_quality_checks", diagnostics.repair_reasons)

    def test_non_boolean_quality_check_values_are_reported(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(quality_checks={**_quality_checks(), "linkedin_ready": "yes"}),
            selected_evidence_ids=[],
            schema_validation_passed=True,
        )

        self.assertEqual(diagnostics.non_boolean_quality_check_keys, ["linkedin_ready"])
        self.assertIn("non_boolean_quality_checks", diagnostics.repair_reasons)
        self.assertIn("model_not_linkedin_ready", diagnostics.repair_reasons)

    def test_evidence_id_leaks_in_post_text_are_reported(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(post_text="This post mentions a0-summary in copy."),
            selected_evidence_ids=["a0-summary"],
            schema_validation_passed=True,
        )

        self.assertEqual(len(diagnostics.evidence_id_leaks), 1)
        self.assertEqual(diagnostics.evidence_id_leaks[0].field_name, "post_text")
        self.assertEqual(diagnostics.evidence_id_leaks[0].value_index, None)
        self.assertIn("evidence_ids_in_human_text", diagnostics.repair_reasons)

    def test_evidence_id_leaks_in_hook_variants_are_reported(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(hook_variants=["Hook one", "Hook with a1-kp0", "Hook three"]),
            selected_evidence_ids=["a1-kp0"],
            schema_validation_passed=True,
        )

        self.assertEqual(len(diagnostics.evidence_id_leaks), 1)
        self.assertEqual(diagnostics.evidence_id_leaks[0].field_name, "hook_variants")
        self.assertEqual(diagnostics.evidence_id_leaks[0].value_index, 1)

    def test_evidence_id_leaks_in_cta_variants_are_reported(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(cta_variants=["CTA one", "CTA two", "CTA with a2-summary"]),
            selected_evidence_ids=["a2-summary"],
            schema_validation_passed=True,
        )

        self.assertEqual(len(diagnostics.evidence_id_leaks), 1)
        self.assertEqual(diagnostics.evidence_id_leaks[0].field_name, "cta_variants")
        self.assertEqual(diagnostics.evidence_id_leaks[0].value_index, 2)

    def test_scaffold_phrase_leaks_are_case_insensitive(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(post_text="Do not merely SEPARATE SIGNALS in final copy."),
            selected_evidence_ids=[],
            schema_validation_passed=True,
        )

        self.assertEqual(len(diagnostics.scaffold_phrase_leaks), 1)
        self.assertEqual(diagnostics.scaffold_phrase_leaks[0].phrase, "separate signals")
        self.assertIn("scaffold_language_in_human_text", diagnostics.repair_reasons)

    def test_source_summary_phrase_leaks_are_case_insensitive(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(hook_variants=["Hook one", "SOURCES MATTER here", "Hook three"]),
            selected_evidence_ids=[],
            schema_validation_passed=True,
        )

        self.assertEqual(len(diagnostics.source_summary_phrase_leaks), 1)
        self.assertEqual(diagnostics.source_summary_phrase_leaks[0].phrase, "sources matter")
        self.assertIn("source_summary_language_in_human_text", diagnostics.repair_reasons)

    def test_system_linkedin_ready_is_false_when_schema_validation_failed(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(),
            selected_evidence_ids=[],
            schema_validation_passed=False,
            schema_validation_error="missing field",
        )

        self.assertFalse(diagnostics.system_linkedin_ready)
        self.assertTrue(diagnostics.deterministic_checks_passed)
        self.assertEqual(diagnostics.schema_validation_error, "missing field")
        self.assertIn("schema_validation_failed", diagnostics.repair_reasons)

    def test_system_linkedin_ready_is_false_when_model_claims_ready_but_checks_fail(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(post_text="This mentions selected evidence directly."),
            selected_evidence_ids=[],
            schema_validation_passed=True,
        )

        self.assertTrue(diagnostics.model_claimed_linkedin_ready)
        self.assertFalse(diagnostics.deterministic_checks_passed)
        self.assertFalse(diagnostics.system_linkedin_ready)

    def test_clean_payload_returns_system_linkedin_ready_true(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _payload(),
            selected_evidence_ids=["a0-summary"],
            schema_validation_passed=True,
        )

        self.assertTrue(diagnostics.deterministic_checks_passed)
        self.assertTrue(diagnostics.system_linkedin_ready)
        self.assertEqual(diagnostics.repair_reasons, [])

    def test_to_dict_returns_serializable_dictionaries_with_leak_entries(self) -> None:
        leak = TextLeak(
            field_name="post_text",
            value_index=None,
            phrase="selected evidence",
            leak_type="scaffold_phrase",
        )
        self.assertEqual(
            leak.to_dict(),
            {
                "field_name": "post_text",
                "value_index": None,
                "phrase": "selected evidence",
                "leak_type": "scaffold_phrase",
            },
        )

        diagnostics = diagnose_final_post_payload(
            _payload(post_text="This repeats selected evidence."),
            selected_evidence_ids=[],
            schema_validation_passed=True,
        )

        diagnostics_dict = diagnostics.to_dict()
        self.assertIsInstance(diagnostics_dict, dict)
        self.assertEqual(
            diagnostics_dict["scaffold_phrase_leaks"][0],
            {
                "field_name": "post_text",
                "value_index": None,
                "phrase": "selected evidence",
                "leak_type": "scaffold_phrase",
            },
        )


def _payload(**overrides):
    payload = {
        "post_text": "A concise human-facing post about the selected topic.",
        "hook_variants": ["Hook one", "Hook two", "Hook three"],
        "cta_variants": ["CTA one", "CTA two", "CTA three"],
        "hashtags": ["#AI"],
        "quality_checks": _quality_checks(),
        "carousel_outline": [],
    }
    payload.update(overrides)
    return payload


def _quality_checks():
    return {
        "linkedin_ready": True,
        "uses_only_provided_facts": True,
        "has_clear_point_of_view": True,
    }
