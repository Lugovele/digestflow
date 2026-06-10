import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.topics.models import Topic
from apps.digests.models import Digest, DigestRun
from services.packaging import generate_content_package_for_digest
from services.packaging.generator import (
    PackagingGenerationResult,
    build_editorial_review_prompt,
    _collect_repairable_payload_issues,
    _evaluate_linkedin_post_mechanics,
    _evaluate_post_brief_alignment,
    _evaluate_repair_rewrite_delta,
    _extract_banned_phrases_from_repair_reasons,
    _find_avoid_angle_match,
    _find_concrete_detail_match,
    _generate_editorial_review_via_llm,
    _generate_post_brief_via_llm,
    _validate_editorial_review_payload,
    _validate_post_brief_payload,
    normalize_linkedin_hashtags,
)
from services.packaging.validators import ContentPackageValidationError


@override_settings(OPENAI_API_KEY="sk-your-key", PACKAGING_EDITORIAL_REVIEW_ENABLED=False)
class PackagingArticlesOnlyTests(TestCase):
    def _create_digest_for_packaging(self, username: str = "packaging-test-user") -> Digest:
        user = get_user_model().objects.create_user(username=username)
        topic = Topic.objects.create(
            user=user,
            name="Personal Branding",
            keywords=["personal branding"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_PACKAGING,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        return Digest.objects.create(
            run=run,
            title="Digest for Personal Branding",
            payload={
                "title": "Digest for Personal Branding",
                "articles": [
                    {
                        "url": "https://example.com/article",
                        "title": "Personal branding article",
                        "summary": "People trust current evidence of expertise more than polished claims.",
                        "key_points": ["Build in public gives people evidence of current judgment."],
                        "content_type": "opinion",
                        "confidence": 0.9,
                    }
                ],
            },
            quality_score=0.8,
        )

    def _package_payload(self, post_text: str, **extra_fields) -> dict:
        payload = {
            "post_text": post_text,
            "hook_variants": ["Opening one", "Opening two", "Opening three"],
            "cta_variants": ["Closing one", "Closing two", "Closing three"],
            "hashtags": ["#PersonalBranding", "#BuildInPublic", "#Trust"],
            "quality_checks": {
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
                "linkedin_ready": True,
            },
        }
        payload.update(extra_fields)
        return payload

    def _post_brief_payload(self, **overrides) -> dict:
        payload = {
            "target_reader": "Founders building visible expertise",
            "reader_pain_or_mistake": "They polish positioning before proving judgment.",
            "hook_type": "reader_pain",
            "sharp_claim": "A useful personal brand is evidence of current judgment.",
            "credibility_basis": (
                "Grounded in article summaries about build in public and current evidence of expertise."
            ),
            "tension": "Visibility helps only when people can see what to trust you with.",
            "pattern_interrupt": "Visibility without judgment creates attention without trust.",
            "evidence_points": [
                "Build in public gives people evidence of current judgment.",
                "People trust current evidence of expertise more than polished claims.",
            ],
            "concrete_details": [
                "Build in public gives people evidence of current judgment.",
                "People trust current evidence of expertise more than polished claims.",
            ],
            "human_angle": "A practitioner noticing when brand activity lacks decision evidence.",
            "practical_takeaway": "Audit whether recent posts show decisions, not just activity.",
            "ending_reframe": "A brand is a repeated signal of what problems you can solve.",
            "suggested_hook_direction": "Lead with the trust gap, not logo polish.",
            "avoid_angle": "Avoid generic advice about authentic storytelling.",
        }
        payload.update(overrides)
        return payload

    def _author_profile(self) -> dict:
        return {
            "role": "Operations strategist",
            "background": "Leads editorial workflow redesign.",
            "focus": "handoffs, validation, and repeatable systems",
            "voice": "sharp and practical",
            "style_constraints": [
                "avoid generic AI phrasing",
                "make the tension explicit",
                "end with a practical takeaway",
            ],
        }

    def _brief_generation_result(self) -> tuple[dict, str, str, dict]:
        post_brief = self._post_brief_payload()
        return (
            post_brief,
            "brief prompt",
            json.dumps(post_brief),
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    def _editorial_review_payload(self, **overrides) -> dict:
        payload = {
            "passed": False,
            "score": 6,
            "issues": ["too_generic", "weak_hook"],
            "strengths": ["Uses one grounded detail."],
            "repair_instructions": ["Make the opening more specific and practitioner-led."],
        }
        payload.update(overrides)
        return payload

    def test_extract_banned_phrases_from_repair_reasons(self) -> None:
        phrases = _extract_banned_phrases_from_repair_reasons(
            [
                "banned_phrase:resonate",
                "brief_alignment:missing_concrete_detail",
                "banned_phrase:landscape",
                "banned_phrase:resonate",
                "post_mechanics:generic_opening",
            ]
        )

        self.assertEqual(phrases, ["resonate", "landscape"])

    def test_collect_repairable_payload_issues_flags_overlength_post_text(self) -> None:
        issues = _collect_repairable_payload_issues(self._package_payload("x" * 1301))

        self.assertEqual(issues, ["post_text_too_long"])

    def test_collect_repairable_payload_issues_ignores_valid_length_post_text(self) -> None:
        issues = _collect_repairable_payload_issues(self._package_payload("x" * 1300))

        self.assertEqual(issues, [])

    def test_collect_repairable_payload_issues_ignores_missing_post_text(self) -> None:
        payload = self._package_payload("A useful post has text.")
        payload.pop("post_text")

        issues = _collect_repairable_payload_issues(payload)

        self.assertEqual(issues, [])

    def test_repair_rewrite_delta_fails_exact_same_post_text(self) -> None:
        payload = self._package_payload(
            "A useful personal brand proves judgment before polish. "
            "Build in public gives people evidence of current judgment."
        )

        report = _evaluate_repair_rewrite_delta(payload, payload, ["vague_language_density"])

        self.assertFalse(report["passed"])
        self.assertIn("repair_text_too_similar", report["issues"])
        self.assertIn("shared_sentence_ratio", report["signals"])
        self.assertIn("shared_bigram_ratio", report["signals"])
        self.assertIn("weak_word_count", report["signals"])
        self.assertIn("repaired_word_count", report["signals"])

    def test_repair_rewrite_delta_fails_when_most_sentences_are_shared(self) -> None:
        weak_payload = self._package_payload(
            "A useful personal brand proves judgment before polish. "
            "Build in public gives people evidence of current judgment. "
            "If your posts only describe expertise, they are asking for trust."
        )
        repaired_payload = self._package_payload(
            "A useful personal brand proves judgment before polish. "
            "Build in public gives people evidence of current judgment. "
            "Check whether your latest posts show decisions, not just claims."
        )

        report = _evaluate_repair_rewrite_delta(weak_payload, repaired_payload, ["long_paragraph"])

        self.assertFalse(report["passed"])
        self.assertIn("repair_text_too_similar", report["issues"])
        self.assertGreaterEqual(report["signals"]["shared_sentence_ratio"], 0.6)

    def test_repair_rewrite_delta_passes_for_material_rewrite(self) -> None:
        weak_payload = self._package_payload(
            "In the landscape of personal branding, your message should resonate with your audience. "
            "Authentic storytelling is essential for professional growth and meaningful engagement."
        )
        repaired_payload = self._package_payload(
            "A personal brand breaks when people cannot see current judgment.\n\n"
            "Build in public works only when it shows decisions, tradeoffs, and lessons.\n\n"
            "Check your last post: does it prove what people can trust you with now?"
        )

        report = _evaluate_repair_rewrite_delta(weak_payload, repaired_payload, ["banned_phrase:resonate"])

        self.assertTrue(report["passed"])
        self.assertEqual(report["issues"], [])

    def test_repair_rewrite_delta_fails_missing_repair_text(self) -> None:
        report = _evaluate_repair_rewrite_delta(
            self._package_payload("A useful post has text."),
            self._package_payload(""),
            ["post_mechanics:missing_post_text"],
        )

        self.assertFalse(report["passed"])
        self.assertIn("missing_repair_text", report["issues"])

    def test_repair_rewrite_delta_does_not_false_fail_short_overlap(self) -> None:
        report = _evaluate_repair_rewrite_delta(
            self._package_payload("Brand Lag hurts."),
            self._package_payload("Brand Lag shows the gap."),
            ["brief_alignment:missing_concrete_detail"],
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["issues"], [])

    def test_post_brief_alignment_passes_when_post_uses_concrete_detail(self) -> None:
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. That matters because polish alone is weaker proof.\n\n"
            "A brand is a repeated signal of what problems you can solve."
        )

        report = _evaluate_post_brief_alignment(payload, self._post_brief_payload())

        self.assertTrue(report["checked"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["issues"], [])
        self.assertEqual(
            report["details"]["concrete_detail_match"]["match_type"],
            "exact_phrase",
        )

    def test_concrete_detail_match_detects_number_overlap(self) -> None:
        match = _find_concrete_detail_match(
            "The useful signal is simple: 89% of professionals value aligned narratives.",
            ["89% of professionals see value in aligning personal narratives with corporate missions."],
        )

        self.assertEqual(
            match,
            {
                "matched": True,
                "matched_detail": "89% of professionals see value in aligning personal narratives with corporate missions.",
                "matched_fragment": "89%",
                "match_type": "number_overlap",
            },
        )

    def test_concrete_detail_match_detects_light_paraphrase(self) -> None:
        match = _find_concrete_detail_match(
            "Brand Lag is eliminated when contemporary expertise is reflected in the public signal.",
            ["Effective strategies eliminate Brand Lag by reflecting contemporary expertise."],
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["match_type"], "meaningful_fragment")
        self.assertEqual(
            match["matched_detail"],
            "Effective strategies eliminate Brand Lag by reflecting contemporary expertise.",
        )

    def test_concrete_detail_match_ignores_generic_topic_language(self) -> None:
        match = _find_concrete_detail_match(
            "A useful personal brand shows current judgment and clear positioning.",
            ["Effective strategies eliminate Brand Lag by reflecting contemporary expertise."],
        )

        self.assertIsNone(match)

    def test_concrete_detail_match_ignores_unrelated_detail(self) -> None:
        match = _find_concrete_detail_match(
            "The post mentions a logo redesign and a clearer profile photo.",
            ["Effective strategies eliminate Brand Lag by reflecting contemporary expertise."],
        )

        self.assertIsNone(match)

    def test_post_brief_alignment_fails_when_post_text_includes_url(self) -> None:
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. Read more at https://example.com."
        )

        report = _evaluate_post_brief_alignment(payload, self._post_brief_payload())

        self.assertFalse(report["passed"])
        self.assertIn("url_in_post_text", report["issues"])

    def test_post_brief_alignment_fails_when_post_text_contains_cta_phrase(self) -> None:
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. What do you think?"
        )

        report = _evaluate_post_brief_alignment(payload, self._post_brief_payload())

        self.assertFalse(report["passed"])
        self.assertIn("post_text_ends_with_question", report["issues"])
        self.assertIn("cta_phrase_in_post_text:what do you think?", report["issues"])

    def test_post_brief_alignment_does_not_flag_normal_follow_wording_as_cta(self) -> None:
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. Follow the workflow from claim to proof."
        )

        report = _evaluate_post_brief_alignment(payload, self._post_brief_payload())

        self.assertTrue(report["passed"])
        self.assertNotIn("cta_phrase_in_post_text:follow", report["issues"])

    def test_post_brief_alignment_flags_precise_follow_cta(self) -> None:
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. Follow me for more."
        )

        report = _evaluate_post_brief_alignment(payload, self._post_brief_payload())

        self.assertFalse(report["passed"])
        self.assertIn("cta_phrase_in_post_text:follow me", report["issues"])

    def test_post_brief_alignment_fails_when_concrete_details_are_missing(self) -> None:
        payload = self._package_payload(
            "The post stays on angle, but it avoids the specific proof from the brief.\n\n"
            "It talks around the idea instead of using the grounded detail."
        )

        report = _evaluate_post_brief_alignment(payload, self._post_brief_payload())

        self.assertFalse(report["passed"])
        self.assertIn("missing_concrete_detail", report["issues"])
        self.assertNotIn("concrete_detail_match", report["details"])

    def test_post_brief_alignment_fails_when_avoid_angle_is_echoed(self) -> None:
        post_brief = self._post_brief_payload(
            avoid_angle="Avoid generic advice about authentic storytelling.",
            concrete_details=["Build in public gives people evidence of current judgment."],
        )
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. Authentic storytelling is still treated as the answer."
        )

        report = _evaluate_post_brief_alignment(payload, post_brief)

        self.assertFalse(report["passed"])
        self.assertIn("avoid_angle_in_post_text", report["issues"])
        self.assertEqual(
            report["details"]["avoid_angle_match"]["matched_fragment"],
            "authentic storytelling",
        )
        self.assertEqual(
            report["details"]["avoid_angle_match"]["match_type"],
            "meaningful_fragment",
        )

    def test_post_brief_alignment_fails_when_specific_avoid_angle_is_repeated(self) -> None:
        post_brief = self._post_brief_payload(
            avoid_angle="personal branding is about looking polished",
            concrete_details=["Build in public gives people evidence of current judgment."],
        )
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. "
            "The weak version says personal branding is about looking polished."
        )

        report = _evaluate_post_brief_alignment(payload, post_brief)

        self.assertFalse(report["passed"])
        self.assertIn("avoid_angle_in_post_text", report["issues"])
        self.assertEqual(
            report["details"]["avoid_angle_match"],
            {
                "matched": True,
                "matched_fragment": "personal branding is about looking polished",
                "match_type": "exact_phrase",
            },
        )

    def test_post_brief_alignment_does_not_treat_topic_words_as_avoid_angle_drift(self) -> None:
        post_brief = self._post_brief_payload(
            avoid_angle="Avoid broad generalizations about personal branding without acknowledging its nuances.",
            concrete_details=["Effective strategies eliminate Brand Lag by reflecting contemporary expertise."],
        )
        payload = self._package_payload(
            "A useful personal brand reflects current expertise.\n\n"
            "Effective strategies eliminate Brand Lag by reflecting contemporary expertise. Personal branding still needs a narrow signal."
        )

        report = _evaluate_post_brief_alignment(payload, post_brief)

        self.assertTrue(report["passed"])
        self.assertNotIn("avoid_angle_in_post_text", report["issues"])
        self.assertNotIn("avoid_angle_match", report["details"])

    def test_post_brief_alignment_does_not_match_grounded_automation_wording(self) -> None:
        post_brief = self._post_brief_payload(
            avoid_angle="Avoid generic automation strategy advice.",
            concrete_details=["Workflow automation reduces manual review when validation is explicit."],
        )
        payload = self._package_payload(
            "A useful workflow system breaks when validation is unclear.\n\n"
            "Workflow automation reduces manual review when validation is explicit."
        )

        report = _evaluate_post_brief_alignment(payload, post_brief)

        self.assertTrue(report["passed"])
        self.assertNotIn("avoid_angle_in_post_text", report["issues"])
        self.assertNotIn("avoid_angle_match", report["details"])

    def test_avoid_angle_match_detects_specific_meaningful_fragment(self) -> None:
        match = _find_avoid_angle_match(
            "The lazy answer is to build a personal brand by posting more often.",
            "build a personal brand by posting more often",
        )

        self.assertEqual(
            match,
            {
                "matched": True,
                "matched_fragment": "build a personal brand by posting more often",
                "match_type": "exact_phrase",
            },
        )

    def test_avoid_angle_match_ignores_broad_topic_only_angle(self) -> None:
        match = _find_avoid_angle_match(
            "A useful personal brand shows current judgment.",
            "generic personal branding advice",
        )

        self.assertIsNone(match)

    def test_post_brief_alignment_skips_when_post_brief_is_missing(self) -> None:
        payload = self._package_payload("A useful personal brand is evidence of current judgment.")

        report = _evaluate_post_brief_alignment(payload, None)

        self.assertFalse(report["checked"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["reason"], "missing_post_brief")

    def test_linkedin_post_mechanics_passes_for_mechanically_strong_post(self) -> None:
        payload = self._package_payload(
            "A useful personal brand breaks when proof is missing.\n\n"
            "Build in public gives people evidence of current judgment, but polish alone only asks for trust.\n\n"
            "Check whether your last post shows a decision, not just a claim."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertTrue(report["checked"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["issues"], [])
        self.assertIn("first_line_word_count", report["signals"])
        self.assertTrue(report["signals"]["has_pattern_interrupt_signal"])
        self.assertGreater(report["signals"]["concrete_detail_count"], 0)
        self.assertGreaterEqual(report["signals"]["generic_language_count"], 0)

    def test_linkedin_post_mechanics_fails_on_url_in_post_text(self) -> None:
        payload = self._package_payload(
            "A useful personal brand breaks when proof is missing.\n\n"
            "Build in public gives people evidence of current judgment, but read more at www.example.com."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertFalse(report["passed"])
        self.assertIn("url_in_post_text", report["issues"])

    def test_linkedin_post_mechanics_fails_on_cta_in_post_text(self) -> None:
        payload = self._package_payload(
            "A useful personal brand breaks when proof is missing.\n\n"
            "Build in public gives people evidence of current judgment, but comment below with your view."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertFalse(report["passed"])
        self.assertIn("cta_in_post_text", report["issues"])

    def test_linkedin_post_mechanics_fails_on_generic_opening(self) -> None:
        payload = self._package_payload(
            "In today's world, personal branding matters more than ever.\n\n"
            "Build in public gives people evidence of current judgment, but proof still matters."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertFalse(report["passed"])
        self.assertIn("generic_opening", report["issues"])

    def test_linkedin_post_mechanics_fails_when_post_text_ends_with_question(self) -> None:
        payload = self._package_payload(
            "A useful personal brand breaks when proof is missing.\n\n"
            "Build in public gives people evidence of current judgment, but will your audience see it?"
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertFalse(report["passed"])
        self.assertIn("post_text_ends_with_question", report["issues"])

    def test_linkedin_post_mechanics_warns_when_first_line_is_too_long(self) -> None:
        payload = self._package_payload(
            "A useful personal brand breaks when people see polished claims without enough current evidence of judgment in public today.\n\n"
            "Build in public gives people evidence of current judgment, but polish alone only asks for trust."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertTrue(report["passed"])
        self.assertIn("hook_may_be_too_long", report["warnings"])

    def test_linkedin_post_mechanics_warns_when_first_line_is_too_short(self) -> None:
        payload = self._package_payload(
            "Proof matters.\n\n"
            "Build in public gives people evidence of current judgment, but polish alone only asks for trust."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertTrue(report["passed"])
        self.assertIn("hook_may_be_too_short", report["warnings"])

    def test_linkedin_post_mechanics_warns_when_pattern_interrupt_is_missing(self) -> None:
        payload = self._package_payload(
            "A useful personal brand breaks when proof is missing.\n\n"
            "Build in public gives people evidence of current judgment. Decisions make trust easier to evaluate."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertTrue(report["passed"])
        self.assertIn("missing_pattern_interrupt_signal", report["warnings"])

    def test_linkedin_post_mechanics_warns_when_specificity_is_low(self) -> None:
        payload = self._package_payload(
            "A useful personal brand breaks when proof is missing.\n\n"
            "The point is clear, but the writing remains broad and general."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload(concrete_details=[]))

        self.assertTrue(report["passed"])
        self.assertIn("low_specificity", report["warnings"])

    def test_linkedin_post_mechanics_warns_when_ending_is_weak(self) -> None:
        payload = self._package_payload(
            "A useful personal brand breaks when proof is missing.\n\n"
            "Build in public gives people evidence of current judgment, but polish alone only asks for trust. Start today."
        )

        report = _evaluate_linkedin_post_mechanics(payload, self._post_brief_payload())

        self.assertTrue(report["passed"])
        self.assertIn("weak_ending", report["warnings"])

    def test_validate_post_brief_payload_accepts_valid_brief(self) -> None:
        payload = self._post_brief_payload()

        normalized = _validate_post_brief_payload(payload)

        self.assertEqual(normalized, payload)

    def test_validate_post_brief_payload_strips_surrounding_whitespace(self) -> None:
        payload = self._post_brief_payload(
            target_reader="  Founders building visible expertise  ",
            hook_type="  reader_pain  ",
            credibility_basis="  Grounded in article summaries.  ",
            pattern_interrupt="  Visibility without judgment creates attention without trust.  ",
            evidence_points=[
                "  Build in public gives people evidence.  ",
                "  Polished claims are weaker than proof.  ",
            ],
            concrete_details=[
                "  Current evidence of expertise.  ",
                "  ",
                "  Build in public as proof.  ",
            ],
            human_angle="  A practitioner observation.  ",
        )

        normalized = _validate_post_brief_payload(payload)

        self.assertEqual(normalized["target_reader"], "Founders building visible expertise")
        self.assertEqual(normalized["hook_type"], "reader_pain")
        self.assertEqual(normalized["credibility_basis"], "Grounded in article summaries.")
        self.assertEqual(
            normalized["pattern_interrupt"],
            "Visibility without judgment creates attention without trust.",
        )
        self.assertEqual(
            normalized["evidence_points"],
            [
                "Build in public gives people evidence.",
                "Polished claims are weaker than proof.",
            ],
        )
        self.assertEqual(
            normalized["concrete_details"],
            ["Current evidence of expertise.", "Build in public as proof."],
        )
        self.assertEqual(normalized["human_angle"], "A practitioner observation.")

    def test_validate_post_brief_payload_missing_required_field_fails(self) -> None:
        payload = self._post_brief_payload()
        payload.pop("sharp_claim")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_empty_string_field_fails(self) -> None:
        payload = self._post_brief_payload(sharp_claim="   ")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_missing_hook_type_fails(self) -> None:
        payload = self._post_brief_payload()
        payload.pop("hook_type")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_invalid_hook_type_fails(self) -> None:
        payload = self._post_brief_payload(hook_type="generic_advice")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_missing_credibility_basis_fails(self) -> None:
        payload = self._post_brief_payload()
        payload.pop("credibility_basis")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_missing_pattern_interrupt_fails(self) -> None:
        payload = self._post_brief_payload()
        payload.pop("pattern_interrupt")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_missing_human_angle_fails(self) -> None:
        payload = self._post_brief_payload()
        payload.pop("human_angle")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_missing_evidence_points_fails(self) -> None:
        payload = self._post_brief_payload()
        payload.pop("evidence_points")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_non_list_evidence_points_fails(self) -> None:
        payload = self._post_brief_payload(evidence_points="One point. Another point.")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_requires_two_non_empty_evidence_points(self) -> None:
        payload = self._post_brief_payload(evidence_points=["One grounded point.", "   ", 123])

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_trims_evidence_points_to_four(self) -> None:
        payload = self._post_brief_payload(
            evidence_points=[
                "Point one.",
                "Point two.",
                "Point three.",
                "Point four.",
                "Point five.",
            ],
        )

        normalized = _validate_post_brief_payload(payload)

        self.assertEqual(
            normalized["evidence_points"],
            ["Point one.", "Point two.", "Point three.", "Point four."],
        )

    def test_validate_post_brief_payload_missing_concrete_details_fails(self) -> None:
        payload = self._post_brief_payload()
        payload.pop("concrete_details")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_non_list_concrete_details_fails(self) -> None:
        payload = self._post_brief_payload(concrete_details="One detail. Another detail.")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_allows_empty_concrete_details(self) -> None:
        payload = self._post_brief_payload(concrete_details=["   ", 123, None])

        normalized = _validate_post_brief_payload(payload)

        self.assertEqual(normalized["concrete_details"], [])

    def test_validate_post_brief_payload_trims_concrete_details_to_six(self) -> None:
        payload = self._post_brief_payload(
            concrete_details=[
                "Detail one.",
                "Detail two.",
                "Detail three.",
                "Detail four.",
                "Detail five.",
                "Detail six.",
                "Detail seven.",
            ],
        )

        normalized = _validate_post_brief_payload(payload)

        self.assertEqual(
            normalized["concrete_details"],
            [
                "Detail one.",
                "Detail two.",
                "Detail three.",
                "Detail four.",
                "Detail five.",
                "Detail six.",
            ],
        )

    def test_validate_post_brief_payload_strips_extra_keys(self) -> None:
        payload = self._post_brief_payload(extra_key="remove me", metadata={"debug": True})

        normalized = _validate_post_brief_payload(payload)

        self.assertNotIn("extra_key", normalized)
        self.assertNotIn("metadata", normalized)
        self.assertEqual(
            list(normalized.keys()),
            [
                "target_reader",
                "reader_pain_or_mistake",
                "hook_type",
                "sharp_claim",
                "credibility_basis",
                "tension",
                "pattern_interrupt",
                "evidence_points",
                "concrete_details",
                "human_angle",
                "practical_takeaway",
                "ending_reframe",
                "suggested_hook_direction",
                "avoid_angle",
            ],
        )

    def test_validate_editorial_review_payload_accepts_valid_review(self) -> None:
        review = _validate_editorial_review_payload(
            self._editorial_review_payload(
                issues=["too_generic", "unknown_issue", "weak_hook"],
                strengths=[" Uses one grounded detail. ", ""],
                repair_instructions=[" Replace generic advice. ", ""],
                extra_field="ignored",
            )
        )

        self.assertEqual(
            review,
            {
                "passed": False,
                "score": 6,
                "issues": ["too_generic", "weak_hook"],
                "strengths": ["Uses one grounded detail."],
                "repair_instructions": ["Replace generic advice."],
            },
        )
        self.assertNotIn("extra_field", review)

    def test_validate_editorial_review_payload_invalid_score_fails(self) -> None:
        with self.assertRaises(ContentPackageValidationError):
            _validate_editorial_review_payload(self._editorial_review_payload(score=11))

    def test_validate_editorial_review_payload_missing_required_field_fails(self) -> None:
        payload = self._editorial_review_payload()
        payload.pop("strengths")

        with self.assertRaises(ContentPackageValidationError):
            _validate_editorial_review_payload(payload)

    def test_validate_editorial_review_payload_non_bool_passed_fails(self) -> None:
        with self.assertRaises(ContentPackageValidationError):
            _validate_editorial_review_payload(self._editorial_review_payload(passed="false"))

    def test_validate_editorial_review_payload_non_list_fields_fail(self) -> None:
        for field_name in ["issues", "strengths", "repair_instructions"]:
            payload = self._editorial_review_payload()
            payload[field_name] = "not a list"

            with self.subTest(field_name=field_name):
                with self.assertRaises(ContentPackageValidationError):
                    _validate_editorial_review_payload(payload)

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_post_brief_via_llm_returns_normalized_brief(self, mock_openai_client) -> None:
        digest = self._create_digest_for_packaging("brief-llm-user")
        response_payload = self._post_brief_payload(
            target_reader="  Founders building visible expertise  ",
            evidence_points=[
                "  Build in public gives people evidence.  ",
                "Polished claims are weaker than proof.",
                "Extra evidence is kept.",
                "Fourth evidence point is kept.",
                "Fifth evidence point is trimmed.",
            ],
            extra_key="remove me",
        )
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(response_payload),
            usage={"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
        )

        post_brief, _prompt, _response_text, _usage = _generate_post_brief_via_llm(
            digest,
            digest.get_articles(),
            self._author_profile(),
        )

        self.assertEqual(post_brief["target_reader"], "Founders building visible expertise")
        self.assertEqual(len(post_brief["evidence_points"]), 4)
        self.assertNotIn("extra_key", post_brief)

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_post_brief_via_llm_returns_prompt_and_raw_response_text(self, mock_openai_client) -> None:
        digest = self._create_digest_for_packaging("brief-prompt-response-user")
        response_text = json.dumps(self._post_brief_payload())
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=response_text,
            usage=None,
        )

        _post_brief, prompt, raw_response_text, _usage = _generate_post_brief_via_llm(
            digest,
            digest.get_articles(),
            self._author_profile(),
        )

        self.assertIn("Digest for Personal Branding", prompt)
        self.assertIn("People trust current evidence of expertise more than polished claims.", prompt)
        self.assertEqual(raw_response_text, response_text)

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_post_brief_via_llm_returns_token_usage(self, mock_openai_client) -> None:
        digest = self._create_digest_for_packaging("brief-token-user")
        usage = {"prompt_tokens": 21, "completion_tokens": 43, "total_tokens": 64}
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(self._post_brief_payload()),
            usage=usage,
        )

        _post_brief, _prompt, _response_text, returned_usage = _generate_post_brief_via_llm(
            digest,
            digest.get_articles(),
            self._author_profile(),
        )

        self.assertEqual(returned_usage, usage)

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_post_brief_via_llm_invalid_json_fails(self, mock_openai_client) -> None:
        digest = self._create_digest_for_packaging("brief-invalid-json-user")
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="not json",
            usage=None,
        )

        with self.assertRaises(ContentPackageValidationError):
            _generate_post_brief_via_llm(digest, digest.get_articles(), self._author_profile())

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_post_brief_via_llm_structurally_invalid_brief_fails(self, mock_openai_client) -> None:
        digest = self._create_digest_for_packaging("brief-invalid-shape-user")
        invalid_payload = self._post_brief_payload(evidence_points=["Only one point."])
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(invalid_payload),
            usage=None,
        )

        with self.assertRaises(ContentPackageValidationError):
            _generate_post_brief_via_llm(digest, digest.get_articles(), self._author_profile())

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_post_brief_via_llm_prompt_includes_article_evidence_and_author_profile(
        self,
        mock_openai_client,
    ) -> None:
        digest = self._create_digest_for_packaging("brief-author-evidence-user")
        author_profile = self._author_profile()
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(self._post_brief_payload()),
            usage=None,
        )

        _generate_post_brief_via_llm(digest, digest.get_articles(), author_profile)

        prompt = mock_openai_client.return_value.generate_text.call_args.kwargs["prompt"]
        self.assertIn("Operations strategist", prompt)
        self.assertIn("Leads editorial workflow redesign.", prompt)
        self.assertIn("People trust current evidence of expertise more than polished claims.", prompt)
        self.assertIn("Build in public gives people evidence of current judgment.", prompt)
        self.assertTrue(mock_openai_client.return_value.generate_text.call_args.kwargs["json_mode"])

    @patch("services.packaging.generator._generate_payload_via_llm")
    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_post_brief_via_llm_does_not_call_final_post_generation(
        self,
        mock_openai_client,
        mock_generate_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("brief-no-final-post-user")
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(self._post_brief_payload()),
            usage=None,
        )

        _generate_post_brief_via_llm(digest, digest.get_articles(), self._author_profile())

        mock_generate_payload.assert_not_called()
        mock_openai_client.return_value.generate_text.assert_called_once()

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_editorial_review_via_llm_returns_normalized_review(self, mock_openai_client) -> None:
        digest = self._create_digest_for_packaging("editorial-review-llm-user")
        response_payload = self._editorial_review_payload(
            score=7.5,
            issues=["too_generic", "unknown_issue"],
            strengths=[" Uses a concrete detail. "],
            repair_instructions=[" Make the hook sharper. "],
            extra_key="remove me",
        )
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(response_payload),
            usage={"prompt_tokens": 12, "completion_tokens": 13, "total_tokens": 25},
        )

        review, _prompt, _response_text, _usage = _generate_editorial_review_via_llm(
            digest,
            self._package_payload("A useful personal brand is evidence of current judgment."),
            self._author_profile(),
            self._post_brief_payload(),
            {"status": "pass"},
            {"passed": True},
            {"passed": True},
            repair_delta={"passed": True},
            repair_reasons=[],
        )

        self.assertEqual(review["score"], 7.5)
        self.assertEqual(review["issues"], ["too_generic"])
        self.assertEqual(review["strengths"], ["Uses a concrete detail."])
        self.assertNotIn("extra_key", review)

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_editorial_review_via_llm_invalid_json_fails(self, mock_openai_client) -> None:
        digest = self._create_digest_for_packaging("editorial-review-invalid-json-user")
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="not json",
            usage=None,
        )

        with self.assertRaises(ContentPackageValidationError):
            _generate_editorial_review_via_llm(
                digest,
                self._package_payload("A useful personal brand is evidence of current judgment."),
                self._author_profile(),
                self._post_brief_payload(),
                {"status": "pass"},
                {"passed": True},
                {"passed": True},
            )

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_editorial_review_via_llm_structurally_invalid_review_fails(
        self,
        mock_openai_client,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-invalid-shape-user")
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(self._editorial_review_payload(score=0)),
            usage=None,
        )

        with self.assertRaises(ContentPackageValidationError):
            _generate_editorial_review_via_llm(
                digest,
                self._package_payload("A useful personal brand is evidence of current judgment."),
                self._author_profile(),
                self._post_brief_payload(),
                {"status": "pass"},
                {"passed": True},
                {"passed": True},
            )

    @patch("services.packaging.generator.OpenAIClient")
    def test_generate_editorial_review_via_llm_prompt_includes_deterministic_context(
        self,
        mock_openai_client,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-context-user")
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=json.dumps(self._editorial_review_payload()),
            usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        )

        _review, prompt, response_text, usage = _generate_editorial_review_via_llm(
            digest,
            self._package_payload("A useful personal brand is evidence of current judgment."),
            self._author_profile(),
            self._post_brief_payload(),
            {"status": "retry", "reasons": ["vague_language_density"]},
            {"passed": True, "details": {"concrete_detail_match": {"matched": True}}},
            {"passed": True, "warnings": ["hook_may_be_too_long"]},
            repair_delta={"passed": True},
            repair_reasons=["vague_language_density"],
        )

        self.assertIn("Digest for Personal Branding", prompt)
        self.assertIn("A useful personal brand is evidence of current judgment.", prompt)
        self.assertIn("vague_language_density", prompt)
        self.assertIn("hook_may_be_too_long", prompt)
        self.assertEqual(response_text, json.dumps(self._editorial_review_payload()))
        self.assertEqual(usage, {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11})
        self.assertTrue(mock_openai_client.return_value.generate_text.call_args.kwargs["json_mode"])

    def test_normalize_linkedin_hashtags_prefixes_trailing_keyword_line(self) -> None:
        post_text = (
            "Share your personal branding strategies in the comments!\n\n"
            "PersonalBranding Authority Storytelling BrandLag VisualIdentity"
        )

        normalized = normalize_linkedin_hashtags(post_text)

        self.assertEqual(
            normalized,
            "Share your personal branding strategies in the comments!\n\n"
            "#PersonalBranding #Authority #Storytelling #BrandLag #VisualIdentity",
        )

    def test_normalize_linkedin_hashtags_prefixes_comma_separated_keywords(self) -> None:
        post_text = "Share your thoughts.\n\nPersonalBranding, Authority, Storytelling"

        normalized = normalize_linkedin_hashtags(post_text)

        self.assertEqual(
            normalized,
            "Share your thoughts.\n\n#PersonalBranding #Authority #Storytelling",
        )

    def test_normalize_linkedin_hashtags_preserves_existing_tags_and_deduplicates(self) -> None:
        post_text = "Share your thoughts.\n\n#PersonalBranding Authority Storytelling #Authority"

        normalized = normalize_linkedin_hashtags(post_text)

        self.assertEqual(
            normalized,
            "Share your thoughts.\n\n#PersonalBranding #Authority #Storytelling",
        )

    def test_normalize_linkedin_hashtags_leaves_normal_prose_unchanged(self) -> None:
        post_text = "This is a normal final sentence about personal branding."

        normalized = normalize_linkedin_hashtags(post_text)

        self.assertEqual(normalized, post_text)

    def test_packaging_uses_digest_get_articles_not_legacy_digest_fields(self) -> None:
        user = get_user_model().objects.create_user(username="packaging-user")
        topic = Topic.objects.create(
            user=user,
            name="Workflow systems",
            keywords=["workflow"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_GENERATING_DIGEST,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Workflow systems",
            payload={
                "title": "Digest for Workflow systems",
                "articles": [
                    {
                        "url": "https://example.com/article-1",
                        "title": "Workflow redesign article",
                        "summary": "The article argues that workflow redesign matters before adding AI.",
                        "key_points": [
                            "Review time dropped after teams changed the handoff.",
                            "The model helped only after validation became clearer.",
                        ],
                        "content_type": "opinion",
                        "confidence": 0.9,
                    }
                ],
            },
            quality_score=0.0,
        )

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(digest.has_articles())
        self.assertIn("workflow redesign matters", content_package.post_text.lower())
        self.assertEqual(debug_info["provider"], "mock")

    def test_packaging_uses_digest_helper_as_article_source(self) -> None:
        user = get_user_model().objects.create_user(username="packaging-helper-user")
        topic = Topic.objects.create(
            user=user,
            name="Helper topic",
            keywords=["helper"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_GENERATING_DIGEST,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Helper topic",
            payload={"version": 1, "title": "Digest for Helper topic", "articles": []},
            quality_score=0.0,
        )

        helper_articles = [
            {
                "url": "https://example.com/helper-article",
                "title": "Helper article title",
                "summary": "The article argues that workflow fixes come before AI.",
                "key_points": ["The team changed the workflow before the model step."],
                "content_type": "opinion",
                "confidence": 0.8,
            }
        ]

        with patch.object(Digest, "get_articles", return_value=helper_articles):
            content_package, _debug_info = generate_content_package_for_digest(digest)

        self.assertIn("workflow fixes come before ai", content_package.post_text.lower())

    def test_packaging_returns_safe_fallback_when_articles_are_missing(self) -> None:
        user = get_user_model().objects.create_user(username="packaging-fallback-user")
        topic = Topic.objects.create(
            user=user,
            name="Sparse topic",
            keywords=["sparse"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_GENERATING_DIGEST,
            metrics={"digest_stage": {"status": "completed", "articles_count": 0}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Sparse topic",
            payload={"title": "Digest for Sparse topic", "articles": []},
            quality_score=0.0,
        )

        content_package, _debug_info = generate_content_package_for_digest(digest)

        self.assertFalse(digest.has_articles())
        self.assertIn("No post draft articles were available.", content_package.post_text)

    @patch("services.packaging.generator._generate_packaging_payload")
    def test_packaging_saves_normalized_linkedin_hashtags_in_post_text_and_hashtag_list(
        self,
        mock_generate_packaging_payload,
    ) -> None:
        user = get_user_model().objects.create_user(username="packaging-normalizer-user")
        topic = Topic.objects.create(
            user=user,
            name="Hashtag normalization",
            keywords=["linkedin"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_PACKAGING,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Hashtag normalization",
            payload={"title": "Digest for Hashtag normalization", "articles": [{"title": "One", "summary": "Two"}]},
            quality_score=0.0,
        )
        mock_generate_packaging_payload.return_value = PackagingGenerationResult(
            prompt="prompt",
            response_text="{}",
            payload={
                "post_text": "Share your thoughts.\n\n#PersonalBranding Authority Storytelling Authority",
                "hook_variants": ["Opening one", "Opening two", "Opening three"],
                "cta_variants": ["Closing one", "Closing two", "Closing three"],
                "hashtags": ["PersonalBranding", "Authority", "Storytelling", "Authority"],
                "carousel_outline": [],
                "quality_checks": {
                    "uses_only_provided_facts": True,
                    "has_clear_point_of_view": True,
                    "linkedin_ready": True,
                },
            },
            provider="mock",
            is_mock=True,
            fallback_reason="",
            tokens=None,
            estimated_cost_usd=None,
        )

        content_package, _debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(
            content_package.post_text,
            "Share your thoughts.\n\n#PersonalBranding #Authority #Storytelling",
        )
        self.assertEqual(
            content_package.hashtags,
            ["#PersonalBranding", "#Authority", "#Storytelling"],
        )

    @patch("services.packaging.generator._generate_packaging_payload")
    def test_packaging_strips_extra_model_keys_before_saving(
        self,
        mock_generate_packaging_payload,
    ) -> None:
        user = get_user_model().objects.create_user(username="packaging-extra-keys-user")
        topic = Topic.objects.create(
            user=user,
            name="Extra key normalization",
            keywords=["linkedin"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_PACKAGING,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Extra key normalization",
            payload={"title": "Digest for Extra key normalization", "articles": [{"title": "One", "summary": "Two"}]},
            quality_score=0.0,
        )
        mock_generate_packaging_payload.return_value = PackagingGenerationResult(
            prompt="prompt",
            response_text="{}",
            payload={
                "post_text": "A useful post body.",
                "hook_variants": ["Opening one", "Opening two", "Opening three"],
                "cta_variants": ["Closing one", "Closing two", "Closing three"],
                "hashtags": ["#Workflow"],
                "quality_checks": {
                    "uses_only_provided_facts": True,
                    "has_clear_point_of_view": True,
                    "linkedin_ready": True,
                },
                "carousel_outline": [{"slide": 1, "title": "Extra", "bullets": ["Drop me"]}],
                "title": "Extra title",
                "metadata": {"source": "model"},
                "slides": [{"title": "Slide"}],
            },
            provider="openai",
            is_mock=False,
            fallback_reason="",
            tokens=None,
            estimated_cost_usd=None,
        )

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, "A useful post body.")
        self.assertEqual(content_package.hook_variants, ["Opening one", "Opening two", "Opening three"])
        self.assertEqual(content_package.cta_variants, ["Closing one", "Closing two", "Closing three"])
        self.assertEqual(content_package.hashtags, ["#Workflow"])
        self.assertEqual(content_package.carousel_outline, [])
        self.assertEqual(debug_info["validation_report"]["carousel_outline_count"], 0)

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_packaging_generates_brief_before_final_post_generation(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("brief-flow-order-user")
        call_order = []

        def brief_side_effect(*_args, **_kwargs):
            call_order.append("brief")
            return self._brief_generation_result()

        def payload_side_effect(*_args, **_kwargs):
            call_order.append("payload")
            return (
                self._package_payload(
                    "A personal brand works when it proves current judgment.\n\n"
                    "Build in public gives people evidence of current judgment. If people can see decisions and tradeoffs, they know what to trust you with."
                ),
                "final prompt",
                "final response",
                None,
            )

        mock_generate_brief.side_effect = brief_side_effect
        mock_generate_payload.side_effect = payload_side_effect

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(call_order, ["brief", "payload"])
        self.assertEqual(debug_info["post_brief"], self._post_brief_payload())
        self.assertEqual(debug_info["post_brief_prompt"], "brief prompt")
        self.assertTrue(debug_info["brief_alignment"]["passed"])
        self.assertTrue(debug_info["post_mechanics"]["passed"])
        mock_repair_payload.assert_not_called()

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_packaging_passes_validated_brief_to_final_post_generation(
        self,
        mock_generate_payload,
        mock_generate_brief,
    ) -> None:
        digest = self._create_digest_for_packaging("brief-passed-user")
        post_brief = self._post_brief_payload()
        mock_generate_brief.return_value = (post_brief, "brief prompt", json.dumps(post_brief), None)
        mock_generate_payload.return_value = (
            self._package_payload(
                "A useful personal brand proves judgment before polish.\n\n"
                "Build in public gives people evidence of current judgment. If people only see claims, they cannot tell what changed in your thinking."
            ),
            "final prompt",
            "final response",
            None,
        )

        _content_package, debug_info = generate_content_package_for_digest(digest)

        mock_generate_payload.assert_called_once()
        self.assertEqual(mock_generate_payload.call_args.kwargs["post_brief"], post_brief)
        self.assertEqual(debug_info["post_brief"], post_brief)
        self.assertEqual(debug_info["post_brief_prompt"], "brief prompt")
        self.assertTrue(debug_info["brief_alignment"]["passed"])
        self.assertIn("post_mechanics", debug_info)
        self.assertTrue(debug_info["post_mechanics"]["passed"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_packaging_keeps_final_tokens_and_reports_brief_tokens_separately(
        self,
        mock_generate_payload,
        mock_generate_brief,
    ) -> None:
        digest = self._create_digest_for_packaging("brief-token-accounting-user")
        post_brief = self._post_brief_payload()
        brief_tokens = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        final_tokens = {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        mock_generate_brief.return_value = (post_brief, "brief prompt", json.dumps(post_brief), brief_tokens)
        mock_generate_payload.return_value = (
            self._package_payload(
                "A useful personal brand proves judgment before polish.\n\n"
                "Build in public gives people evidence of current judgment. If people only see claims, they cannot tell what changed in your thinking."
            ),
            "final prompt",
            "final response",
            final_tokens,
        )

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(debug_info["tokens"], final_tokens)
        self.assertEqual(debug_info["post_brief_tokens"], brief_tokens)
        self.assertTrue(debug_info["brief_alignment"]["passed"])
        self.assertTrue(debug_info["post_mechanics"]["passed"])

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_packaging_debug_info_includes_editorial_review_on_real_path(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-debug-user")
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Visibility without judgment creates attention without trust. Build in public gives people evidence of current judgment.\n\n"
            "Audit whether recent posts show decisions, not just activity."
        )
        review = self._editorial_review_payload(passed=True, score=8, issues=[])
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (payload, "final prompt", "final response", None)
        mock_generate_review.return_value = (
            review,
            "review prompt",
            json.dumps(review),
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, payload["post_text"])
        self.assertEqual(debug_info["provider"], "openai")
        self.assertFalse(debug_info["is_mock"])
        self.assertEqual(debug_info["editorial_review"], review)
        self.assertEqual(debug_info["editorial_review_prompt"], "review prompt")
        self.assertEqual(debug_info["editorial_review_response_text"], json.dumps(review))
        self.assertEqual(
            debug_info["editorial_review_tokens"],
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        self.assertFalse(debug_info["repair_attempted"])
        self.assertFalse(debug_info["editorial_review_used_for_repair"])
        self.assertFalse(debug_info["editorial_review_triggered_repair"])
        self.assertEqual(debug_info["editorial_repair_reasons"], [])
        mock_generate_review.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_editorial_review_failure_triggers_existing_single_repair_path(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-triggers-repair-user")
        initial_payload = self._package_payload(
            "A personal brand breaks when people cannot tell what to trust you with.\n\n"
            "A logo can signal care, but evidence does the heavier work. Build in public gives people evidence of current judgment.\n\n"
            "If your last ten posts only describe expertise, they are asking people to assume trust."
        )
        repair_payload = self._package_payload(
            "A useful personal brand proves current judgment before polish.\n\n"
            "Build in public gives people evidence of current judgment because decisions and tradeoffs are visible.\n\n"
            "Audit the last ten posts: if they show activity but not decisions, they create attention without evidence."
        )
        review = self._editorial_review_payload(
            passed=False,
            score=5,
            issues=["too_generic", "weak_hook"],
            repair_instructions=["Make the rewrite more practitioner-led."],
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (initial_payload, "final prompt", "final response", None)
        mock_generate_review.return_value = (review, "review prompt", json.dumps(review), None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertTrue(debug_info["editorial_review_triggered_repair"])
        self.assertTrue(debug_info["editorial_review_used_for_repair"])
        self.assertIn("editorial_review:failed", debug_info["repair_reasons"])
        self.assertIn("editorial_review:score_below_threshold", debug_info["repair_reasons"])
        self.assertIn("editorial_review:too_generic", debug_info["repair_reasons"])
        self.assertEqual(debug_info["editorial_repair_reasons"], debug_info["repair_reasons"])
        self.assertEqual(mock_repair_payload.call_args.kwargs["editorial_review"], review)
        mock_generate_review.assert_called_once()
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_editorial_review_low_score_triggers_repair_even_when_passed(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-low-score-user")
        initial_payload = self._package_payload(
            "A personal brand breaks when people cannot tell what to trust you with.\n\n"
            "A logo can signal care, but evidence does the heavier work. Build in public gives people evidence of current judgment.\n\n"
            "If your last ten posts only describe expertise, they are asking people to assume trust."
        )
        repair_payload = self._package_payload(
            "A useful personal brand proves current judgment before polish.\n\n"
            "Build in public gives people evidence of current judgment because decisions and tradeoffs are visible.\n\n"
            "Audit the last ten posts: if they show activity but not decisions, they create attention without evidence."
        )
        review = self._editorial_review_payload(passed=True, score=6, issues=["low_reader_value"])
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (initial_payload, "final prompt", "final response", None)
        mock_generate_review.return_value = (review, "review prompt", json.dumps(review), None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["editorial_review_triggered_repair"])
        self.assertIn("editorial_review:score_below_threshold", debug_info["repair_reasons"])
        self.assertIn("editorial_review:low_reader_value", debug_info["repair_reasons"])
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_editorial_review_float_score_below_threshold_triggers_repair(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-float-score-user")
        initial_payload = self._package_payload(
            "A personal brand breaks when people cannot tell what to trust you with.\n\n"
            "A logo can signal care, but evidence does the heavier work. Build in public gives people evidence of current judgment.\n\n"
            "If your last ten posts only describe expertise, they are asking people to assume trust."
        )
        repair_payload = self._package_payload(
            "A useful personal brand proves current judgment before polish.\n\n"
            "Build in public gives people evidence of current judgment because decisions and tradeoffs are visible.\n\n"
            "Audit the last ten posts: if they show activity but not decisions, they create attention without evidence."
        )
        review = self._editorial_review_payload(passed=True, score=6.5, issues=["low_reader_value"])
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (initial_payload, "final prompt", "final response", None)
        mock_generate_review.return_value = (review, "review prompt", json.dumps(review), None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["editorial_review_triggered_repair"])
        self.assertIn("editorial_review:score_below_threshold", debug_info["repair_reasons"])
        self.assertIn("editorial_review:low_reader_value", debug_info["repair_reasons"])
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_editorial_review_failure_does_not_cause_mock_fallback(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-error-user")
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Visibility without judgment creates attention without trust. Build in public gives people evidence of current judgment.\n\n"
            "Audit whether recent posts show decisions, not just activity."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (payload, "final prompt", "final response", None)
        mock_generate_review.side_effect = RuntimeError("review unavailable")

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, payload["post_text"])
        self.assertEqual(debug_info["provider"], "openai")
        self.assertFalse(debug_info["is_mock"])
        self.assertEqual(debug_info["editorial_review"], {})
        self.assertEqual(debug_info["editorial_review_error"], "review unavailable")
        self.assertFalse(debug_info["repair_attempted"])
        self.assertFalse(debug_info["editorial_review_used_for_repair"])
        self.assertFalse(debug_info["editorial_review_triggered_repair"])
        self.assertEqual(debug_info["editorial_repair_reasons"], [])
        mock_generate_review.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_editorial_repair_does_not_add_second_retry(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-single-retry-user")
        initial_payload = self._package_payload(
            "A personal brand breaks when people cannot tell what to trust you with.\n\n"
            "A logo can signal care, but evidence does the heavier work. Build in public gives people evidence of current judgment.\n\n"
            "If your last ten posts only describe expertise, they are asking people to assume trust."
        )
        still_weak_payload = self._package_payload("In the landscape of personal branding, cohesive signals resonate.")
        review = self._editorial_review_payload(passed=False, score=5, issues=["too_generic"])
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (initial_payload, "final prompt", "final response", None)
        mock_generate_review.return_value = (review, "review prompt", json.dumps(review), None)
        mock_repair_payload.return_value = (still_weak_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertTrue(debug_info["editorial_review_triggered_repair"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn("LinkedIn post editorial repair did not produce a valid package", debug_info["fallback_reason"])
        self.assertTrue(content_package.post_text)
        self.assertEqual(mock_repair_payload.call_count, 1)

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_editorial_review_is_generated_before_deterministic_repair_and_passed_to_prompt(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-guides-repair-user")
        weak_payload = self._package_payload(
            "This draft includes a source link instead of a finished insight.\n\n"
            "Read more at https://example.com."
        )
        repair_payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Visibility without judgment creates attention without trust. Build in public gives people evidence of current judgment.\n\n"
            "Audit whether recent posts show decisions, not just activity."
        )
        review = self._editorial_review_payload(
            passed=False,
            score=5,
            issues=["too_generic", "weak_hook"],
            repair_instructions=["Make the rewrite more practitioner-led."],
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "final prompt", "final response", None)
        mock_generate_review.return_value = (
            review,
            "review prompt",
            json.dumps(review),
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertEqual(debug_info["editorial_review"], review)
        self.assertTrue(debug_info["editorial_review_used_for_repair"])
        self.assertFalse(debug_info["editorial_review_triggered_repair"])
        self.assertEqual(debug_info["editorial_repair_reasons"], [])
        self.assertIn("post_mechanics:url_in_post_text", debug_info["repair_reasons"])
        self.assertEqual(mock_repair_payload.call_args.kwargs["editorial_review"], review)
        self.assertEqual(mock_repair_payload.call_args.kwargs["editorial_review_error"], "")
        mock_generate_review.assert_called_once()
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_editorial_review_failure_before_repair_does_not_block_repair(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-repair-error-user")
        weak_payload = self._package_payload(
            "This draft includes a source link instead of a finished insight.\n\n"
            "Read more at https://example.com."
        )
        repair_payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Visibility without judgment creates attention without trust. Build in public gives people evidence of current judgment.\n\n"
            "Audit whether recent posts show decisions, not just activity."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "final prompt", "final response", None)
        mock_generate_review.side_effect = RuntimeError("review unavailable")
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertEqual(debug_info["provider"], "openai")
        self.assertFalse(debug_info["is_mock"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertEqual(debug_info["editorial_review"], {})
        self.assertEqual(debug_info["editorial_review_error"], "review unavailable")
        self.assertFalse(debug_info["editorial_review_used_for_repair"])
        self.assertIsNone(mock_repair_payload.call_args.kwargs["editorial_review"])
        self.assertEqual(mock_repair_payload.call_args.kwargs["editorial_review_error"], "review unavailable")
        mock_generate_review.assert_called_once()
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test", PACKAGING_EDITORIAL_REVIEW_ENABLED=True)
    @patch("services.packaging.generator._generate_editorial_review_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    def test_mock_fallback_path_keeps_empty_editorial_review_fields(
        self,
        mock_generate_brief,
        mock_generate_review,
    ) -> None:
        digest = self._create_digest_for_packaging("editorial-review-fallback-user")
        mock_generate_brief.side_effect = ContentPackageValidationError("brief failed")

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertEqual(debug_info["editorial_review"], {})
        self.assertEqual(debug_info["editorial_review_prompt"], "")
        self.assertEqual(debug_info["editorial_review_response_text"], "")
        self.assertIsNone(debug_info["editorial_review_tokens"])
        self.assertEqual(debug_info["editorial_review_error"], "")
        self.assertFalse(debug_info["editorial_review_used_for_repair"])
        mock_generate_review.assert_not_called()
        self.assertTrue(content_package.post_text)

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_overlength_initial_post_text_triggers_repair_and_saves_real_result(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("overlength-repair-user")
        overlength_text = (
            "A useful personal brand is evidence of current judgment.\n\n"
            + ("Build in public gives people evidence of current judgment, but polished claims only ask for trust. " * 18)
            + "\n\nA brand is a repeated signal of what problems you can solve."
        )
        self.assertGreater(len(overlength_text), 1300)
        repair_text = (
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people a current proof trail: decisions, tradeoffs, and lessons, not polish.\n\n"
            "A brand is a repeated signal of what problems you can solve."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (self._package_payload(overlength_text), "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (self._package_payload(repair_text), "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(debug_info["provider"], "openai")
        self.assertFalse(debug_info["is_mock"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertIn("post_text_too_long", debug_info["repair_reasons"])
        self.assertLessEqual(len(content_package.post_text), 1300)
        self.assertTrue(debug_info["brief_alignment"]["passed"])
        self.assertTrue(debug_info["post_mechanics"]["passed"])
        self.assertTrue(debug_info["repair_delta"]["passed"])
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_overlength_repair_still_overlength_falls_back_with_length_reason(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("overlength-repair-still-long-user")
        overlength_text = (
            "A useful personal brand is evidence of current judgment.\n\n"
            + ("Build in public gives people evidence of current judgment, but polished claims only ask for trust. " * 18)
            + "\n\nA brand is a repeated signal of what problems you can solve."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (self._package_payload(overlength_text), "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (self._package_payload(overlength_text), "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        mock_repair_payload.assert_called_once()
        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn("1300", debug_info["fallback_reason"])
        self.assertTrue(content_package.post_text)

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_brief_alignment_failure_triggers_existing_repair_path(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("brief-alignment-repair-user")
        weak_payload = self._package_payload(
            "A useful personal brand proves judgment before polish.\n\n"
            "The body stays broad and never uses the grounded concrete detail."
        )
        repair_payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. That matters because polished claims are weaker than proof.\n\n"
            "A brand is a repeated signal of what problems you can solve."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertIn("brief_alignment:missing_concrete_detail", debug_info["repair_reasons"])
        self.assertTrue(debug_info["brief_alignment"]["passed"])
        self.assertIn("concrete_detail_match", debug_info["brief_alignment"]["details"])
        self.assertTrue(debug_info["post_mechanics"]["passed"])
        self.assertTrue(debug_info["repair_delta"]["passed"])
        self.assertIn("shared_bigram_ratio", debug_info["repair_delta"]["signals"])
        diagnostics = debug_info["concrete_detail_diagnostics"]
        self.assertEqual(
            diagnostics["required_details"],
            [
                "Build in public gives people evidence of current judgment.",
                "People trust current evidence of expertise more than polished claims.",
            ],
        )
        self.assertTrue(diagnostics["initial_missing"])
        self.assertIsNone(diagnostics["initial_match"])
        self.assertFalse(diagnostics["missing_after_repair"])
        self.assertEqual(diagnostics["repair_match"]["match_type"], "exact_phrase")
        self.assertIn("The body stays broad", diagnostics["post_text_excerpt"])
        self.assertIn("Build in public gives people evidence", diagnostics["repair_text_excerpt"])
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_brief_alignment_still_falls_back_when_repair_misses_concrete_detail(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("brief-alignment-repair-still-missing-user")
        weak_payload = self._package_payload(
            "A useful personal brand proves judgment before polish.\n\n"
            "The body stays broad and never uses the grounded concrete detail."
        )
        repair_payload = self._package_payload(
            "A useful personal brand proves judgment before polish.\n\n"
            "The body still stays broad and avoids the specific proof from the brief."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        mock_repair_payload.assert_called_once()
        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertIn("brief_alignment:missing_concrete_detail", debug_info["fallback_reason"])
        diagnostics = debug_info["concrete_detail_diagnostics"]
        self.assertTrue(diagnostics["initial_missing"])
        self.assertIsNone(diagnostics["initial_match"])
        self.assertIsNone(diagnostics["repair_match"])
        self.assertTrue(diagnostics["missing_after_repair"])
        self.assertIn("The body still stays broad", diagnostics["repair_text_excerpt"])
        self.assertEqual(mock_repair_payload.call_count, 1)
        self.assertTrue(content_package.post_text)

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_near_identical_text_falls_back_with_repair_delta_reason(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("repair-delta-too-similar-user")
        weak_payload = self._package_payload(
            "A useful personal brand proves judgment before polish.\n\n"
            "The body stays broad and never uses the grounded concrete detail."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (weak_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        mock_repair_payload.assert_called_once()
        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertIn("repair_delta:repair_text_too_similar", debug_info["fallback_reason"])
        self.assertEqual(mock_repair_payload.call_count, 1)
        self.assertTrue(content_package.post_text)

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_post_mechanics_hard_issue_triggers_existing_repair_path(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("mechanics-repair-user")
        weak_payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment, but read more at https://example.com."
        )
        repair_payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment, but polished claims only ask for trust.\n\n"
            "Check whether your last post shows a decision, not just a claim."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertIn("post_mechanics:url_in_post_text", debug_info["repair_reasons"])
        self.assertTrue(debug_info["post_mechanics"]["passed"])
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_post_mechanics_generic_opening_triggers_repair_and_saves_specific_opening(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("mechanics-generic-opening-repair-user")
        weak_payload = self._package_payload(
            "Many professionals polish their personal brand before proving current judgment.\n\n"
            "Build in public gives people evidence of current judgment, but the opening is too broad."
        )
        repair_payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Visibility without judgment creates attention without trust. Build in public gives people evidence of current judgment, while polished positioning only asks people to believe you.\n\n"
            "Audit whether recent posts show decisions, not just activity. A brand is a repeated signal of what problems you can solve."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertIn("post_mechanics:generic_opening", debug_info["repair_reasons"])
        self.assertTrue(debug_info["post_mechanics"]["passed"])
        self.assertTrue(debug_info["repair_delta"]["passed"])
        mock_repair_payload.assert_called_once()

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_post_mechanics_generic_opening_still_falls_back_when_repair_keeps_generic_opening(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("mechanics-generic-opening-still-generic-user")
        weak_payload = self._package_payload(
            "Many professionals need a stronger personal brand to stand out.\n\n"
            "Build in public gives people evidence of current judgment, but this opening stays generic."
        )
        repair_payload = self._package_payload(
            "Many professionals polish their personal brand before proving current judgment.\n\n"
            "Visibility without judgment creates attention without trust. Build in public gives people evidence of current judgment, while polished positioning only asks people to believe you.\n\n"
            "Audit whether recent posts show decisions, not just activity. A brand is a repeated signal of what problems you can solve."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        mock_repair_payload.assert_called_once()
        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn("post_mechanics:generic_opening", debug_info["repair_reasons"])
        self.assertIn("post_mechanics:generic_opening", debug_info["fallback_reason"])
        self.assertEqual(mock_repair_payload.call_count, 1)
        self.assertTrue(content_package.post_text)

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_post_mechanics_warnings_do_not_trigger_repair_by_themselves(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("mechanics-warning-user")
        payload = self._package_payload(
            "A useful personal brand is evidence of current judgment.\n\n"
            "Build in public gives people evidence of current judgment. Decisions make trust easier to evaluate."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (payload, "initial prompt", "initial response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, payload["post_text"])
        self.assertFalse(debug_info["repair_attempted"])
        self.assertEqual(debug_info["repair_delta"], {})
        self.assertTrue(debug_info["post_mechanics"]["passed"])
        self.assertIn("missing_pattern_interrupt_signal", debug_info["post_mechanics"]["warnings"])
        mock_repair_payload.assert_not_called()

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_brief_generation_failure_falls_back_without_final_post_generation(
        self,
        mock_generate_payload,
        mock_generate_brief,
    ) -> None:
        digest = self._create_digest_for_packaging("brief-failure-user")
        mock_generate_brief.side_effect = ContentPackageValidationError("Post brief evidence_points must include at least 2 non-empty strings.")

        content_package, debug_info = generate_content_package_for_digest(digest)

        mock_generate_payload.assert_not_called()
        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertIn("LinkedIn post brief generation/validation failed", debug_info["fallback_reason"])
        self.assertIn("Post brief evidence_points", debug_info["fallback_reason"])
        self.assertIsNone(debug_info["post_brief"])
        self.assertEqual(debug_info["post_brief_prompt"], "")
        self.assertTrue(content_package.post_text)

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_weak_valid_post_with_banned_phrase_triggers_repair(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-repair-user")
        weak_payload = self._package_payload(
            "Your personal brand must resonate across every touchpoint.\n\n"
            "That sounds smooth, but it hides the real work."
        )
        repair_payload = self._package_payload(
            "A personal brand breaks when people cannot see your current judgment.\n\n"
            "Polished claims are not enough. Show the decisions, lessons, and tradeoffs that prove what you can be trusted with."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertIn("banned_phrase:resonate", debug_info["repair_reasons"])
        diagnostics = debug_info["banned_phrase_diagnostics"]
        self.assertEqual(diagnostics["banned_phrases"], ["resonate"])
        self.assertEqual(diagnostics["initial_matches"][0]["phrase"], "resonate")
        self.assertEqual(diagnostics["initial_matches"][0]["field"], "post_text")
        self.assertEqual(diagnostics["repair_matches"], [])
        self.assertFalse(diagnostics["regressed_after_repair"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_broad_opening_triggers_repair(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-broad-opening-user")
        weak_payload = self._package_payload(
            "In the landscape of personal branding, professionals need a cohesive presence."
        )
        repair_payload = self._package_payload(
            "Most personal brands fail because trust is too vague.\n\n"
            "People need evidence of current judgment, not another polished positioning line."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn("broad_opening:in the landscape of", debug_info["repair_reasons"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_generic_first_line_triggers_repair(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-generic-opening-user")
        weak_payload = self._package_payload(
            "Authentic storytelling is essential for any personal brand to thrive.\n\n"
            "The post says useful things, but it starts like generic advice."
        )
        repair_payload = self._package_payload(
            "Most personal brands fail when people cannot see your current judgment.\n\n"
            "Look at your last 10 posts. If they show activity but not decisions, people get visibility without evidence."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn(
            "weak_generic_opening:authentic storytelling is essential",
            debug_info["repair_reasons"],
        )

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_vague_language_density_triggers_repair(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-vague-density-user")
        weak_payload = self._package_payload(
            "A personal brand grows through authentic storytelling and audience engagement.\n\n"
            "The journey builds trust, visibility, and narrative development over time."
        )
        repair_payload = self._package_payload(
            "A personal brand weakens when people cannot name your current judgment.\n\n"
            "Check your last 10 posts. If they show updates but not decisions, your signal is too soft."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn("vague_language_density", debug_info["repair_reasons"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_current_generic_advice_sample_triggers_repair(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-current-bad-sample-user")
        weak_payload = self._package_payload(
            "Authentic storytelling is essential for any personal brand to thrive.\n\n"
            "It creates trust, audience engagement, narrative growth, and visibility through the journey."
        )
        repair_payload = self._package_payload(
            "Most personal brands fail when people cannot tell what changed in your judgment.\n\n"
            "Look at your last 10 posts. If they show polish but not tradeoffs, people see activity without proof."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn(
            "weak_generic_opening:authentic storytelling is essential",
            debug_info["repair_reasons"],
        )
        self.assertIn("vague_language_density", debug_info["repair_reasons"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_soft_length_limit_triggers_repair_and_saves_shorter_output(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-length-user")
        weak_text = "A personal brand breaks when trust is unclear. " + ("Evidence matters. " * 65)
        self.assertGreater(len(weak_text), 1200)
        self.assertLessEqual(len(weak_text), 1300)
        repair_text = (
            "A personal brand breaks when trust is unclear.\n\n"
            "Build in public gives people evidence of current judgment. If people only see claims, they cannot tell what your judgment is worth."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (self._package_payload(weak_text), "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (self._package_payload(repair_text), "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_text)
        self.assertLess(len(content_package.post_text), 1150)
        self.assertIn("soft_length_limit", debug_info["repair_reasons"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_strong_valid_post_saves_without_repair(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-pass-user")
        strong_text = (
            "A personal brand breaks when people cannot tell what to trust you with.\n\n"
            "A logo can signal care, but evidence does the heavier work. Build in public gives people evidence of current judgment.\n\n"
            "If your last ten posts only describe expertise, they are not building trust. They are asking for it."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (self._package_payload(strong_text), "initial prompt", "initial response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, strong_text)
        self.assertFalse(debug_info["repair_attempted"])
        self.assertEqual(debug_info["quality_gate"]["status"], "pass")
        diagnostics = debug_info["concrete_detail_diagnostics"]
        self.assertFalse(diagnostics["initial_missing"])
        self.assertEqual(diagnostics["initial_match"]["match_type"], "exact_phrase")
        self.assertIsNone(diagnostics["repair_match"])
        self.assertFalse(diagnostics["missing_after_repair"])
        mock_repair_payload.assert_not_called()

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_output_strips_extra_keys_before_saving(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-extra-key-user")
        weak_payload = self._package_payload("Your brand should resonate with a cohesive audience.")
        repair_payload = self._package_payload(
            "A personal brand fails when evidence is missing.\n\n"
            "Build in public gives people evidence of current judgment. Show the work that proves your judgment is current.",
            carousel_outline=[{"slide": 1, "title": "Extra"}],
            metadata={"source": "repair"},
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.carousel_outline, [])
        self.assertEqual(debug_info["validation_report"]["carousel_outline_count"], 0)
        self.assertTrue(debug_info["repair_succeeded"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_output_with_only_long_paragraph_is_split_and_saved(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-long-repair-user")
        weak_payload = self._package_payload("Your brand should resonate with the right people.")
        repair_text = (
            "A personal brand fails when people cannot see your current judgment. "
            "A logo can help recognition, but it does not prove what you can be trusted with. "
            "Build in public works because it gives people evidence of decisions, lessons, and tradeoffs. "
            "That evidence closes the gap between what people remember about you and what you can do now. "
            "If your content only describes expertise, it is not building trust. "
            "It is asking people to assume it. "
            "That is a weak bet when trust depends on recent proof."
        )
        self.assertGreater(len(repair_text), 450)
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (self._package_payload(repair_text), "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["repair_succeeded"])
        self.assertIn("\n\n", content_package.post_text)
        self.assertEqual(debug_info["repair_quality_gate"]["status"], "pass")

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_happens_at_most_once_and_falls_back_when_still_weak(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-repair-fail-user")
        weak_payload = self._package_payload("Your brand should resonate in a changing landscape.")
        still_weak_payload = self._package_payload("In the landscape of personal branding, cohesive signals resonate.")
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (still_weak_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        mock_repair_payload.assert_called_once()
        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertTrue(content_package.post_text)
        self.assertIn("One article points to:", content_package.post_text)
        self.assertIn("banned_phrase:resonate", debug_info["fallback_reason"])
        diagnostics = debug_info["banned_phrase_diagnostics"]
        self.assertEqual(diagnostics["banned_phrases"], ["resonate", "landscape"])
        self.assertTrue(
            any(match["phrase"] == "resonate" and match["field"] == "post_text" for match in diagnostics["initial_matches"])
        )
        self.assertTrue(
            any(match["phrase"] == "resonate" and match["field"] == "post_text" for match in diagnostics["repair_matches"])
        )
        self.assertTrue(diagnostics["regressed_after_repair"])
        self.assertEqual(mock_repair_payload.call_count, 1)

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_post_brief_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_provider_error_uses_existing_safe_fallback(
        self,
        mock_generate_payload,
        mock_generate_brief,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-repair-error-user")
        weak_payload = self._package_payload("Your brand should resonate in a changing landscape.")
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.side_effect = RuntimeError("repair connection failed")

        content_package, debug_info = generate_content_package_for_digest(digest)

        mock_repair_payload.assert_called_once()
        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertIn("Fallback", debug_info["fallback_reason"])
        self.assertTrue(content_package.post_text)
