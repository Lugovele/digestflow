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
    _generate_post_brief_via_llm,
    _validate_post_brief_payload,
    normalize_linkedin_hashtags,
)
from services.packaging.validators import ContentPackageValidationError


@override_settings(OPENAI_API_KEY="sk-your-key")
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
            "sharp_claim": "A useful personal brand is evidence of current judgment.",
            "tension": "Visibility helps only when people can see what to trust you with.",
            "evidence_points": [
                "Build in public gives people evidence of current judgment.",
                "People trust current evidence of expertise more than polished claims.",
            ],
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

    def test_validate_post_brief_payload_accepts_valid_brief(self) -> None:
        payload = self._post_brief_payload()

        normalized = _validate_post_brief_payload(payload)

        self.assertEqual(normalized, payload)

    def test_validate_post_brief_payload_strips_surrounding_whitespace(self) -> None:
        payload = self._post_brief_payload(
            target_reader="  Founders building visible expertise  ",
            evidence_points=[
                "  Build in public gives people evidence.  ",
                "  Polished claims are weaker than proof.  ",
            ],
        )

        normalized = _validate_post_brief_payload(payload)

        self.assertEqual(normalized["target_reader"], "Founders building visible expertise")
        self.assertEqual(
            normalized["evidence_points"],
            [
                "Build in public gives people evidence.",
                "Polished claims are weaker than proof.",
            ],
        )

    def test_validate_post_brief_payload_missing_required_field_fails(self) -> None:
        payload = self._post_brief_payload()
        payload.pop("sharp_claim")

        with self.assertRaises(ContentPackageValidationError):
            _validate_post_brief_payload(payload)

    def test_validate_post_brief_payload_empty_string_field_fails(self) -> None:
        payload = self._post_brief_payload(sharp_claim="   ")

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
                "sharp_claim",
                "tension",
                "evidence_points",
                "practical_takeaway",
                "ending_reframe",
                "suggested_hook_direction",
                "avoid_angle",
            ],
        )

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
                    "If people can see decisions and tradeoffs, they know what to trust you with."
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
                "If people only see claims, they cannot tell what changed in your thinking."
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
                "If people only see claims, they cannot tell what changed in your thinking."
            ),
            "final prompt",
            "final response",
            final_tokens,
        )

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(debug_info["tokens"], final_tokens)
        self.assertEqual(debug_info["post_brief_tokens"], brief_tokens)

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
            "If people only see claims, they cannot tell what your judgment is worth. Show the work that proves it."
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
            "A logo can signal care, but evidence does the heavier work. Build in public gives people a current record of your judgment.\n\n"
            "If your last ten posts only describe expertise, they are not building trust. They are asking for it."
        )
        mock_generate_brief.return_value = self._brief_generation_result()
        mock_generate_payload.return_value = (self._package_payload(strong_text), "initial prompt", "initial response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, strong_text)
        self.assertFalse(debug_info["repair_attempted"])
        self.assertEqual(debug_info["quality_gate"]["status"], "pass")
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
            "Show the work that proves your judgment is current.",
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
