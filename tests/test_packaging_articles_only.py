from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.topics.models import Topic
from apps.digests.models import Digest, DigestRun
from services.packaging import generate_content_package_for_digest
from services.packaging.generator import PackagingGenerationResult, normalize_linkedin_hashtags


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
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_weak_valid_post_with_banned_phrase_triggers_repair(
        self,
        mock_generate_payload,
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
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_payload["post_text"])
        self.assertTrue(debug_info["repair_attempted"])
        self.assertTrue(debug_info["repair_succeeded"])
        self.assertIn("banned_phrase:resonate", debug_info["repair_reasons"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_broad_opening_triggers_repair(
        self,
        mock_generate_payload,
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
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn("broad_opening:in the landscape of", debug_info["repair_reasons"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_generic_first_line_triggers_repair(
        self,
        mock_generate_payload,
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
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_vague_language_density_triggers_repair(
        self,
        mock_generate_payload,
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
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        _content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["repair_attempted"])
        self.assertIn("vague_language_density", debug_info["repair_reasons"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_current_generic_advice_sample_triggers_repair(
        self,
        mock_generate_payload,
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
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_soft_length_limit_triggers_repair_and_saves_shorter_output(
        self,
        mock_generate_payload,
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
        mock_generate_payload.return_value = (self._package_payload(weak_text), "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (self._package_payload(repair_text), "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, repair_text)
        self.assertLess(len(content_package.post_text), 1150)
        self.assertIn("soft_length_limit", debug_info["repair_reasons"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_strong_valid_post_saves_without_repair(
        self,
        mock_generate_payload,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-pass-user")
        strong_text = (
            "A personal brand breaks when people cannot tell what to trust you with.\n\n"
            "A logo can signal care, but evidence does the heavier work. Build in public gives people a current record of your judgment.\n\n"
            "If your last ten posts only describe expertise, they are not building trust. They are asking for it."
        )
        mock_generate_payload.return_value = (self._package_payload(strong_text), "initial prompt", "initial response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, strong_text)
        self.assertFalse(debug_info["repair_attempted"])
        self.assertEqual(debug_info["quality_gate"]["status"], "pass")
        mock_repair_payload.assert_not_called()

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_output_strips_extra_keys_before_saving(
        self,
        mock_generate_payload,
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
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (repair_payload, "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.carousel_outline, [])
        self.assertEqual(debug_info["validation_report"]["carousel_outline_count"], 0)
        self.assertTrue(debug_info["repair_succeeded"])

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_output_with_only_long_paragraph_is_split_and_saved(
        self,
        mock_generate_payload,
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
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.return_value = (self._package_payload(repair_text), "repair prompt", "repair response", None)

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(debug_info["repair_succeeded"])
        self.assertIn("\n\n", content_package.post_text)
        self.assertEqual(debug_info["repair_quality_gate"]["status"], "pass")

    @override_settings(OPENAI_API_KEY="sk-test")
    @patch("services.packaging.generator._repair_packaging_payload_via_llm")
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_happens_at_most_once_and_falls_back_when_still_weak(
        self,
        mock_generate_payload,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-repair-fail-user")
        weak_payload = self._package_payload("Your brand should resonate in a changing landscape.")
        still_weak_payload = self._package_payload("In the landscape of personal branding, cohesive signals resonate.")
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
    @patch("services.packaging.generator._generate_payload_via_llm")
    def test_repair_provider_error_uses_existing_safe_fallback(
        self,
        mock_generate_payload,
        mock_repair_payload,
    ) -> None:
        digest = self._create_digest_for_packaging("quality-repair-error-user")
        weak_payload = self._package_payload("Your brand should resonate in a changing landscape.")
        mock_generate_payload.return_value = (weak_payload, "initial prompt", "initial response", None)
        mock_repair_payload.side_effect = RuntimeError("repair connection failed")

        content_package, debug_info = generate_content_package_for_digest(digest)

        mock_repair_payload.assert_called_once()
        self.assertEqual(debug_info["provider"], "mock")
        self.assertTrue(debug_info["is_mock"])
        self.assertIn("Fallback", debug_info["fallback_reason"])
        self.assertTrue(content_package.post_text)
