from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.test import SimpleTestCase

from apps.digests.models import Digest, DigestRun
from apps.topics.models import Topic
from services.ai.digest_smoke_test import build_prompt as build_article_prompt
from services.packaging.generator import build_carousel_prompt, build_post_prompt, build_post_repair_prompt


class PromptUsageTests(SimpleTestCase):
    def test_digest_generation_uses_per_article_prompt(self):
        article = {
            "title": "Article title",
            "source_name": "Example source",
            "url": "https://example.com/article-1",
            "content": "Article content",
        }

        with patch("services.ai.digest_smoke_test.render_prompt", return_value="PROMPT") as mock_render:
            result = build_article_prompt(article)

        self.assertEqual(result, "PROMPT")
        mock_render.assert_called_once()
        self.assertEqual(mock_render.call_args.args[0], "digest/analyze_single_article.txt")

    def test_packaging_uses_article_based_prompt_templates(self):
        topic = Topic(name="Workflow topic")
        run = DigestRun(topic=topic)
        digest = Digest(
            run=run,
            title="Digest for Workflow topic",
            payload={"version": 1, "title": "Digest for Workflow topic", "articles": []},
        )
        articles = [
            {
                "url": "https://example.com/article-1",
                "title": "Prompt article title",
                "summary": "Summary",
                "key_points": ["Point"],
                "content_type": "news",
                "confidence": 0.8,
            }
        ]
        author_profile = {
            "role": "AI Automation Specialist",
            "background": "Builds workflow systems.",
            "focus": "workflow design",
            "voice": "analytical",
            "style_constraints": ["one", "two", "three"],
        }

        with patch("services.packaging.generator.build_prompt", return_value="PROMPT") as mock_build:
            build_post_prompt(digest, articles, author_profile)
            build_carousel_prompt(digest, articles, author_profile)

        used_templates = [call.args[0] for call in mock_build.call_args_list]
        self.assertEqual(
            used_templates,
            [
                "linkedin/generate_post_from_articles.txt",
                "linkedin/generate_carousel_from_articles.txt",
            ],
        )

    def test_deprecated_prompts_live_only_under_deprecated_directory(self):
        prompts_root = Path(settings.BASE_DIR) / "prompts"

        self.assertFalse((prompts_root / "digest" / "generate_digest.txt").exists())
        self.assertFalse((prompts_root / "linkedin" / "generate_post.txt").exists())
        self.assertTrue((prompts_root / "deprecated" / "generate_digest.txt").exists())
        self.assertTrue((prompts_root / "deprecated" / "generate_post.txt").exists())

    def test_active_article_based_prompts_do_not_require_global_digest_inputs(self):
        prompts_root = Path(settings.BASE_DIR) / "prompts"
        post_prompt = (prompts_root / "linkedin" / "generate_post_from_articles.txt").read_text(encoding="utf-8")
        carousel_prompt = (prompts_root / "linkedin" / "generate_carousel_from_articles.txt").read_text(
            encoding="utf-8"
        )

        self.assertIn("{articles}", post_prompt)
        self.assertIn("{articles}", carousel_prompt)
        self.assertNotIn("{summary}", post_prompt)
        self.assertNotIn("{key_points}", post_prompt)
        self.assertNotIn("{sources}", post_prompt)

    def test_linkedin_post_prompt_uses_author_profile_and_anti_recap_constraints(self):
        prompts_root = Path(settings.BASE_DIR) / "prompts"
        post_prompt = (prompts_root / "linkedin" / "generate_post_from_articles.txt").read_text(encoding="utf-8")

        self.assertIn("{author_role}", post_prompt)
        self.assertIn("{author_background}", post_prompt)
        self.assertIn("{author_focus}", post_prompt)
        self.assertIn("{author_voice}", post_prompt)
        self.assertIn("{style_constraint_1}", post_prompt)
        self.assertIn("{style_constraint_2}", post_prompt)
        self.assertIn("{style_constraint_3}", post_prompt)
        self.assertIn("Use source facts as evidence, not as the structure of the post", post_prompt)
        self.assertIn('structure the post as "one article says" or "another article says"', post_prompt)
        self.assertIn("write like a report, digest, or research memo", post_prompt)

    def test_build_post_prompt_renders_author_profile_values_without_unresolved_placeholders_and_length_rules(self):
        topic = Topic(name="Workflow topic")
        run = DigestRun(topic=topic)
        digest = Digest(
            run=run,
            title="Digest for Workflow topic",
            payload={"version": 1, "title": "Digest for Workflow topic", "articles": []},
        )
        articles = [
            {
                "url": "https://example.com/article-1",
                "title": "Prompt article title",
                "summary": "Workflow speed improved after the team fixed handoffs.",
                "key_points": ["Validation got clearer before the automation layer paid off."],
                "content_type": "opinion",
                "confidence": 0.8,
            }
        ]
        author_profile = {
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

        rendered_prompt = build_post_prompt(digest, articles, author_profile)

        self.assertIn("Operations strategist", rendered_prompt)
        self.assertIn("Leads editorial workflow redesign.", rendered_prompt)
        self.assertIn("handoffs, validation, and repeatable systems", rendered_prompt)
        self.assertIn("sharp and practical", rendered_prompt)
        self.assertIn("avoid generic AI phrasing", rendered_prompt)
        self.assertIn("make the tension explicit", rendered_prompt)
        self.assertIn("end with a practical takeaway", rendered_prompt)

        self.assertNotIn("{author_role}", rendered_prompt)
        self.assertNotIn("{author_background}", rendered_prompt)
        self.assertNotIn("{author_focus}", rendered_prompt)
        self.assertNotIn("{author_voice}", rendered_prompt)
        self.assertNotIn("{style_constraint_1}", rendered_prompt)
        self.assertNotIn("{style_constraint_2}", rendered_prompt)
        self.assertNotIn("{style_constraint_3}", rendered_prompt)

        self.assertIn("LinkedIn-native expert post", rendered_prompt)
        self.assertIn("Use source facts as evidence, not as the structure of the post", rendered_prompt)
        self.assertIn('structure the post as "one article says" or "another article says"', rendered_prompt)
        self.assertIn("write like a report, digest, or research memo", rendered_prompt)
        self.assertIn("`post_text` must be no more than 1300 characters", rendered_prompt)
        self.assertIn("Prefer 900-1200 characters for `post_text` to leave validation buffer", rendered_prompt)
        self.assertIn("This limit applies only to `post_text`, not to the full JSON response", rendered_prompt)
        self.assertIn("Write like a practitioner with a point of view, not like a content marketer", rendered_prompt)
        self.assertIn("Sound like someone who has seen this problem in real work", rendered_prompt)
        self.assertIn("Use the author profile as a lens, not as a bio", rendered_prompt)
        self.assertIn("Start `post_text` with a direct, specific claim", rendered_prompt)
        self.assertIn("Include one concrete reader pain, mistake, wrong optimization, missing signal, or cost", rendered_prompt)
        self.assertIn("Convert source facts into one practical interpretation", rendered_prompt)
        self.assertIn('"In the landscape of..."', rendered_prompt)
        self.assertIn('"In today\'s world..."', rendered_prompt)
        self.assertIn('"Many professionals mistakenly..."', rendered_prompt)
        self.assertIn("`post_text` must not end with a question or CTA", rendered_prompt)
        self.assertIn("End the body with a sharp takeaway, practical diagnostic, or memorable reframing", rendered_prompt)
        self.assertIn("CTA questions belong only in `cta_variants`", rendered_prompt)
        self.assertIn("CTA variants should sound natural, not salesy", rendered_prompt)
        self.assertIn('"Let\'s discuss"', rendered_prompt)
        self.assertIn('"Start building today"', rendered_prompt)
        self.assertIn("Return exactly the keys shown in the schema below.", rendered_prompt)
        self.assertIn("Do not return `carousel_outline`.", rendered_prompt)
        self.assertIn("Do not return any extra keys.", rendered_prompt)
        self.assertIn("If tempted to include carousel content, omit it.", rendered_prompt)
        self.assertIn("resonate", rendered_prompt)
        self.assertIn("compelling story", rendered_prompt)
        self.assertIn("authentic self", rendered_prompt)
        self.assertIn("elevate your brand", rendered_prompt)
        self.assertIn("unlock potential", rendered_prompt)
        self.assertIn("systemic alignment", rendered_prompt)
        self.assertIn("holistic", rendered_prompt)
        self.assertIn("leverage", rendered_prompt)
        self.assertIn("landscape", rendered_prompt)

    def test_build_post_repair_prompt_renders_quality_repair_contract(self):
        topic = Topic(name="Personal Branding")
        run = DigestRun(topic=topic)
        digest = Digest(
            run=run,
            title="Digest for Personal Branding",
            payload={"version": 1, "title": "Digest for Personal Branding", "articles": []},
        )
        articles = [
            {
                "url": "https://example.com/article-1",
                "title": "Prompt article title",
                "summary": "Build in public gives people current evidence of expertise.",
                "key_points": ["Brand lag appears when reputation trails current work."],
                "content_type": "opinion",
                "confidence": 0.8,
            }
        ]
        author_profile = {
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
        weak_payload = {
            "post_text": "In the landscape of personal branding, your message should resonate.",
            "hook_variants": ["One", "Two", "Three"],
            "cta_variants": ["One", "Two", "Three"],
            "hashtags": ["#PersonalBranding"],
            "quality_checks": {
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
                "linkedin_ready": True,
            },
        }
        quality_report = {"status": "retry", "reasons": ["banned_phrase:resonate"]}

        rendered_prompt = build_post_repair_prompt(digest, articles, author_profile, weak_payload, quality_report)

        self.assertTrue((Path(settings.BASE_DIR) / "prompts" / "linkedin" / "repair_post_quality.txt").exists())
        self.assertIn("Return exactly this JSON shape", rendered_prompt)
        self.assertIn('"post_text": "string"', rendered_prompt)
        self.assertIn("The repaired payload must remove every retry reason listed above", rendered_prompt)
        self.assertIn("`post_text` must be under 1150 characters", rendered_prompt)
        self.assertIn("banned_phrase:resonate", rendered_prompt)
        self.assertIn("Do not use any phrase from the AVOID list anywhere in `post_text`", rendered_prompt)
        self.assertIn("Avoid vague abstract terms unless the source fact requires the exact word", rendered_prompt)
        self.assertIn("Do not write general advice", rendered_prompt)
        self.assertIn("Replace abstract claims with a concrete diagnostic", rendered_prompt)
        self.assertIn("Include one practical test the reader can apply immediately", rendered_prompt)
        self.assertIn("Include one concrete mistake, missing signal, or cost", rendered_prompt)
        self.assertIn("Write as a practitioner pointing out a pattern, not as a content marketer", rendered_prompt)
        self.assertIn("Do not start with an \"authentic storytelling is essential\" style opening", rendered_prompt)
        self.assertIn("End with a sharp diagnostic or reframing", rendered_prompt)
        self.assertIn("Look at your last 10 posts", rendered_prompt)
        self.assertIn("storytelling", rendered_prompt)
        self.assertIn("visibility", rendered_prompt)
        self.assertIn("narrative", rendered_prompt)
        self.assertIn("trust", rendered_prompt)
        self.assertIn("audience", rendered_prompt)
        self.assertIn("polished outcomes", rendered_prompt)
        self.assertIn("Do not return extra keys", rendered_prompt)
        self.assertIn("Do not include `carousel_outline`", rendered_prompt)
        self.assertIn("Keep factual claims grounded in the provided source facts", rendered_prompt)
        self.assertIn("End `post_text` with an insight, diagnostic, or practical takeaway, not a question", rendered_prompt)
        self.assertIn("replace the idea with plainer wording instead of reusing the phrase", rendered_prompt)
        self.assertIn("In the landscape of personal branding", rendered_prompt)
