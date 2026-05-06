from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.test import SimpleTestCase

from apps.digests.models import Digest, DigestRun
from apps.topics.models import Topic
from services.ai.digest_smoke_test import build_prompt as build_article_prompt
from services.packaging.generator import build_carousel_prompt, build_post_prompt


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
