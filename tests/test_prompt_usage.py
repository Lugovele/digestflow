from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.test import SimpleTestCase

from apps.digests.models import Digest, DigestRun
from apps.topics.models import Topic
from services.ai.digest_smoke_test import build_prompt as build_article_prompt
from services.packaging.generator import (
    build_carousel_prompt,
    build_post_brief_prompt,
    build_post_prompt,
    build_post_repair_prompt,
)


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
            build_post_brief_prompt(digest, articles, author_profile)
            build_post_prompt(digest, articles, author_profile)
            build_carousel_prompt(digest, articles, author_profile)

        used_templates = [call.args[0] for call in mock_build.call_args_list]
        self.assertEqual(
            used_templates,
            [
                "linkedin/generate_post_brief_from_articles.txt",
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

    def test_linkedin_post_brief_prompt_declares_editorial_contract(self):
        prompts_root = Path(settings.BASE_DIR) / "prompts"
        brief_prompt_path = prompts_root / "linkedin" / "generate_post_brief_from_articles.txt"

        self.assertTrue(brief_prompt_path.exists())

        brief_prompt = brief_prompt_path.read_text(encoding="utf-8")

        for field_name in [
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
        ]:
            self.assertIn(f'"{field_name}"', brief_prompt)

        self.assertIn("The brief is an editorial LinkedIn-quality brief, not a summary.", brief_prompt)
        self.assertIn("Do not write the final post.", brief_prompt)
        self.assertIn("Do not write final post prose.", brief_prompt)
        self.assertIn("Choose one angle only.", brief_prompt)
        self.assertIn(
            "Choose exactly one `hook_type` from: `personal_action`, `reader_pain`, `counterintuitive_fact`.",
            brief_prompt,
        )
        self.assertIn("Design the hook direction for the first 8-12 words of the final post.", brief_prompt)
        self.assertIn("Include a pattern interrupt for the first third of the final post.", brief_prompt)
        self.assertIn("Use source facts as evidence.", brief_prompt)
        self.assertIn("Evidence points must be grounded in article summaries/key_points.", brief_prompt)
        self.assertIn("Prefer 2-4 concise evidence points.", brief_prompt)
        self.assertIn("`credibility_basis` must explain what the claim is based on", brief_prompt)
        self.assertIn("Extract `concrete_details` only when grounded in article evidence or the author profile.", brief_prompt)
        self.assertIn("Do not invent numbers, names, cases, personal experiences, results, or metrics.", brief_prompt)
        self.assertIn("`human_angle` must be non-fabricated", brief_prompt)
        self.assertIn("`avoid_angle` must explicitly name the generic angle to avoid.", brief_prompt)
        self.assertIn("human expert LinkedIn post", brief_prompt)

    def test_build_post_brief_prompt_renders_author_profile_and_article_evidence_without_placeholders(self):
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

        rendered_prompt = build_post_brief_prompt(digest, articles, author_profile)

        self.assertIn("Operations strategist", rendered_prompt)
        self.assertIn("Leads editorial workflow redesign.", rendered_prompt)
        self.assertIn("handoffs, validation, and repeatable systems", rendered_prompt)
        self.assertIn("sharp and practical", rendered_prompt)
        self.assertIn("avoid generic AI phrasing", rendered_prompt)
        self.assertIn("make the tension explicit", rendered_prompt)
        self.assertIn("end with a practical takeaway", rendered_prompt)
        self.assertIn("Workflow speed improved after the team fixed handoffs.", rendered_prompt)
        self.assertIn("Validation got clearer before the automation layer paid off.", rendered_prompt)
        self.assertIn("Prompt article title", rendered_prompt)

        for placeholder in [
            "{author_role}",
            "{author_background}",
            "{author_focus}",
            "{author_voice}",
            "{style_constraint_1}",
            "{style_constraint_2}",
            "{style_constraint_3}",
            "{articles}",
            "{topic_name}",
            "{digest_title}",
        ]:
            self.assertNotIn(placeholder, rendered_prompt)

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
        self.assertIn(
            "Use `hook_type`, `pattern_interrupt`, `credibility_basis`, `concrete_details`, and `human_angle` from the post brief",
            rendered_prompt,
        )
        self.assertIn(
            "The first line must follow the brief's `hook_type` and reflect `sharp_claim` or `tension`",
            rendered_prompt,
        )
        self.assertIn("Use `pattern_interrupt` in the first third of `post_text`", rendered_prompt)
        self.assertIn("Use `credibility_basis` to decide how strongly the claim can be stated", rendered_prompt)
        self.assertIn("Use `concrete_details` only when they are present in the brief", rendered_prompt)
        self.assertIn("Use `human_angle` as the tone lens without inventing personal experience", rendered_prompt)
        self.assertIn("BRIEF ALIGNMENT RULES", rendered_prompt)
        self.assertIn("The final post must be a transformation of the brief, not a new interpretation.", rendered_prompt)
        self.assertIn("First line must follow `hook_type` and reflect `sharp_claim` or `tension`.", rendered_prompt)
        self.assertIn("The first third of `post_text` must include `pattern_interrupt`.", rendered_prompt)
        self.assertIn("Use at least one `concrete_details` item if available.", rendered_prompt)
        self.assertIn("Use `evidence_points` as the evidence backbone.", rendered_prompt)
        self.assertIn("Do not use the `avoid_angle`.", rendered_prompt)
        self.assertIn("Do not introduce a broader topic than the brief.", rendered_prompt)
        self.assertIn("Start `post_text` with a direct, specific claim", rendered_prompt)
        self.assertIn("Include one concrete reader pain, mistake, wrong optimization, missing signal, or cost", rendered_prompt)
        self.assertIn("Convert source facts into one practical interpretation", rendered_prompt)
        self.assertIn('"In the landscape of..."', rendered_prompt)
        self.assertIn('"In today\'s world..."', rendered_prompt)
        self.assertIn('"Many professionals mistakenly..."', rendered_prompt)
        self.assertIn("`post_text` must not end with a question or CTA", rendered_prompt)
        self.assertIn("End the body with a sharp takeaway, practical diagnostic, or memorable reframing", rendered_prompt)
        self.assertIn("CTA questions belong only in `cta_variants`", rendered_prompt)
        self.assertIn("Do not put CTA questions inside `post_text`", rendered_prompt)
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
        self.assertIn("Make claims only when the articles support them", rendered_prompt)
        self.assertIn("Do not fabricate numbers, names, examples, or conclusions not grounded in the inputs", rendered_prompt)
        self.assertIn("Do not invent personal experience", rendered_prompt)
        self.assertIn("Do not invent numbers or cases", rendered_prompt)
        self.assertIn("Do not include URLs in `post_text`", rendered_prompt)

    def test_build_post_prompt_includes_post_brief_and_angle_constraints(self):
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
        post_brief = {
            "target_reader": "Operations leaders",
            "reader_pain_or_mistake": "They automate before the handoff is clear.",
            "hook_type": "reader_pain",
            "sharp_claim": "Speed exposes unclear workflow ownership.",
            "credibility_basis": "Grounded in article summaries about handoff clarity and validation.",
            "tension": "Automation helps only after validation is explicit.",
            "pattern_interrupt": "Faster systems make unclear ownership more visible.",
            "evidence_points": [
                "Workflow speed improved after the team fixed handoffs.",
                "Validation got clearer before the automation layer paid off.",
            ],
            "concrete_details": [
                "Workflow speed improved after handoffs changed.",
                "Validation got clearer before automation helped.",
            ],
            "human_angle": "A practitioner noticing the ownership gap before tool adoption.",
            "practical_takeaway": "Check the handoff before adding automation.",
            "ending_reframe": "The useful system is the one that makes ownership visible.",
            "suggested_hook_direction": "Lead with the ownership gap.",
            "avoid_angle": "Avoid generic AI productivity advice.",
        }

        rendered_prompt = build_post_prompt(digest, articles, author_profile, post_brief=post_brief)

        self.assertIn("Post brief:", rendered_prompt)
        self.assertIn("Operations leaders", rendered_prompt)
        self.assertIn("reader_pain", rendered_prompt)
        self.assertIn("Speed exposes unclear workflow ownership.", rendered_prompt)
        self.assertIn("Grounded in article summaries about handoff clarity and validation.", rendered_prompt)
        self.assertIn("Faster systems make unclear ownership more visible.", rendered_prompt)
        self.assertIn("Workflow speed improved after handoffs changed.", rendered_prompt)
        self.assertIn("A practitioner noticing the ownership gap before tool adoption.", rendered_prompt)
        self.assertIn("Avoid generic AI productivity advice.", rendered_prompt)
        self.assertIn("Use the post brief as the editorial direction", rendered_prompt)
        self.assertIn(
            "Use `hook_type`, `pattern_interrupt`, `credibility_basis`, `concrete_details`, and `human_angle` from the post brief",
            rendered_prompt,
        )
        self.assertIn("Do not choose a new angle", rendered_prompt)
        self.assertIn("Do not broaden beyond the brief", rendered_prompt)
        self.assertIn("Source articles are grounding material, not permission to expand into a broad essay", rendered_prompt)
        self.assertNotIn("{post_brief}", rendered_prompt)

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
        post_brief = {
            "target_reader": "Founders building visible expertise",
            "reader_pain_or_mistake": "They polish positioning before proving judgment.",
            "hook_type": "reader_pain",
            "sharp_claim": "A useful personal brand is evidence of current judgment.",
            "credibility_basis": "Grounded in article summaries about build in public.",
            "tension": "Visibility helps only when people can see what to trust you with.",
            "pattern_interrupt": "Visibility without judgment creates attention without trust.",
            "evidence_points": ["Build in public gives people current evidence of expertise."],
            "concrete_details": ["Build in public gives people current evidence of expertise."],
            "human_angle": "A practitioner noticing weak proof signals.",
            "practical_takeaway": "Audit whether recent posts show decisions.",
            "ending_reframe": "A brand is a repeated signal of what problems you can solve.",
            "suggested_hook_direction": "Lead with the trust gap.",
            "avoid_angle": "Avoid generic advice about authentic storytelling.",
        }

        rendered_prompt = build_post_repair_prompt(
            digest,
            articles,
            author_profile,
            weak_payload,
            quality_report,
            post_brief=post_brief,
        )

        self.assertTrue((Path(settings.BASE_DIR) / "prompts" / "linkedin" / "repair_post_quality.txt").exists())
        self.assertIn("Return exactly this JSON shape", rendered_prompt)
        self.assertIn('"post_text": "string"', rendered_prompt)
        self.assertIn("The repaired payload must remove every retry reason listed above", rendered_prompt)
        self.assertIn("Blocked phrases from repair reasons:", rendered_prompt)
        self.assertIn("Remove every exact phrase named in the repair reasons", rendered_prompt)
        self.assertIn(
            "If a reason is `banned_phrase:<phrase>`, the repaired `post_text`, `hook_variants`, "
            "`cta_variants`, and `hashtags` must not contain `<phrase>`",
            rendered_prompt,
        )
        self.assertIn("Do not replace a banned phrase with another banned or generic phrase", rendered_prompt)
        self.assertIn("Prefer specific operational language instead of vague substitutes", rendered_prompt)
        self.assertIn("Validated post brief:", rendered_prompt)
        self.assertIn("A useful personal brand is evidence of current judgment.", rendered_prompt)
        self.assertIn("Preserve the validated post brief", rendered_prompt)
        self.assertIn("Do not choose a new angle", rendered_prompt)
        self.assertIn("Use at least one `concrete_details` item from the brief if present", rendered_prompt)
        self.assertIn("Use the brief's `evidence_points` as the evidence backbone", rendered_prompt)
        self.assertIn("Use `human_angle` as the tone lens without inventing personal experience", rendered_prompt)
        self.assertIn("Remove `avoid_angle` drift from `post_text`", rendered_prompt)
        self.assertIn("Remove URLs and CTA questions from `post_text`", rendered_prompt)
        self.assertIn("Keep CTA questions only in `cta_variants`", rendered_prompt)
        self.assertIn("Remove CTA phrases from `post_text`", rendered_prompt)
        self.assertIn("Replace generic openings with a specific first line", rendered_prompt)
        self.assertIn("Keep the first line compact", rendered_prompt)
        self.assertIn("Include a concrete detail when available", rendered_prompt)
        self.assertIn("`post_text` must be under 1150 characters", rendered_prompt)
        self.assertIn("banned_phrase:resonate", rendered_prompt)
        self.assertIn('"resonate"', rendered_prompt)
        self.assertIn("Do not use any phrase from the AVOID list anywhere in `post_text`", rendered_prompt)
        self.assertIn("elevate", rendered_prompt)
        self.assertIn("leverage", rendered_prompt)
        self.assertIn("holistic", rendered_prompt)
        self.assertIn("seamless", rendered_prompt)
        self.assertIn("landscape", rendered_prompt)
        self.assertIn("unlock", rendered_prompt)
        self.assertIn("potential", rendered_prompt)
        self.assertIn("authentic", rendered_prompt)
        self.assertIn("powerful", rendered_prompt)
        self.assertIn("meaningful", rendered_prompt)
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
