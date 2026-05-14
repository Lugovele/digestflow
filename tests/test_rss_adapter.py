import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bs4 import BeautifulSoup
from django.test import SimpleTestCase

from services.sources.rss_adapter import (
    _remove_boilerplate_nodes,
    _fetch_feed_content,
    _extract_html_content_diagnostics,
    build_dev_to_api_url,
    detect_source_type,
    fetch_dev_to_article_content,
    fetch_generic_web_article,
    fetch_rss_articles,
    get_rss_debug_snapshot,
    normalize_source_url,
)


class RSSAdapterTests(SimpleTestCase):
    def test_normalize_source_url_detects_dev_to_tag_and_builds_internal_api_url(self):
        normalized = normalize_source_url("https://dev.to/t/ai")

        self.assertEqual(normalized.platform, "dev.to")
        self.assertEqual(normalized.source_type, "devto_tag")
        self.assertEqual(normalized.detection_reason, "matched dev.to topic pattern")
        self.assertEqual(normalized.metadata["tag"], "ai")
        self.assertEqual(normalized.original_url, "https://dev.to/t/ai")
        self.assertEqual(normalized.normalized_url, "https://dev.to/api/articles?tag=ai")
        self.assertEqual(detect_source_type("https://dev.to/t/ai"), "devto_tag")
        self.assertEqual(build_dev_to_api_url("ai"), "https://dev.to/api/articles?tag=ai")

    def test_normalize_source_url_accepts_dev_to_tag_variants_and_author_profiles(self):
        tag_variant = normalize_source_url(" https://DEV.to//t/ai/?ref=top#section ")
        author_variant = normalize_source_url("https://dev.to/michael_rakutko/?ref=foo#about")

        self.assertEqual(tag_variant.original_url, "https://dev.to/t/ai")
        self.assertEqual(tag_variant.normalized_url, "https://dev.to/api/articles?tag=ai")
        self.assertEqual(tag_variant.source_type, "devto_tag")
        self.assertEqual(tag_variant.platform, "dev.to")

        self.assertEqual(author_variant.original_url, "https://dev.to/michael_rakutko")
        self.assertEqual(author_variant.normalized_url, "https://dev.to/feed/michael_rakutko")
        self.assertEqual(author_variant.source_type, "devto_author")
        self.assertEqual(author_variant.platform, "dev.to")
        self.assertEqual(author_variant.metadata["author"], "michael_rakutko")
        self.assertEqual(author_variant.detection_reason, "matched dev.to author profile")

    def test_normalize_source_url_preserves_meaningful_query_params_for_generic_article_pages(self):
        normalized = normalize_source_url(
            "https://www.stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237"
        )

        self.assertEqual(
            normalized.original_url,
            "https://stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237",
        )
        self.assertEqual(
            normalized.normalized_url,
            "https://stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237",
        )
        self.assertEqual(normalized.source_type, "generic_html")

    def test_normalize_source_url_removes_tracking_query_params_but_keeps_meaningful_ones(self):
        normalized = normalize_source_url(
            "https://example.com/article?id=123&utm_source=newsletter&utm_medium=email&fbclid=tracking"
        )

        self.assertEqual(normalized.original_url, "https://example.com/article?id=123")
        self.assertEqual(normalized.normalized_url, "https://example.com/article?id=123")

    def test_fetch_dev_to_tag_page_uses_internal_api_and_returns_real_article_urls(self):
        article_list = [
            {
                "id": 101,
                "title": "AI workflow article",
                "url": "https://dev.to/alice/ai-workflow-article",
                "description": "Short description",
                "tag_list": ["ai"],
                "published_at": "2026-05-05T10:00:00Z",
            }
        ]
        full_content = "Full article content " * 20

        with patch(
            "services.sources.rss_adapter.fetch_dev_to_article_list",
            return_value=article_list,
        ), patch(
            "services.sources.rss_adapter.fetch_dev_to_article_content",
            return_value={
                "title": "AI workflow article",
                "url": "https://dev.to/alice/ai-workflow-article",
                "description": "Short description",
                "content": full_content,
                "published_at": "2026-05-05T10:00:00Z",
                "metadata": {"reading_time_minutes": 4},
            },
        ):
            items = fetch_rss_articles("https://dev.to/t/ai")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://dev.to/alice/ai-workflow-article")
        self.assertEqual(items[0]["title"], "AI workflow article")
        self.assertEqual(items[0]["source_url"], "https://dev.to/t/ai")
        self.assertEqual(items[0]["source_api_url"], "https://dev.to/api/articles?tag=ai")
        self.assertEqual(items[0]["content"], full_content.strip())
        self.assertEqual(items[0]["metadata"]["source_type"], "devto_tag")
        self.assertEqual(items[0]["metadata"]["detection_reason"], "matched dev.to topic pattern")
        self.assertFalse(items[0]["metadata"]["content_unavailable"])

    def test_fetch_dev_to_tag_page_marks_articles_without_full_content(self):
        article_list = [
            {
                "id": 101,
                "title": "Thin article",
                "url": "https://dev.to/alice/thin-article",
                "description": "Short description",
            }
        ]

        with patch(
            "services.sources.rss_adapter.fetch_dev_to_article_list",
            return_value=article_list,
        ), patch(
            "services.sources.rss_adapter.fetch_dev_to_article_content",
            return_value=None,
        ):
            items = fetch_rss_articles("https://dev.to/t/ai")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://dev.to/alice/thin-article")
        self.assertEqual(items[0]["content"], "")
        self.assertTrue(items[0]["metadata"]["content_unavailable"])

    def test_direct_dev_to_article_url_fetches_single_article_content(self):
        full_article_text = (
            "A workflow team first mapped every approval step before adding automation. "
            "That redesign removed duplicate checks, clarified owners, and reduced review loops. "
            "Only after the process was simplified did the AI layer improve draft speed without creating extra cleanup work."
        )
        html = """
        <html>
          <head><title>Real dev.to article</title></head>
          <body><article><p>{full_article_text}</p></article></body>
        </html>
        """.format(full_article_text=full_article_text)

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=html):
            items = fetch_rss_articles("https://dev.to/alice/real-devto-article")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://dev.to/alice/real-devto-article")
        self.assertEqual(items[0]["title"], "Real dev.to article")
        self.assertIn("A workflow team first mapped every approval step before adding automation.", items[0]["content"])
        self.assertEqual(items[0]["source_url"], "https://dev.to/alice/real-devto-article")
        self.assertIsNone(items[0]["source_api_url"])
        self.assertEqual(items[0]["metadata"]["extraction_method"], "article_tag")
        self.assertGreater(items[0]["metadata"]["extracted_content_length"], 0)
        self.assertIsNone(items[0]["metadata"]["extraction_warning"])

    def test_fetch_generic_web_article_accepts_readable_article_page(self):
        html = """
        <html>
          <head><title>The science of safe and healthy baby sleep</title></head>
          <body>
            <article>
              <h1>The science of safe and healthy baby sleep</h1>
              <p>Researchers found that calmer bedtime routines, consistent sleep cues, and safer crib setup can reduce overnight disruption.</p>
              <p>Parents also benefit from practical guidance on wake windows, naps, and age-appropriate sleep expectations during the first year.</p>
            </article>
          </body>
        </html>
        """

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=html):
            article = fetch_generic_web_article(
                "https://www.bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep"
            )

        self.assertIsNotNone(article)
        self.assertEqual(article["title"], "The science of safe and healthy baby sleep")
        self.assertEqual(
            article["url"],
            "https://bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep",
        )
        self.assertEqual(article["source_type"], "web_article")
        self.assertIn("wake windows", article["content"])

    def test_fetch_rss_articles_accepts_generic_html_article_url(self):
        html = """
        <html>
          <head><title>Workflow case study</title></head>
          <body>
            <main>
              <article>
                <p>Teams that documented approval rules before automation reduced rework and made article selection more reliable.</p>
                <p>They also cleaned up intake forms so editors could judge source quality faster during each digest run.</p>
              </article>
            </main>
          </body>
        </html>
        """

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=html):
            items = fetch_rss_articles("https://example.com/articles/workflow-case-study")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "Workflow case study")
        self.assertEqual(items[0]["source_type"], "web_article")
        self.assertIn("approval rules", items[0]["content"])

    def test_fetch_generic_web_article_rejects_navigation_only_page(self):
        html = """
        <html>
          <head><title>Home</title></head>
          <body>
            <nav>Products Pricing Login About Contact</nav>
            <footer>Cookie settings Newsletter Support</footer>
          </body>
        </html>
        """

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=html):
            article = fetch_generic_web_article("https://example.com/")

        self.assertIsNone(article)

    def test_fetch_generic_web_article_accepts_lullaby_trust_live_page_shape(self):
        html = """
        <html>
          <head><title>Baby sleep patterns | The Lullaby Trust</title></head>
          <body class="wp-singular page-template-default mobile-slide-out-menu-enabled">
            <nav>Baby safety Bereavement support Professionals hub Donate</nav>
            <main role="main" id="main-content">
              <section class="featured-banner__textbox">
                <h1 class="featured-banner__heading">Baby sleep patterns: how long should my baby sleep?</h1>
                <div class="featured-banner__copy">
                  <p>Parents and carers often worry about their babies' sleep and might try tips and hacks to get them to sleep longer, but these can actually be dangerous.</p>
                </div>
              </section>
              <section class="text-and-media">
                <div class="text-and-media__copy wysiwyg">
                  <h2>It's typical for babies to wake up often</h2>
                  <p>The first year when your baby wakes up often can be tough, and just when you think you have things figured out, your baby's sleep pattern changes again.</p>
                  <h3>How much sleep do babies need?</h3>
                  <p>Babies have small stomachs and will wake often throughout the night to feed. Every baby is different and sleep patterns vary greatly, so use this as a guide.</p>
                  <h3>Feeling exhausted?</h3>
                  <p>Sleep deprivation can feel intense, so many families benefit from simple routines, shared support, and realistic expectations during the first months.</p>
                </div>
              </section>
            </main>
            <footer>Learn more about keeping your baby safe Newsletter Contact us</footer>
          </body>
        </html>
        """

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=html):
            article = fetch_generic_web_article(
                "https://www.lullabytrust.org.uk/baby-safety/being-a-parent-or-caregiver/baby-sleep-patterns/"
            )

        self.assertIsNotNone(article)
        self.assertEqual(article["title"], "Baby sleep patterns | The Lullaby Trust")
        self.assertIn("wake up often", article["content"])

    def test_fetch_generic_web_article_accepts_johns_hopkins_style_medical_article_layout(self):
        html = """
        <html>
          <head>
            <title>Infant Safe Sleep | Johns Hopkins Medicine</title>
          </head>
          <body class="site-body utility-nav-enabled">
            <header>
              <nav>Find a Doctor Locations MyChart Pay Bill Request Appointment</nav>
            </header>
            <div class="layout">
              <aside class="sidebar-nav">
                <h2>In this section</h2>
                <ul>
                  <li><a href="/health/conditions-and-diseases">Conditions and diseases</a></li>
                  <li><a href="/health/wellness-and-prevention">Wellness and prevention</a></li>
                  <li><a href="/health/treatment-tests-and-therapies">Treatments and tests</a></li>
                </ul>
              </aside>
              <main role="main">
                <article class="article-content">
                  <h1>Infant Safe Sleep</h1>
                  <p class="article-intro">Safe sleep guidance helps lower the risk of sleep-related infant deaths and gives families a clearer bedtime routine from the first days at home.</p>
                  <div class="article-body">
                    <p>Babies should always be placed on their backs for every sleep, including naps and overnight sleep. A firm, flat sleep surface and a crib free of loose blankets, pillows, and toys helps reduce risk.</p>
                    <p>Room-sharing without bed-sharing is recommended during the early months. Parents and caregivers can also watch for overheating, keep sleep clothing simple, and ask their pediatrician for help if they are worried about frequent waking or feeding patterns.</p>
                    <h2>Creating a safer sleep space</h2>
                    <p>Use a safety-approved crib, bassinet, or portable play yard with a fitted sheet. Keep soft bedding and sleep positioners out of the sleep space, and return the baby to their own sleep surface after feeding or comforting.</p>
                  </div>
                </article>
                <section class="related-content">
                  <h2>Related</h2>
                  <p>Learn more about pediatric wellness, childbirth classes, and hospital visitor information.</p>
                </section>
                <section class="promo-callout">
                  <p>Request an appointment or sign up for our health newsletter.</p>
                </section>
              </main>
            </div>
            <footer>About us Contact us Newsroom For providers</footer>
          </body>
        </html>
        """

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=html):
            article = fetch_generic_web_article(
                "https://www.hopkinsmedicine.org/health/wellness-and-prevention/infant-safe-sleep"
            )

        self.assertIsNotNone(article)
        self.assertEqual(article["title"], "Infant Safe Sleep | Johns Hopkins Medicine")
        self.assertEqual(article["source_type"], "web_article")
        self.assertIn("placed on their backs for every sleep", article["content"])
        self.assertIn("firm, flat sleep surface", article["content"])
        self.assertIn("Room-sharing without bed-sharing is recommended", article["content"])
        self.assertGreater(len(article["content"]), 450)
        self.assertNotIn("Find a Doctor Locations MyChart Pay Bill Request Appointment", article["content"])
        self.assertNotIn("Request an appointment or sign up for our health newsletter", article["content"])

    def test_fetch_generic_web_article_rejects_short_unstructured_page_even_with_title(self):
        html = """
        <html>
          <head><title>Baby sleep</title></head>
          <body>
            <div>Baby sleep basics. Learn more today.</div>
          </body>
        </html>
        """

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=html):
            article = fetch_generic_web_article("https://example.com/baby-sleep")

        self.assertIsNone(article)

    def test_fetch_dev_to_article_content_preserves_markdown_headings(self):
        payload = {
            "id": 101,
            "title": "Architect A Personalized Multi-Agent System with Long-Term Memory",
            "url": "https://dev.to/alice/multi-agent-memory",
            "description": "A detailed article.",
            "body_markdown": (
                "# Architect A Personalized Multi-Agent System with Long-Term Memory\n\n"
                "## Long-Term Memory\n"
                "Memory helps preserve context.\n\n"
                "## Governance Layer\n"
                "Controls access across agent steps."
            ),
            "body_html": (
                "<h1>Architect A Personalized Multi-Agent System with Long-Term Memory</h1>"
                "<h2>Long-Term Memory</h2><p>Memory helps preserve context.</p>"
                "<h2>Governance Layer</h2><p>Controls access across agent steps.</p>"
            ),
        }

        with patch("services.sources.rss_adapter._fetch_json", return_value=payload):
            article = fetch_dev_to_article_content(101)

        self.assertIsNotNone(article)
        metadata = article["metadata"]
        self.assertEqual(metadata["heading_extraction_strategy"], "markdown_headings")
        self.assertEqual(metadata["raw_html_heading_count"], 3)
        self.assertEqual(metadata["extracted_heading_count"], 3)
        self.assertEqual(
            metadata["headings"],
            [
                "Architect A Personalized Multi-Agent System with Long-Term Memory",
                "Long-Term Memory",
                "Governance Layer",
            ],
        )
        self.assertEqual(
            metadata["sample_detected_headings"],
            [
                "Architect A Personalized Multi-Agent System with Long-Term Memory",
                "Long-Term Memory",
                "Governance Layer",
            ],
        )

    def test_fetch_dev_to_article_content_does_not_treat_numeric_id_as_url_when_api_lookup_fails(self):
        with patch("services.sources.rss_adapter._fetch_json", return_value=None), patch(
            "services.sources.rss_adapter._fetch_url_text"
        ) as mock_fetch_url_text:
            article = fetch_dev_to_article_content("3630333")

        self.assertIsNone(article)
        mock_fetch_url_text.assert_not_called()

    def test_extract_html_content_diagnostics_prefers_article_tag_and_removes_boilerplate(self):
        html = """
        <html>
          <body>
            <header><nav>Home Pricing Docs Sign in</nav></header>
            <article>
              <h1>Why teams redesign workflow before adding AI</h1>
              <p>Teams that mapped decisions before introducing AI reduced review loops.</p>
              <p>They removed duplicate approvals, clarified owners, and improved source traceability.</p>
              <ul><li>Clear handoffs</li><li>Fewer review repeats</li></ul>
            </article>
            <footer>Subscribe Share Cookie preferences</footer>
          </body>
        </html>
        """

        extraction = _extract_html_content_diagnostics(html)

        self.assertEqual(extraction["extraction_method"], "article_tag")
        self.assertGreater(extraction["extracted_content_length"], 120)
        self.assertIn("Teams that mapped decisions before introducing AI reduced review loops.", extraction["content"])
        self.assertNotIn("Home Pricing Docs", extraction["content"])
        self.assertNotIn("Subscribe Share Cookie", extraction["content"])
        article_candidate = next(
            candidate for candidate in extraction["extraction_candidates"] if candidate["selector"] == "article_tag"
        )
        main_candidate = next(
            candidate for candidate in extraction["extraction_candidates"] if candidate["selector"] == "main_tag"
        )
        self.assertTrue(article_candidate["found"])
        self.assertGreater(article_candidate["text_length"], 120)
        self.assertIsNone(article_candidate["rejection_reason"])
        self.assertEqual(main_candidate["rejection_reason"], "not found")
        self.assertEqual(extraction["raw_html_heading_count"], 1)
        self.assertEqual(extraction["extracted_heading_count"], 1)
        self.assertEqual(extraction["heading_extraction_strategy"], "html_headings")
        self.assertEqual(extraction["headings"], ["Why teams redesign workflow before adding AI"])

    def test_extract_html_content_diagnostics_falls_back_to_main_content(self):
        html = """
        <html>
          <body>
            <main>
              <section>
                <h1>Structured intake fixed triage only after labels improved</h1>
                <p>Operators moved faster after replacing free-form requests with structured intake.</p>
                <p>Routing still broke until teams standardized labels and clarified queue ownership.</p>
              </section>
            </main>
          </body>
        </html>
        """

        extraction = _extract_html_content_diagnostics(html)

        self.assertEqual(extraction["extraction_method"], "main_tag")
        self.assertGreater(extraction["extracted_content_length"], 120)
        self.assertIn("Operators moved faster after replacing free-form requests with structured intake.", extraction["content"])

    def test_extract_html_content_diagnostics_marks_navigation_only_pages_as_weak(self):
        html = """
        <html>
          <body>
            <nav>Home Pricing Login About Contact</nav>
            <div class="menu">Products Platform Enterprise Careers</div>
            <footer>Cookie settings Newsletter Sign up</footer>
          </body>
        </html>
        """

        extraction = _extract_html_content_diagnostics(html)

        self.assertIn(extraction["extraction_method"], {"fallback_text", "no_candidate_text"})
        self.assertLess(extraction["extracted_content_length"], 200)
        self.assertIsNotNone(extraction["extraction_warning"])
        article_candidate = next(
            candidate for candidate in extraction["extraction_candidates"] if candidate["selector"] == "article_tag"
        )
        self.assertFalse(article_candidate["found"])
        self.assertEqual(article_candidate["rejection_reason"], "not found")

    def test_remove_boilerplate_nodes_handles_nested_decomposed_nodes_safely(self):
        html = """
        <html>
          <body>
            <header>
              <div class="banner">
                <span>Cookie banner</span>
              </div>
            </header>
            <article>
              <h1>Useful article</h1>
              <p>Teams got better results after documenting the approval path before adding automation.</p>
              <p>That left less room for editorial confusion and made later model output easier to verify.</p>
            </article>
          </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")

        _remove_boilerplate_nodes(soup)

        extraction = _extract_html_content_diagnostics(str(soup))
        self.assertIn("approval path", extraction["content"])
        self.assertNotIn("Cookie banner", extraction["content"])

    def test_remove_boilerplate_nodes_skips_tags_with_missing_attrs(self):
        html = """
        <html>
          <body>
            <article>
              <p>Operators replaced vague requests with structured intake and clearer review steps.</p>
              <p>This reduced back-and-forth and preserved source traceability for every change.</p>
            </article>
          </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        if article is not None:
            article.attrs = None

        _remove_boilerplate_nodes(soup)

        self.assertIsNotNone(soup.find("article"))

    def test_fetch_rss_articles_reads_local_sample_feed_file(self):
        fixture_path = Path("tests/fixtures/sample_feed.xml")

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=""):
            items = fetch_rss_articles(str(fixture_path))

        self.assertEqual(len(items), 5)
        self.assertEqual(items[0]["source_name"], "DigestFlow Sample Feed")
        self.assertEqual(
            items[0]["title"],
            "AI briefing workflow cut research prep, but editors still blocked publish risk",
        )
        self.assertEqual(items[0]["url"], "https://example.com/articles/ai-briefing-workflow")
        self.assertIsInstance(items[0]["published_at"], str)
        json.dumps(items[0])

    def test_fetch_rss_articles_reads_file_url_sample_feed(self):
        fixture_path = Path("tests/fixtures/sample_feed.xml").resolve()

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=""):
            items = fetch_rss_articles(fixture_path.as_uri(), limit=2)

        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["url"], "https://example.com/articles/support-triage-handoffs")

    def test_fetch_rss_articles_returns_json_serializable_published_at(self):
        fake_feed = SimpleNamespace(
            feed=SimpleNamespace(title="Ars Technica"),
            entries=[
                SimpleNamespace(
                    title="Example title",
                    link="https://example.com/post",
                    summary="<p>Hello <b>world</b></p>",
                    published_parsed=time.struct_time((2026, 4, 30, 12, 0, 0, 0, 120, 0)),
                )
            ],
        )
        fake_feedparser = SimpleNamespace(parse=lambda _url: fake_feed)

        with patch.dict("sys.modules", {"feedparser": fake_feedparser}), patch(
            "services.sources.rss_adapter._fetch_feed_content",
            return_value=b"<rss></rss>",
        ), patch(
            "services.sources.rss_adapter._fetch_url_text",
            return_value="",
        ):
            items = fetch_rss_articles("https://example.com/feed")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "Example title")
        self.assertEqual(items[0]["source_name"], "Ars Technica")
        self.assertIsInstance(items[0]["published_at"], str)
        json.dumps(items[0])

    def test_fetch_rss_articles_prefers_fetched_html_body_over_short_rss_summary(self):
        fake_feed = SimpleNamespace(
            feed=SimpleNamespace(title="Example Feed"),
            entries=[
                SimpleNamespace(
                    title="Example title",
                    link="https://example.com/post",
                    summary="<p>Short RSS summary.</p>",
                    published_parsed=time.struct_time((2026, 4, 30, 12, 0, 0, 0, 120, 0)),
                )
            ],
        )
        fake_feedparser = SimpleNamespace(parse=lambda _url: fake_feed)
        article_html = """
        <html>
          <body>
            <article>
              <h1>Example title</h1>
              <p>Teams that redesigned the approval handoff before adding AI cut review time sharply and stopped
              losing hours to circular edits.</p>
              <p>They also replaced vague prompts with structured intake, which made later model steps more reliable
              and easier for editors to verify before publishing.</p>
            </article>
          </body>
        </html>
        """

        with patch.dict("sys.modules", {"feedparser": fake_feedparser}), patch(
            "services.sources.rss_adapter._fetch_feed_content",
            return_value=b"<rss></rss>",
        ), patch(
            "services.sources.rss_adapter._fetch_url_text",
            return_value=article_html,
        ):
            items = fetch_rss_articles("https://example.com/feed")

        self.assertEqual(len(items), 1)
        self.assertIn("approval handoff", items[0]["content"])
        self.assertNotEqual(items[0]["content"], "Short RSS summary.")
        self.assertEqual(items[0]["metadata"]["extraction_method"], "article_tag")
        self.assertGreater(items[0]["metadata"]["extracted_content_length"], len("Short RSS summary."))
        self.assertEqual(items[0]["metadata"]["final_content_source"], "html_article_body")
        self.assertEqual(items[0]["metadata"]["rss_summary_length"], len("Short RSS summary."))
        self.assertIsNone(items[0]["metadata"]["extraction_warning"])
        self.assertTrue(items[0]["metadata"]["extraction_candidates"])

    def test_fetch_rss_articles_falls_back_to_summary_when_html_fetch_fails(self):
        fake_feed = SimpleNamespace(
            feed=SimpleNamespace(title="Example Feed"),
            entries=[
                SimpleNamespace(
                    title="Example title",
                    link="https://example.com/post",
                    summary="<p>Short RSS summary.</p>",
                )
            ],
        )
        fake_feedparser = SimpleNamespace(parse=lambda _url: fake_feed)

        with patch.dict("sys.modules", {"feedparser": fake_feedparser}), patch(
            "services.sources.rss_adapter._fetch_feed_content",
            return_value=b"<rss></rss>",
        ), patch(
            "services.sources.rss_adapter._fetch_url_text",
            return_value="",
        ):
            items = fetch_rss_articles("https://example.com/feed")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["content"], "Short RSS summary.")
        self.assertEqual(items[0]["metadata"]["extraction_method"], "rss_summary_fallback")
        self.assertEqual(items[0]["metadata"]["extracted_content_length"], len("Short RSS summary."))
        self.assertEqual(items[0]["metadata"]["final_content_source"], "rss_summary")
        self.assertEqual(items[0]["metadata"]["html_extracted_content_length"], 0)
        self.assertEqual(items[0]["metadata"]["extraction_warning"], "html fetch failed; RSS summary used")
        self.assertEqual(items[0]["metadata"]["extraction_candidates"], [])

    def test_fetch_rss_articles_fallback_summary_preserves_html_extraction_candidates(self):
        fake_feed = SimpleNamespace(
            feed=SimpleNamespace(title="Example Feed"),
            entries=[
                SimpleNamespace(
                    title="OpenAI-style article",
                    link="https://example.com/openai-post",
                    summary="<p>Short RSS summary.</p>",
                )
            ],
        )
        fake_feedparser = SimpleNamespace(parse=lambda _url: fake_feed)
        navigation_like_html = """
        <html>
          <body>
            <header>OpenAI News Products Research Safety API Login Pricing</header>
            <nav>Products Research Safety API Login Pricing Enterprise Careers</nav>
            <footer>OpenAI News Support Docs Company</footer>
          </body>
        </html>
        """

        with patch.dict("sys.modules", {"feedparser": fake_feedparser}), patch(
            "services.sources.rss_adapter._fetch_feed_content",
            return_value=b"<rss></rss>",
        ), patch(
            "services.sources.rss_adapter._fetch_url_text",
            return_value=navigation_like_html,
        ):
            items = fetch_rss_articles("https://example.com/feed")

        self.assertEqual(items[0]["metadata"]["extraction_method"], "rss_summary_fallback")
        self.assertIn("RSS summary used", items[0]["metadata"]["extraction_warning"])
        self.assertTrue(items[0]["metadata"]["extraction_candidates"])
        article_candidate = next(
            candidate for candidate in items[0]["metadata"]["extraction_candidates"] if candidate["selector"] == "article_tag"
        )
        fallback_candidate = next(
            candidate for candidate in items[0]["metadata"]["extraction_candidates"] if candidate["selector"] == "fallback_text"
        )
        self.assertFalse(article_candidate["found"])
        self.assertEqual(article_candidate["rejection_reason"], "not found")
        self.assertFalse(fallback_candidate["found"])
        self.assertEqual(fallback_candidate["rejection_reason"], "no readable text extracted")

    def test_debug_snapshot_shows_skip_reason_for_missing_url(self):
        fake_feed = SimpleNamespace(
            feed=SimpleNamespace(title="Example Feed"),
            entries=[
                SimpleNamespace(
                    title="Entry without link",
                    summary="Hello",
                    description="Hello description",
                    published="Wed, 30 Apr 2026 12:00:00 GMT",
                )
            ],
        )
        fake_feedparser = SimpleNamespace(parse=lambda _url: fake_feed)

        with patch.dict("sys.modules", {"feedparser": fake_feedparser}), patch(
            "services.sources.rss_adapter._fetch_feed_content",
            return_value=b"<rss></rss>",
        ):
            snapshot = get_rss_debug_snapshot("https://example.com/feed")

        self.assertEqual(snapshot["feed_title"], "Example Feed")
        self.assertEqual(snapshot["total_entries"], 1)
        self.assertEqual(snapshot["entries"][0]["skip_reason"], "missing url")

    def test_fetch_feed_content_sends_user_agent_and_returns_bytes(self):
        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()

        class FakeOpener:
            def open(self, request, timeout):
                captured["user_agent"] = request.headers.get("User-agent")
                captured["accept"] = request.headers.get("Accept")
                captured["timeout"] = timeout
                return FakeResponse(b"<rss></rss>")

        captured = {}

        with patch("services.sources.rss_adapter.build_opener", return_value=FakeOpener()):
            content = _fetch_feed_content("https://example.com/feed")

        self.assertEqual(content, b"<rss></rss>")
        self.assertEqual(captured["user_agent"], "Mozilla/5.0 (compatible; DigestFlowRSS/0.1)")
        self.assertEqual(captured["accept"], "application/rss+xml, application/xml, text/xml, */*")
        self.assertEqual(captured["timeout"], 15)
