import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from services.sources.rss_adapter import (
    _fetch_feed_content,
    build_dev_to_api_url,
    detect_source_type,
    fetch_rss_articles,
    get_rss_debug_snapshot,
    normalize_source_url,
)


class RSSAdapterTests(SimpleTestCase):
    def test_normalize_source_url_detects_dev_to_tag_and_builds_internal_api_url(self):
        normalized = normalize_source_url("https://dev.to/t/ai")

        self.assertEqual(normalized.platform, "dev.to")
        self.assertEqual(normalized.source_type, "dev_to_tag")
        self.assertEqual(normalized.metadata["tag"], "ai")
        self.assertEqual(normalized.original_url, "https://dev.to/t/ai")
        self.assertEqual(normalized.normalized_url, "https://dev.to/api/articles?tag=ai")
        self.assertEqual(detect_source_type("https://dev.to/t/ai"), "dev_to_tag")
        self.assertEqual(build_dev_to_api_url("ai"), "https://dev.to/api/articles?tag=ai")

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
        self.assertEqual(items[0]["metadata"]["source_type"], "dev_to_tag")
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
        html = """
        <html>
          <head><title>Real dev.to article</title></head>
          <body><article><p>Full article body from HTML.</p></article></body>
        </html>
        """

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=html):
            items = fetch_rss_articles("https://dev.to/alice/real-devto-article")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://dev.to/alice/real-devto-article")
        self.assertEqual(items[0]["title"], "Real dev.to article")
        self.assertIn("Full article body from HTML.", items[0]["content"])
        self.assertEqual(items[0]["source_url"], "https://dev.to/alice/real-devto-article")
        self.assertIsNone(items[0]["source_api_url"])

    def test_fetch_rss_articles_reads_local_sample_feed_file(self):
        fixture_path = Path("tests/fixtures/sample_feed.xml")

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
        ):
            items = fetch_rss_articles("https://example.com/feed")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "Example title")
        self.assertEqual(items[0]["source_name"], "Ars Technica")
        self.assertIsInstance(items[0]["published_at"], str)
        json.dumps(items[0])

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

        captured = {}

        def fake_urlopen(request, timeout):
            captured["user_agent"] = request.headers.get("User-agent")
            captured["accept"] = request.headers.get("Accept")
            captured["timeout"] = timeout
            return FakeResponse(b"<rss></rss>")

        with patch("services.sources.rss_adapter.urlopen", side_effect=fake_urlopen):
            content = _fetch_feed_content("https://example.com/feed")

        self.assertEqual(content, b"<rss></rss>")
        self.assertEqual(captured["user_agent"], "Mozilla/5.0 (compatible; DigestFlowRSS/0.1)")
        self.assertEqual(captured["accept"], "application/rss+xml, application/xml, text/xml, */*")
        self.assertEqual(captured["timeout"], 15)
