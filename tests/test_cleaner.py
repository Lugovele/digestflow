from django.test import SimpleTestCase

from services.processing.cleaner import clean_source_items


LONG_TEXT = (
    "A team working on internal workflows cut prep time from six hours to two and a half, "
    "but editors still had to check every claim before anything shipped. The process moved "
    "faster on paper, yet review stayed manual, handoffs stayed fragile, and teams still "
    "spent time cleaning up mistakes before the final step."
)


class CleanSourceItemsTests(SimpleTestCase):
    def test_cleaner_normalizes_title_and_snippet_whitespace(self):
        raw_items = [
            {
                "title": "  AI update   ",
                "url": "https://example.com/1",
                "source": " Example Source ",
                "snippet": f" First line \n\n second line {LONG_TEXT} ",
            }
        ]

        cleaned = clean_source_items(raw_items)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["title"], "AI update")
        self.assertIn("First line second line", cleaned[0]["snippet"])
        self.assertEqual(cleaned[0]["source"], "Example Source")

    def test_cleaner_removes_items_without_title_or_url(self):
        raw_items = [
            {"title": "Valid title", "url": "https://example.com/1", "snippet": LONG_TEXT},
            {"title": "   ", "url": "https://example.com/2", "snippet": LONG_TEXT},
            {"title": "Missing URL", "url": "   ", "snippet": LONG_TEXT},
        ]

        cleaned = clean_source_items(raw_items)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["url"], "https://example.com/1")

    def test_cleaner_preserves_optional_fields_in_normalized_output(self):
        raw_items = [
            {
                "title": "Signal",
                "url": "https://example.com/1",
                "source": "Research Lab",
                "snippet": LONG_TEXT,
                "published_at": "2026-04-30",
                "source_url": "https://dev.to/t/ai",
                "source_api_url": "https://dev.to/api/articles?tag=ai",
                "description": "Readable description",
                "metadata": {"source_type": "dev_to_tag"},
            }
        ]

        cleaned = clean_source_items(raw_items)

        self.assertEqual(cleaned[0]["published_at"], "2026-04-30")
        self.assertEqual(cleaned[0]["source_url"], "https://dev.to/t/ai")
        self.assertEqual(cleaned[0]["source_api_url"], "https://dev.to/api/articles?tag=ai")
        self.assertEqual(cleaned[0]["description"], "Readable description")
        self.assertEqual(cleaned[0]["metadata"]["source_type"], "dev_to_tag")
        self.assertEqual(
            set(cleaned[0].keys()),
            {
                "title",
                "url",
                "source",
                "source_name",
                "source_url",
                "source_api_url",
                "published_at",
                "snippet",
                "content",
                "description",
                "metadata",
            },
        )

    def test_cleaner_accepts_rss_items_with_snippet_only_and_preserves_source_name(self):
        raw_items = [
            {
                "title": "DEV post",
                "url": "https://dev.to/example-post",
                "source_name": "DEV Community: Example",
                "snippet": LONG_TEXT,
                "published_at": "2026-05-05T10:00:00+00:00",
            }
        ]

        cleaned = clean_source_items(raw_items)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["source"], "DEV Community: Example")
        self.assertEqual(cleaned[0]["source_name"], "DEV Community: Example")
        self.assertEqual(cleaned[0]["snippet"], LONG_TEXT)
        self.assertEqual(cleaned[0]["content"], LONG_TEXT)

    def test_cleaner_converts_html_heavy_content_to_plain_text(self):
        raw_items = [
            {
                "title": "HTML post",
                "url": "https://dev.to/html-post",
                "source_name": "DEV Community: Example",
                "content": (
                    "<div><p>A support team speeds up triage by 28%.</p>"
                    "<p><span>But bad labels still break routing.</span></p>"
                    "<pre><code>handoff_status = broken</code></pre>"
                    "<p>Teams still spend time checking the queue before the next step. "
                    "The workflow moves faster, but the mess still shows up later in review.</p></div>"
                ),
                "snippet": "<p>Short summary</p>",
            }
        ]

        cleaned = clean_source_items(raw_items)

        self.assertEqual(len(cleaned), 1)
        self.assertGreater(len(cleaned[0]["content"]), 0)
        self.assertNotIn("<div>", cleaned[0]["content"])
        self.assertNotIn("<code>", cleaned[0]["content"])
        self.assertIn("A support team speeds up triage by 28%.", cleaned[0]["content"])

    def test_cleaner_drops_items_with_too_little_text_after_html_cleaning(self):
        raw_items = [
            {
                "title": "Short RSS item",
                "url": "https://example.com/short",
                "source_name": "DEV Community: Example",
                "content": "<div><p>Too short.</p></div>",
            }
        ]

        cleaned = clean_source_items(raw_items)

        self.assertEqual(cleaned, [])
