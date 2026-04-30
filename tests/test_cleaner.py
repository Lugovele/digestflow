from django.test import SimpleTestCase

from services.processing.cleaner import clean_source_items


class CleanSourceItemsTests(SimpleTestCase):
    def test_cleaner_normalizes_title_and_snippet_whitespace(self):
        raw_items = [
            {
                "title": "  AI&nbsp; update   ",
                "url": "https://example.com/1",
                "source": " Example Source ",
                "snippet": " First line \n\n second&nbsp;line ",
            }
        ]

        cleaned = clean_source_items(raw_items)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["title"], "AI update")
        self.assertEqual(cleaned[0]["snippet"], "First line second line")
        self.assertEqual(cleaned[0]["source"], "Example Source")

    def test_cleaner_removes_items_without_title_or_url(self):
        raw_items = [
            {"title": "Valid title", "url": "https://example.com/1", "snippet": "Valid snippet"},
            {"title": "   ", "url": "https://example.com/2", "snippet": "Missing title"},
            {"title": "Missing URL", "url": "   ", "snippet": "No URL"},
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
                "snippet": "Useful fact",
                "published_at": "2026-04-30",
            }
        ]

        cleaned = clean_source_items(raw_items)

        self.assertEqual(cleaned[0]["published_at"], "2026-04-30")
        self.assertEqual(
            set(cleaned[0].keys()),
            {"title", "url", "source", "published_at", "snippet"},
        )
