from django.test import SimpleTestCase

from services.processing.deduper import (
    dedupe_source_items,
    dedupe_source_items_with_metrics,
)


class DedupeSourceItemsTests(SimpleTestCase):
    def test_deduper_removes_duplicates_by_normalized_url(self):
        items = [
            {
                "title": "First title",
                "url": "https://example.com/article/",
                "source": "Example",
                "snippet": "One",
            },
            {
                "title": "Different title",
                "url": " https://example.com/article ",
                "source": "Example",
                "snippet": "Two",
            },
        ]

        unique_items, metrics = dedupe_source_items_with_metrics(items)

        self.assertEqual(len(unique_items), 1)
        self.assertEqual(metrics["duplicate_urls_removed"], 1)
        self.assertEqual(metrics["duplicate_titles_removed"], 0)

    def test_deduper_removes_duplicates_by_normalized_title(self):
        items = [
            {
                "title": "AI Weekly Update",
                "url": "https://example.com/article-1",
                "source": "Example",
                "snippet": "One",
            },
            {
                "title": "  ai weekly   update ",
                "url": "https://example.com/article-2",
                "source": "Example",
                "snippet": "Two",
            },
        ]

        unique_items, metrics = dedupe_source_items_with_metrics(items)

        self.assertEqual(len(unique_items), 1)
        self.assertEqual(metrics["duplicate_urls_removed"], 0)
        self.assertEqual(metrics["duplicate_titles_removed"], 1)

    def test_deduper_preserves_order_of_first_unique_items(self):
        items = [
            {
                "title": "First",
                "url": "https://example.com/1",
                "source": "Example",
                "snippet": "One",
            },
            {
                "title": "Second",
                "url": "https://example.com/2",
                "source": "Example",
                "snippet": "Two",
            },
            {
                "title": "first",
                "url": "https://example.com/3",
                "source": "Example",
                "snippet": "Duplicate by title",
            },
        ]

        unique_items = dedupe_source_items(items)

        self.assertEqual(
            [item["url"] for item in unique_items],
            ["https://example.com/1", "https://example.com/2"],
        )
