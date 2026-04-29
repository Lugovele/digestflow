from django.test import SimpleTestCase

from services.processing.ranker import rank_source_items


class RankSourceItemsTests(SimpleTestCase):
    def test_item_with_numbers_and_reduced_gets_higher_score(self):
        items = [
            {
                "title": "Low signal",
                "url": "https://example.com/low",
                "source": "Example Blog",
                "snippet": "A short neutral update without measurable result.",
            },
            {
                "title": "High signal",
                "url": "https://example.com/high",
                "source": "Example Research",
                "snippet": (
                    "The team reduced manual reporting time by 35% and improved weekly review "
                    "quality after changing the workflow."
                ),
            },
        ]

        selected, ranking_scores = rank_source_items(items, top_n=2)

        self.assertEqual(selected[0]["url"], "https://example.com/high")
        self.assertGreater(ranking_scores[0]["score"], ranking_scores[1]["score"])

    def test_ranker_does_not_remove_duplicates(self):
        items = [
            {
                "title": "Duplicate one",
                "url": "https://example.com/shared",
                "source": "Example Blog",
                "snippet": "A short update.",
            },
            {
                "title": "Duplicate two",
                "url": "https://example.com/shared",
                "source": "Example Blog",
                "snippet": "Another short update.",
            },
        ]

        selected, ranking_scores = rank_source_items(items, top_n=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual(len(ranking_scores), 2)
        self.assertEqual(selected[0]["url"], "https://example.com/shared")
        self.assertEqual(selected[1]["url"], "https://example.com/shared")

    def test_equal_scores_keep_original_order(self):
        items = [
            {
                "title": "First",
                "url": "https://example.com/first",
                "source": "Example Blog",
                "snippet": "Neutral snippet with no ranking keywords at all.",
            },
            {
                "title": "Second",
                "url": "https://example.com/second",
                "source": "Example Blog",
                "snippet": "Another neutral snippet with no ranking words.",
            },
        ]

        selected, _ = rank_source_items(items, top_n=2)

        self.assertEqual(
            [item["url"] for item in selected],
            ["https://example.com/first", "https://example.com/second"],
        )

    def test_top_n_limits_selected_items(self):
        items = [
            {
                "title": "One",
                "url": "https://example.com/1",
                "source": "Example Research",
                "snippet": "Reduced time by 20% with a longer operational summary for the team.",
            },
            {
                "title": "Two",
                "url": "https://example.com/2",
                "source": "Example Blog",
                "snippet": "Improved handoff quality in a measurable way for one workflow.",
            },
            {
                "title": "Three",
                "url": "https://example.com/3",
                "source": "Example Report",
                "snippet": "Growth reached 12% after the rollout and cut review cycles.",
            },
        ]

        selected, _ = rank_source_items(items, top_n=2)

        self.assertEqual(len(selected), 2)

    def test_ranking_scores_contains_url_and_score(self):
        items = [
            {
                "title": "One",
                "url": "https://example.com/1",
                "source": "Example Blog",
                "snippet": "A neutral source snippet.",
            }
        ]

        _, ranking_scores = rank_source_items(items, top_n=1)

        self.assertEqual(ranking_scores, [{"url": "https://example.com/1", "score": 0}])
