from django.test import SimpleTestCase

from services.processing.ranker import rank_source_items


class RankSourceItemsTests(SimpleTestCase):
    def test_keyword_match_improves_ranking_score(self):
        items = [
            {
                "title": "General workflow note",
                "url": "https://example.com/low",
                "source": "Example Blog",
                "snippet": "A short neutral update without measurable result.",
            },
            {
                "title": "AI automation rollout",
                "url": "https://example.com/high",
                "source": "Example Research",
                "snippet": (
                    "The team reduced manual reporting time by 35% and improved weekly review "
                    "quality after changing the workflow."
                ),
            },
        ]

        selected, ranking_scores = rank_source_items(
            items,
            keywords=["AI automation"],
            top_n=2,
        )

        self.assertEqual(selected[0]["url"], "https://example.com/high")
        self.assertGreater(ranking_scores[0]["score"], ranking_scores[1]["score"])

    def test_excluded_keywords_reduce_quality_and_filter_weak_items(self):
        items = [
            {
                "title": "AI automation success",
                "url": "https://example.com/keep",
                "source": "Example Research",
                "snippet": "The rollout reduced manual work by 30% and improved reporting quality.",
            },
            {
                "title": "Crypto trend report",
                "url": "https://example.com/drop",
                "source": "Example Report",
                "snippet": "Crypto growth remained strong across several exchanges this quarter.",
            },
        ]

        selected, _ = rank_source_items(
            items,
            keywords=["AI automation"],
            excluded_keywords=["crypto"],
            top_n=2,
            min_quality_score=0.4,
        )

        self.assertEqual([item["url"] for item in selected], ["https://example.com/keep"])

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

    def test_ranking_scores_contains_url_score_and_quality_score(self):
        items = [
            {
                "title": "One",
                "url": "https://example.com/1",
                "source": "Example Blog",
                "snippet": "A neutral source snippet.",
            }
        ]

        _, ranking_scores = rank_source_items(items, top_n=1)

        self.assertEqual(
            ranking_scores,
            [{"url": "https://example.com/1", "score": 0, "quality_score": 0.0}],
        )
