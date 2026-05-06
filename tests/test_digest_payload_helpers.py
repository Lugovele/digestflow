from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.digests.models import Digest, DigestRun
from apps.topics.models import Topic


class DigestPayloadHelperTests(TestCase):
    def test_get_articles_returns_payload_articles(self) -> None:
        user = get_user_model().objects.create_user(username="digest-helper-user")
        topic = Topic.objects.create(
            user=user,
            name="Helper topic",
            keywords=["helper"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic)
        digest = Digest.objects.create(
            run=run,
            title="Digest for Helper topic",
            payload={
                "version": 1,
                "title": "Digest for Helper topic",
                "articles": [
                    {
                        "url": "https://example.com/article-1",
                        "title": "Helper article title",
                        "summary": "Helper summary",
                        "key_points": ["Point one"],
                        "content_type": "news",
                        "confidence": 0.7,
                    }
                ],
            },
        )

        self.assertEqual(digest.get_payload_title(), "Digest for Helper topic")
        self.assertTrue(digest.has_articles())
        self.assertEqual(len(digest.get_articles()), 1)
        self.assertEqual(digest.get_articles()[0]["url"], "https://example.com/article-1")
        self.assertEqual(digest.get_articles()[0]["title"], "Helper article title")

    def test_missing_payload_version_is_treated_as_version_one(self) -> None:
        user = get_user_model().objects.create_user(username="digest-helper-v0-user")
        topic = Topic.objects.create(
            user=user,
            name="Legacy payload topic",
            keywords=["legacy"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic)
        digest = Digest.objects.create(
            run=run,
            title="Digest for Legacy payload topic",
            payload={
                "title": "Digest for Legacy payload topic",
                "articles": [
                    {
                        "url": "https://example.com/article-legacy",
                        "title": "Legacy article title",
                        "summary": "Legacy summary",
                        "key_points": ["Legacy point"],
                        "content_type": "opinion",
                        "confidence": 0.6,
                    }
                ],
            },
        )

        self.assertEqual(digest.get_payload_version(), 1)
        self.assertEqual(digest.get_payload_title(), "Digest for Legacy payload topic")
        self.assertTrue(digest.has_articles())
