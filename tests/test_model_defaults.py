from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.digests.models import DigestRun
from apps.topics.models import Source, Topic


class TopicAndSourceModelTests(TestCase):
    def test_topic_uses_minimal_schema_defaults(self) -> None:
        user = get_user_model().objects.create_user(username="topic-defaults-user")

        topic = Topic.objects.create(
            user=user,
            name="AI workflows",
        )

        self.assertEqual(topic.source_mode, Topic.SOURCE_MODE_AUTOMATIC)
        self.assertEqual(topic.default_quality_threshold, 0.4)
        self.assertIsNone(topic.source_url)

    def test_source_can_be_created_for_topic(self) -> None:
        user = get_user_model().objects.create_user(username="source-user")
        topic = Topic.objects.create(
            user=user,
            name="AI research",
        )

        source = Source.objects.create(
            topic=topic,
            original_url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="dev_to_tag",
            platform="dev.to",
        )

        self.assertEqual(source.topic, topic)
        self.assertEqual(source.validation_status, Source.VALIDATION_PENDING)
        self.assertTrue(source.is_active)
        self.assertEqual(str(source), "https://dev.to/api/articles?tag=ai")


class DigestRunModelDefaultsTests(TestCase):
    def test_digest_run_supports_new_optional_snapshot_fields(self) -> None:
        user = get_user_model().objects.create_user(username="run-defaults-user")
        topic = Topic.objects.create(
            user=user,
            name="Automation systems",
        )

        run = DigestRun.objects.create(topic=topic)

        self.assertEqual(run.result_message, "")
        self.assertEqual(run.source_mode, "")
        self.assertEqual(run.audience_key, "")
        self.assertIsNone(run.quality_threshold_used)
