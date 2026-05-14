from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase


class TopicSourceMigrationTests(TransactionTestCase):
    available_apps = ["apps.topics", "apps.digests", "apps.sources", "django.contrib.auth", "django.contrib.contenttypes"]

    migrate_from = [("topics", "0004_topic_source_mode_discovery_abstraction")]
    migrate_to = [("topics", "0005_topicsource_persistent_multi_source")]

    def setUp(self) -> None:
        super().setUp()
        self.executor = MigrationExecutor(connection)
        self.executor.migrate(self.migrate_from)

        old_apps = self.executor.loader.project_state(self.migrate_from).apps
        User = old_apps.get_model("auth", "User")
        Topic = old_apps.get_model("topics", "Topic")

        user = User.objects.create(username="migration-user")
        Topic.objects.create(
            user=user,
            name="Migrated topic",
            source_url="https://example.com/feed.xml",
            source_mode="hybrid",
            default_quality_threshold=0.4,
            description="",
            keywords=["Migrated topic"],
            excluded_keywords=[],
            is_active=True,
        )

    def test_topic_source_url_is_backfilled_into_topicsource(self) -> None:
        self.executor = MigrationExecutor(connection)
        self.executor.migrate(self.migrate_to)

        apps = self.executor.loader.project_state(self.migrate_to).apps
        Topic = apps.get_model("topics", "Topic")
        TopicSource = apps.get_model("topics", "TopicSource")

        topic = Topic.objects.get(name="Migrated topic")
        sources = TopicSource.objects.filter(topic=topic)

        self.assertEqual(sources.count(), 1)
        source = sources.get()
        self.assertEqual(source.url, "https://example.com/feed.xml")
        self.assertEqual(source.origin, "manual")
        self.assertTrue(source.is_active)
