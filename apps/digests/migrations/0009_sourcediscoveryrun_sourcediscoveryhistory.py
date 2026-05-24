from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("topics", "0009_topicsource_is_pinned"),
        ("digests", "0008_alter_usedarticle_options"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="SourceDiscoveryRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "provider_name",
                    models.CharField(blank=True, max_length=64),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("started", "Started"),
                            ("completed", "Completed"),
                            ("blocked", "Blocked"),
                            ("failed", "Failed"),
                            ("partial_failed", "Partial failed"),
                        ],
                        default="started",
                        max_length=32,
                    ),
                ),
                ("started_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("search_recency_months", models.PositiveSmallIntegerField(default=1)),
                ("search_time_filter", models.CharField(blank=True, max_length=32)),
                ("query_count", models.PositiveIntegerField(default=0)),
                ("provider_result_count", models.PositiveIntegerField(default=0)),
                ("known_url_count", models.PositiveIntegerField(default=0)),
                ("accepted_count", models.PositiveIntegerField(default=0)),
                ("rejected_count", models.PositiveIntegerField(default=0)),
                ("new_suggestions_count", models.PositiveIntegerField(default=0)),
                ("already_known_count", models.PositiveIntegerField(default=0)),
                ("diagnostics", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "topic",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="source_discovery_runs",
                        to="topics.topic",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="source_discovery_runs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at", "-id"],
            },
        ),
        migrations.CreateModel(
            name="SourceDiscoveryHistory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("normalized_url", models.URLField(max_length=500)),
                ("url", models.URLField(max_length=500)),
                ("title", models.CharField(blank=True, max_length=300)),
                ("snippet", models.TextField(blank=True)),
                ("domain", models.CharField(blank=True, max_length=255)),
                ("provider_name", models.CharField(blank=True, max_length=64)),
                ("query_text", models.CharField(blank=True, max_length=500)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("seen", "Seen"),
                            ("shown", "Shown"),
                            ("kept", "Kept"),
                            ("removed_by_user", "Removed by user"),
                            ("rejected_by_quality", "Rejected by quality"),
                        ],
                        default="seen",
                        max_length=32,
                    ),
                ),
                (
                    "last_run_outcome",
                    models.CharField(
                        blank=True,
                        choices=[
                            ("", "None"),
                            ("new_shown", "New shown"),
                            ("already_known", "Already known"),
                            ("duplicate_url", "Duplicate URL"),
                            ("duplicate_domain", "Duplicate domain"),
                            ("previously_removed", "Previously removed"),
                            ("previously_rejected", "Previously rejected"),
                            ("quality_rejected", "Quality rejected"),
                            ("stale_rejected", "Stale rejected"),
                            ("commercial_rejected", "Commercial rejected"),
                        ],
                        default="",
                        max_length=32,
                    ),
                ),
                ("source_content_type", models.CharField(blank=True, max_length=64)),
                ("quality_score", models.FloatField(default=0.0)),
                ("substance_score", models.FloatField(default=0.0)),
                ("commercial_intent_score", models.FloatField(default=0.0)),
                ("quality_rejection_reason", models.TextField(blank=True)),
                ("freshness_status", models.CharField(blank=True, max_length=32)),
                ("detected_publication_date", models.DateField(blank=True, null=True)),
                ("detected_publication_year", models.PositiveSmallIntegerField(blank=True, null=True)),
                ("first_seen_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("last_seen_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("seen_count", models.PositiveIntegerField(default=1)),
                ("created_topic_source", models.BooleanField(default=False)),
                ("diagnostics", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "discovery_run",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="history_items",
                        to="digests.sourcediscoveryrun",
                    ),
                ),
                (
                    "topic",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="source_discovery_history",
                        to="topics.topic",
                    ),
                ),
                (
                    "topic_source",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="source_discovery_history",
                        to="topics.topicsource",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="source_discovery_history",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-last_seen_at", "-id"],
            },
        ),
        migrations.AddConstraint(
            model_name="sourcediscoveryhistory",
            constraint=models.UniqueConstraint(
                fields=("topic", "normalized_url"),
                name="unique_source_discovery_history_per_topic_url",
            ),
        ),
    ]
