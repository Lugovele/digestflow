from django.db import migrations, models


def backfill_committed_topics(apps, schema_editor):
    Topic = apps.get_model("topics", "Topic")
    TopicSource = apps.get_model("topics", "TopicSource")
    DigestRun = apps.get_model("digests", "DigestRun")

    topic_ids_with_sources = set(
        TopicSource.objects.values_list("topic_id", flat=True).distinct()
    )
    topic_ids_with_runs = set(
        DigestRun.objects.values_list("topic_id", flat=True).distinct()
    )

    committed_topic_ids = topic_ids_with_sources.union(topic_ids_with_runs)

    if committed_topic_ids:
        Topic.objects.filter(pk__in=committed_topic_ids, committed_at__isnull=True).update(
            committed_at=models.F("updated_at")
        )

    Topic.objects.filter(
        committed_at__isnull=True,
        source_url__isnull=False,
    ).exclude(source_url="").update(committed_at=models.F("updated_at"))


class Migration(migrations.Migration):

    dependencies = [
        ("topics", "0009_topicsource_is_pinned"),
        ("digests", "0009_sourcediscoveryrun_sourcediscoveryhistory"),
    ]

    operations = [
        migrations.AddField(
            model_name="topic",
            name="committed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.RunPython(backfill_committed_topics, migrations.RunPython.noop),
    ]
