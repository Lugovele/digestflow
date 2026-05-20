from django.db import migrations, models
import django.db.models.deletion
from django.utils import timezone


def backfill_used_article_repeat_visibility(apps, schema_editor):
    UsedArticle = apps.get_model("digests", "UsedArticle")
    for article in UsedArticle.objects.all().iterator():
        timestamp = article.used_at or article.created_at or timezone.now()
        article.first_used_at = timestamp
        article.last_used_at = timestamp
        article.use_count = article.use_count or 1
        article.first_used_in_run_id = article.digest_run_id
        article.last_used_in_run_id = article.digest_run_id
        article.save(
            update_fields=[
                "first_used_at",
                "last_used_at",
                "use_count",
                "first_used_in_run",
                "last_used_in_run",
                "used_at",
            ]
        )


class Migration(migrations.Migration):

    dependencies = [
        ("digests", "0006_usedarticle"),
    ]

    operations = [
        migrations.AddField(
            model_name="usedarticle",
            name="first_used_at",
            field=models.DateTimeField(default=timezone.now),
        ),
        migrations.AddField(
            model_name="usedarticle",
            name="first_used_in_run",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="first_used_articles", to="digests.digestrun"),
        ),
        migrations.AddField(
            model_name="usedarticle",
            name="last_used_at",
            field=models.DateTimeField(default=timezone.now),
        ),
        migrations.AddField(
            model_name="usedarticle",
            name="last_used_in_run",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="last_used_articles", to="digests.digestrun"),
        ),
        migrations.AddField(
            model_name="usedarticle",
            name="use_count",
            field=models.PositiveIntegerField(default=1),
        ),
        migrations.RunPython(backfill_used_article_repeat_visibility, migrations.RunPython.noop),
    ]
