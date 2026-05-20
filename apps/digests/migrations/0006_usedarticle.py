from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("digests", "0005_digestrun_audience_key_and_more"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="UsedArticle",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("normalized_url", models.URLField(max_length=500)),
                ("article_url", models.URLField(max_length=500)),
                ("title", models.CharField(blank=True, max_length=300)),
                ("source_url", models.URLField(blank=True, max_length=500)),
                ("used_at", models.DateTimeField(auto_now=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("digest_run", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="used_articles", to="digests.digestrun")),
                ("topic", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="used_articles", to="topics.topic")),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="used_articles", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["-used_at", "-id"],
                "constraints": [
                    models.UniqueConstraint(fields=("topic", "normalized_url"), name="unique_used_article_per_topic_url"),
                ],
            },
        ),
    ]
