from django.db import models


class Article(models.Model):
    """Нормализованный источник статьи, связанный с Topic."""

    topic = models.ForeignKey(
        "topics.Topic",
        on_delete=models.CASCADE,
        related_name="articles",
    )
    title = models.CharField(max_length=300)
    url = models.URLField(max_length=500)
    source_name = models.CharField(max_length=160)
    snippet = models.TextField()
    published_at = models.DateTimeField(null=True, blank=True)
    raw_payload = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["topic", "url"], name="unique_article_url_per_topic")
        ]

    def __str__(self) -> str:
        return self.title
