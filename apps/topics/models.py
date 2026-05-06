from django.conf import settings
from django.db import models


class Topic(models.Model):
    """Пользовательская тема для регулярной генерации дайджестов."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=160)
    source_url = models.URLField(blank=True, null=True)
    description = models.TextField(blank=True)
    keywords = models.JSONField(default=list, blank=True)
    excluded_keywords = models.JSONField(default=list, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(fields=["user", "name"], name="unique_topic_per_user")
        ]

    def __str__(self) -> str:
        return self.name


class DigestSettings(models.Model):
    """Настройки темы для предсказуемой и экономной генерации."""

    FREQUENCY_CHOICES = [
        ("manual", "Manual"),
        ("daily", "Daily"),
        ("weekly", "Weekly"),
    ]

    topic = models.OneToOneField(
        Topic,
        on_delete=models.CASCADE,
        related_name="digest_settings",
    )
    frequency = models.CharField(max_length=20, choices=FREQUENCY_CHOICES, default="manual")
    max_sources = models.PositiveSmallIntegerField(default=20)
    min_source_quality_score = models.FloatField(default=0.55)
    output_language = models.CharField(max_length=16, default="en")
    include_carousel_outline = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Settings for {self.topic}"
