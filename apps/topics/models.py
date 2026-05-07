from django.conf import settings
from django.db import models


class Topic(models.Model):
    """User-owned topic for digest generation."""

    SOURCE_MODE_AUTOMATIC = "automatic"
    SOURCE_MODE_CUSTOM_ONLY = "custom_only"
    SOURCE_MODE_HYBRID = "hybrid"
    SOURCE_MODE_CHOICES = [
        (SOURCE_MODE_AUTOMATIC, "Automatic"),
        (SOURCE_MODE_CUSTOM_ONLY, "Custom only"),
        (SOURCE_MODE_HYBRID, "Hybrid"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=160)
    source_url = models.URLField(blank=True, null=True)
    source_mode = models.CharField(
        max_length=20,
        choices=SOURCE_MODE_CHOICES,
        default=SOURCE_MODE_AUTOMATIC,
    )
    default_quality_threshold = models.FloatField(default=0.4)
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


class Source(models.Model):
    """Minimal persistent topic-level source record."""

    VALIDATION_PENDING = "pending"
    VALIDATION_VALID = "valid"
    VALIDATION_INVALID = "invalid"
    VALIDATION_CHOICES = [
        (VALIDATION_PENDING, "Pending"),
        (VALIDATION_VALID, "Valid"),
        (VALIDATION_INVALID, "Invalid"),
    ]

    topic = models.ForeignKey(
        Topic,
        on_delete=models.CASCADE,
        related_name="sources",
    )
    original_url = models.URLField()
    normalized_url = models.URLField()
    source_type = models.CharField(max_length=50, blank=True)
    platform = models.CharField(max_length=50, blank=True)
    validation_status = models.CharField(
        max_length=16,
        choices=VALIDATION_CHOICES,
        default=VALIDATION_PENDING,
    )
    last_validation_error = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["topic_id", "id"]

    def __str__(self) -> str:
        return self.normalized_url or self.original_url


class DigestSettings(models.Model):
    """Topic settings for predictable generation behavior."""

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
