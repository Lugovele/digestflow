from django.conf import settings
from django.db import models
from django.db.models import Max
from django.db.models import Q


class TopicSourceMode(models.TextChoices):
    DISCOVERY_ONLY = "discovery_only", "new only"
    CURATED_ONLY = "curated_only", "saved only"
    HYBRID = "hybrid", "saved & new"


class TopicSourceOrigin(models.TextChoices):
    MANUAL = "manual", "Manual"
    DISCOVERED = "discovered", "Discovered"
    CURATED = "curated", "Curated"


class Topic(models.Model):
    """User-owned topic for digest generation."""

    # Backward-compatible aliases used across existing pipeline/tests.
    SOURCE_MODE_AUTOMATIC = TopicSourceMode.DISCOVERY_ONLY
    SOURCE_MODE_CUSTOM_ONLY = TopicSourceMode.CURATED_ONLY
    SOURCE_MODE_HYBRID = TopicSourceMode.HYBRID
    SOURCE_MODE_CHOICES = TopicSourceMode.choices

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=160)
    source_url = models.URLField(blank=True, null=True)
    source_mode = models.CharField(
        max_length=20,
        choices=SOURCE_MODE_CHOICES,
        default=TopicSourceMode.DISCOVERY_ONLY,
    )
    default_quality_threshold = models.FloatField(default=0.4)
    description = models.TextField(blank=True)
    keywords = models.JSONField(default=list, blank=True)
    excluded_keywords = models.JSONField(default=list, blank=True)
    focus_initialized = models.BooleanField(default=False)
    display_order = models.PositiveIntegerField(default=0, db_index=True)
    is_active = models.BooleanField(default=True)
    committed_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["display_order", "name"]
        constraints = [
            models.UniqueConstraint(fields=["user", "name"], name="unique_topic_per_user")
        ]

    def __str__(self) -> str:
        return self.name

    def save(self, *args, **kwargs):
        if self._state.adding and not self.display_order:
            max_display_order = (
                type(self).objects.filter(user=self.user).aggregate(max_value=Max("display_order")).get("max_value") or 0
            )
            self.display_order = max_display_order + 1
        super().save(*args, **kwargs)

    @property
    def uses_source_discovery(self) -> bool:
        return self.source_mode in {
            TopicSourceMode.DISCOVERY_ONLY,
            TopicSourceMode.HYBRID,
        }

    @property
    def uses_curated_sources(self) -> bool:
        return self.source_mode in {
            TopicSourceMode.CURATED_ONLY,
            TopicSourceMode.HYBRID,
        }


class TopicSource(models.Model):
    """Persistent topic-level source record."""

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
    name = models.CharField(max_length=160, blank=True)
    url = models.URLField()
    normalized_url = models.URLField()
    source_type = models.CharField(max_length=50, blank=True)
    origin = models.CharField(
        max_length=16,
        choices=TopicSourceOrigin.choices,
        default=TopicSourceOrigin.CURATED,
    )
    platform = models.CharField(max_length=50, blank=True)
    validation_status = models.CharField(
        max_length=16,
        choices=VALIDATION_CHOICES,
        default=VALIDATION_PENDING,
    )
    last_validation_error = models.TextField(blank=True)
    is_pinned = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["topic_id", "id"]
        db_table = "topics_source"
        constraints = [
            models.UniqueConstraint(
                fields=["topic", "normalized_url"],
                condition=Q(is_active=True),
                name="unique_active_topic_source_url",
            )
        ]

    def __str__(self) -> str:
        return self.name or self.normalized_url or self.url

    @property
    def original_url(self) -> str:
        return self.url

    @original_url.setter
    def original_url(self, value: str) -> None:
        self.url = value


class Source(TopicSource):
    """Backward-compatible proxy alias during the TopicSource transition."""

    class Meta:
        proxy = True


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
