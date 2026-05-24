from django.conf import settings
from django.db import models
from django.utils import timezone


class DigestRun(models.Model):
    """Ключевая сущность запуска для pipeline-first workflow."""

    STATUS_PENDING = "pending"
    STATUS_COLLECTING = "collecting"
    STATUS_PROCESSING = "processing"
    STATUS_GENERATING_DIGEST = "generating_digest"
    STATUS_PACKAGING = "packaging"
    STATUS_COMPLETED = "completed"
    STATUS_INSUFFICIENT_QUALITY = "insufficient_quality"
    STATUS_PARTIAL_FAILED = "partial_failed"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_COLLECTING, "Collecting"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_GENERATING_DIGEST, "Generating digest"),
        (STATUS_PACKAGING, "Packaging"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_INSUFFICIENT_QUALITY, "Insufficient quality"),
        (STATUS_PARTIAL_FAILED, "Partial failed"),
        (STATUS_FAILED, "Failed"),
    ]

    topic = models.ForeignKey(
        "topics.Topic",
        on_delete=models.CASCADE,
        related_name="digest_runs",
    )
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_PENDING)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    input_snapshot = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    result_message = models.TextField(blank=True)
    source_mode = models.CharField(max_length=20, blank=True)
    audience_key = models.CharField(max_length=64, blank=True)
    quality_threshold_used = models.FloatField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"DigestRun #{self.pk} [{self.status}]"


class Digest(models.Model):
    """Структурированный дайджест по обработанным источникам."""

    run = models.OneToOneField(DigestRun, on_delete=models.CASCADE, related_name="digest")
    title = models.CharField(max_length=240)
    payload = models.JSONField(default=dict)
    quality_score = models.FloatField(default=0.0)
    generated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.title

    def get_payload_title(self) -> str:
        payload = self._normalized_payload()
        return str(payload.get("title") or self.title).strip()

    def get_payload_version(self) -> int:
        payload = self._normalized_payload()
        return int(payload.get("version", 1))

    def get_articles(self) -> list[dict]:
        payload = self._normalized_payload()
        raw_articles = payload.get("articles", [])
        if not isinstance(raw_articles, list):
            return []

        normalized_articles: list[dict] = []
        for article in raw_articles:
            if not isinstance(article, dict):
                continue
            normalized_articles.append(
                {
                    "url": str(article.get("url", "")).strip(),
                    "title": str(article.get("title", "")).strip(),
                    "summary": str(article.get("summary", "")).strip(),
                    "key_points": article.get("key_points", []) if isinstance(article.get("key_points"), list) else [],
                    "content_type": str(article.get("content_type", "unknown")).strip() or "unknown",
                    "confidence": article.get("confidence", 0.0),
                }
            )
        return normalized_articles

    def has_articles(self) -> bool:
        return bool(self.get_articles())

    def _normalized_payload(self) -> dict:
        payload = self.payload if isinstance(self.payload, dict) else {}
        version = payload.get("version", 1)
        if not isinstance(version, int):
            version = 1
        return {
            "version": version,
            "title": payload.get("title") or self.title,
            "articles": payload.get("articles", []),
        }


class UsedArticle(models.Model):
    """Historical record of articles actually used in successful digest runs."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="used_articles",
    )
    topic = models.ForeignKey(
        "topics.Topic",
        on_delete=models.CASCADE,
        related_name="used_articles",
    )
    digest_run = models.ForeignKey(
        DigestRun,
        on_delete=models.CASCADE,
        related_name="used_articles",
    )
    first_used_in_run = models.ForeignKey(
        DigestRun,
        on_delete=models.CASCADE,
        related_name="first_used_articles",
        null=True,
        blank=True,
    )
    last_used_in_run = models.ForeignKey(
        DigestRun,
        on_delete=models.CASCADE,
        related_name="last_used_articles",
        null=True,
        blank=True,
    )
    normalized_url = models.URLField(max_length=500)
    article_url = models.URLField(max_length=500)
    title = models.CharField(max_length=300, blank=True)
    source_url = models.URLField(max_length=500, blank=True)
    use_count = models.PositiveIntegerField(default=1)
    first_used_at = models.DateTimeField(default=timezone.now)
    last_used_at = models.DateTimeField(default=timezone.now)
    used_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-last_used_at", "-id"]
        constraints = [
            models.UniqueConstraint(
                fields=["topic", "normalized_url"],
                name="unique_used_article_per_topic_url",
            )
        ]

    def __str__(self) -> str:
        return self.title or self.normalized_url or self.article_url


class SourceDiscoveryRun(models.Model):
    """One provider-backed source discovery execution for a topic."""

    STATUS_STARTED = "started"
    STATUS_COMPLETED = "completed"
    STATUS_BLOCKED = "blocked"
    STATUS_FAILED = "failed"
    STATUS_PARTIAL_FAILED = "partial_failed"

    STATUS_CHOICES = [
        (STATUS_STARTED, "Started"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_BLOCKED, "Blocked"),
        (STATUS_FAILED, "Failed"),
        (STATUS_PARTIAL_FAILED, "Partial failed"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="source_discovery_runs",
    )
    topic = models.ForeignKey(
        "topics.Topic",
        on_delete=models.CASCADE,
        related_name="source_discovery_runs",
    )
    provider_name = models.CharField(max_length=64, blank=True)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_STARTED)
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    search_recency_months = models.PositiveSmallIntegerField(default=1)
    search_time_filter = models.CharField(max_length=32, blank=True)
    query_count = models.PositiveIntegerField(default=0)
    provider_result_count = models.PositiveIntegerField(default=0)
    known_url_count = models.PositiveIntegerField(default=0)
    accepted_count = models.PositiveIntegerField(default=0)
    rejected_count = models.PositiveIntegerField(default=0)
    new_suggestions_count = models.PositiveIntegerField(default=0)
    already_known_count = models.PositiveIntegerField(default=0)
    diagnostics = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"SourceDiscoveryRun #{self.pk} [{self.status}]"


class SourceDiscoveryHistory(models.Model):
    """Topic-level memory of URLs seen through provider-backed source discovery."""

    STATUS_SEEN = "seen"
    STATUS_SHOWN = "shown"
    STATUS_KEPT = "kept"
    STATUS_REMOVED_BY_USER = "removed_by_user"
    STATUS_REJECTED_BY_QUALITY = "rejected_by_quality"

    STATUS_CHOICES = [
        (STATUS_SEEN, "Seen"),
        (STATUS_SHOWN, "Shown"),
        (STATUS_KEPT, "Kept"),
        (STATUS_REMOVED_BY_USER, "Removed by user"),
        (STATUS_REJECTED_BY_QUALITY, "Rejected by quality"),
    ]

    OUTCOME_NONE = ""
    OUTCOME_NEW_SHOWN = "new_shown"
    OUTCOME_ALREADY_KNOWN = "already_known"
    OUTCOME_DUPLICATE_URL = "duplicate_url"
    OUTCOME_DUPLICATE_DOMAIN = "duplicate_domain"
    OUTCOME_PREVIOUSLY_REMOVED = "previously_removed"
    OUTCOME_PREVIOUSLY_REJECTED = "previously_rejected"
    OUTCOME_QUALITY_REJECTED = "quality_rejected"
    OUTCOME_STALE_REJECTED = "stale_rejected"
    OUTCOME_COMMERCIAL_REJECTED = "commercial_rejected"

    RUN_OUTCOME_CHOICES = [
        (OUTCOME_NONE, "None"),
        (OUTCOME_NEW_SHOWN, "New shown"),
        (OUTCOME_ALREADY_KNOWN, "Already known"),
        (OUTCOME_DUPLICATE_URL, "Duplicate URL"),
        (OUTCOME_DUPLICATE_DOMAIN, "Duplicate domain"),
        (OUTCOME_PREVIOUSLY_REMOVED, "Previously removed"),
        (OUTCOME_PREVIOUSLY_REJECTED, "Previously rejected"),
        (OUTCOME_QUALITY_REJECTED, "Quality rejected"),
        (OUTCOME_STALE_REJECTED, "Stale rejected"),
        (OUTCOME_COMMERCIAL_REJECTED, "Commercial rejected"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="source_discovery_history",
    )
    topic = models.ForeignKey(
        "topics.Topic",
        on_delete=models.CASCADE,
        related_name="source_discovery_history",
    )
    discovery_run = models.ForeignKey(
        SourceDiscoveryRun,
        on_delete=models.SET_NULL,
        related_name="history_items",
        null=True,
        blank=True,
    )
    topic_source = models.ForeignKey(
        "topics.TopicSource",
        on_delete=models.SET_NULL,
        related_name="source_discovery_history",
        null=True,
        blank=True,
    )
    normalized_url = models.URLField(max_length=500)
    url = models.URLField(max_length=500)
    title = models.CharField(max_length=300, blank=True)
    snippet = models.TextField(blank=True)
    domain = models.CharField(max_length=255, blank=True)
    provider_name = models.CharField(max_length=64, blank=True)
    query_text = models.CharField(max_length=500, blank=True)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_SEEN)
    last_run_outcome = models.CharField(
        max_length=32,
        choices=RUN_OUTCOME_CHOICES,
        default=OUTCOME_NONE,
        blank=True,
    )
    source_content_type = models.CharField(max_length=64, blank=True)
    quality_score = models.FloatField(default=0.0)
    substance_score = models.FloatField(default=0.0)
    commercial_intent_score = models.FloatField(default=0.0)
    quality_rejection_reason = models.TextField(blank=True)
    freshness_status = models.CharField(max_length=32, blank=True)
    detected_publication_date = models.DateField(null=True, blank=True)
    detected_publication_year = models.PositiveSmallIntegerField(null=True, blank=True)
    first_seen_at = models.DateTimeField(default=timezone.now)
    last_seen_at = models.DateTimeField(default=timezone.now)
    seen_count = models.PositiveIntegerField(default=1)
    created_topic_source = models.BooleanField(default=False)
    diagnostics = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-last_seen_at", "-id"]
        constraints = [
            models.UniqueConstraint(
                fields=["topic", "normalized_url"],
                name="unique_source_discovery_history_per_topic_url",
            )
        ]

    def __str__(self) -> str:
        return self.title or self.normalized_url or self.url
