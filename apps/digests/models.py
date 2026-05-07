from django.db import models


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
