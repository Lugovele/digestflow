from django.db import models


class DigestRun(models.Model):
    """Ключевая сущность запуска для pipeline-first workflow."""

    STATUS_PENDING = "pending"
    STATUS_COLLECTING = "collecting"
    STATUS_PROCESSING = "processing"
    STATUS_GENERATING_DIGEST = "generating_digest"
    STATUS_PACKAGING = "packaging"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_COLLECTING, "Collecting"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_GENERATING_DIGEST, "Generating digest"),
        (STATUS_PACKAGING, "Packaging"),
        (STATUS_COMPLETED, "Completed"),
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
    summary = models.TextField()
    key_points = models.JSONField(default=list)
    sources = models.JSONField(default=list)
    quality_score = models.FloatField(default=0.0)
    generated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.title
