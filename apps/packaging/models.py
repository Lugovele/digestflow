from django.db import models


class ContentPackage(models.Model):
    """Пакет для ручной публикации, без автопостинга в MVP."""

    digest = models.OneToOneField(
        "digests.Digest",
        on_delete=models.CASCADE,
        related_name="content_package",
    )
    post_text = models.TextField()
    hook_variants = models.JSONField(default=list)
    cta_variants = models.JSONField(default=list)
    hashtags = models.JSONField(default=list)
    carousel_outline = models.JSONField(default=list, blank=True)
    validation_report = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"ContentPackage for {self.digest.title}"

    def topic_name(self) -> str:
        return self.digest.run.topic.name

    def post_text_length(self) -> int:
        return len(self.post_text or "")

    def primary_hook(self) -> str:
        return self.hook_variants[0] if self.hook_variants else ""

    def primary_cta(self) -> str:
        return self.cta_variants[0] if self.cta_variants else ""

    def hashtags_text(self) -> str:
        return " ".join(self.hashtags)

    def is_publish_ready(self) -> bool:
        return self.validation_report.get("status") == "valid"
