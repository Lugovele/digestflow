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

    def __str__(self) -> str:
        return f"ContentPackage for {self.digest_id}"
