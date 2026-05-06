from django.contrib import admin

from .models import Digest, DigestRun


@admin.register(DigestRun)
class DigestRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "topic",
        "status",
        "has_digest",
        "has_content_package",
        "started_at",
        "finished_at",
        "created_at",
    )
    list_filter = ("status", "created_at", "finished_at")
    search_fields = ("topic__name", "error_message")
    readonly_fields = (
        "topic",
        "status",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
        "input_snapshot",
        "metrics",
        "error_message",
        "has_digest",
        "has_content_package",
    )
    list_select_related = ("topic",)

    @admin.display(boolean=True, description="Digest")
    def has_digest(self, obj: DigestRun) -> bool:
        return hasattr(obj, "digest")

    @admin.display(boolean=True, description="Package")
    def has_content_package(self, obj: DigestRun) -> bool:
        return hasattr(obj, "digest") and hasattr(obj.digest, "content_package")


@admin.register(Digest)
class DigestAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "run", "quality_score", "generated_at")
    search_fields = ("title", "run__topic__name")
    readonly_fields = ("payload",)
