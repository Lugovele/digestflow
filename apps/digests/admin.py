from django.contrib import admin

from .models import Digest, DigestRun


@admin.register(DigestRun)
class DigestRunAdmin(admin.ModelAdmin):
    list_display = ("id", "topic", "status", "started_at", "finished_at", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("topic__name", "error_message")
    readonly_fields = ("metrics", "input_snapshot", "error_message")


@admin.register(Digest)
class DigestAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "run", "quality_score", "generated_at")
    search_fields = ("title", "summary")
    readonly_fields = ("key_points", "sources")
