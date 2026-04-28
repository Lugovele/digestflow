from django.contrib import admin

from .models import DigestSettings, Topic


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "user", "is_active", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("name", "description")


@admin.register(DigestSettings)
class DigestSettingsAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "topic",
        "frequency",
        "max_sources",
        "min_source_quality_score",
        "include_carousel_outline",
    )
    list_filter = ("frequency", "include_carousel_outline")
