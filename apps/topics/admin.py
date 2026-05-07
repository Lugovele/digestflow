from django.contrib import admin

from .models import DigestSettings, Source, Topic


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "user",
        "source_mode",
        "default_quality_threshold",
        "is_active",
        "updated_at",
    )
    list_filter = ("is_active", "source_mode")
    search_fields = ("name", "description", "source_url")


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


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "topic",
        "platform",
        "source_type",
        "validation_status",
        "is_active",
        "updated_at",
    )
    list_filter = ("validation_status", "is_active", "platform", "source_type")
    search_fields = ("original_url", "normalized_url", "topic__name")
