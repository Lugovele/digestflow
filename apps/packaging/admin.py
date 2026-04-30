import json

from django.contrib import admin
from django.utils.html import format_html

from .models import ContentPackage


@admin.register(ContentPackage)
class ContentPackageAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "topic_name",
        "digest",
        "is_publish_ready",
        "post_text_length",
        "created_at",
    )
    search_fields = ("digest__title", "digest__run__topic__name", "post_text")
    list_select_related = ("digest__run__topic",)
    readonly_fields = (
        "digest",
        "topic_name",
        "post_text_length",
        "primary_hook",
        "primary_cta",
        "hashtags_text",
        "hooks_pretty",
        "ctas_pretty",
        "hashtags_pretty",
        "carousel_pretty",
        "validation_pretty",
        "validation_report",
        "created_at",
        "updated_at",
    )
    fieldsets = (
        (
            "Context",
            {
                "fields": (
                    "digest",
                    "topic_name",
                    "created_at",
                    "updated_at",
                )
            },
        ),
        (
            "Main Post",
            {
                "fields": (
                    "post_text",
                    "post_text_length",
                )
            },
        ),
        (
            "Hooks and CTA",
            {
                "fields": (
                    "primary_hook",
                    "primary_cta",
                    "hooks_pretty",
                    "ctas_pretty",
                )
            },
        ),
        (
            "Hashtags and Carousel",
            {
                "fields": (
                    "hashtags_text",
                    "hashtags_pretty",
                    "carousel_pretty",
                )
            },
        ),
        (
            "Validation",
            {
                "fields": (
                    "validation_pretty",
                    "validation_report",
                )
            },
        ),
    )

    def is_publish_ready(self, obj: ContentPackage) -> bool:
        return obj.is_publish_ready()

    is_publish_ready.boolean = True
    is_publish_ready.short_description = "Valid"

    def hooks_pretty(self, obj: ContentPackage) -> str:
        return self._render_json_block(obj.hook_variants)

    hooks_pretty.short_description = "Hook Variants"

    def ctas_pretty(self, obj: ContentPackage) -> str:
        return self._render_json_block(obj.cta_variants)

    ctas_pretty.short_description = "CTA Variants"

    def hashtags_pretty(self, obj: ContentPackage) -> str:
        return self._render_json_block(obj.hashtags)

    hashtags_pretty.short_description = "Hashtags"

    def carousel_pretty(self, obj: ContentPackage) -> str:
        return self._render_json_block(obj.carousel_outline)

    carousel_pretty.short_description = "Carousel Outline"

    def validation_pretty(self, obj: ContentPackage) -> str:
        return self._render_json_block(obj.validation_report)

    validation_pretty.short_description = "Validation Report"

    def _render_json_block(self, value) -> str:
        return format_html(
            "<pre style='white-space: pre-wrap; margin: 0;'>{}</pre>",
            json.dumps(value, ensure_ascii=False, indent=2),
        )
