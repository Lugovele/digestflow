from django.contrib import admin

from .models import ContentPackage


@admin.register(ContentPackage)
class ContentPackageAdmin(admin.ModelAdmin):
    list_display = ("id", "digest", "created_at", "updated_at")
    search_fields = ("digest__title", "post_text")
    readonly_fields = (
        "hook_variants",
        "cta_variants",
        "hashtags",
        "carousel_outline",
        "validation_report",
    )
