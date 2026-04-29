from django.contrib import admin

from .models import Article


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "topic", "source_name", "published_at", "created_at")
    list_filter = ("source_name", "created_at")
    search_fields = ("title", "url", "snippet", "topic__name")
