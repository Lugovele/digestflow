import json

from django.core.management.base import BaseCommand

from services.sources.rss_adapter import fetch_rss_articles, get_rss_debug_snapshot


class Command(BaseCommand):
    help = "Показывает RSS articles для указанного URL без записи в базу."

    def add_arguments(self, parser):
        parser.add_argument("--url", type=str, required=True, help="RSS feed URL")
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Показать сырые RSS entry и причины фильтрации.",
        )

    def handle(self, *args, **options):
        feed_url = options["url"].strip()
        debug_mode = options["debug"]

        if debug_mode:
            debug_snapshot = get_rss_debug_snapshot(feed_url)
            self.stdout.write("=== RSS DEBUG ===")
            self.stdout.write(f"feed_url: {debug_snapshot['feed_url']}")
            self.stdout.write(f"feed_title: {debug_snapshot['feed_title'] or '<empty>'}")
            self.stdout.write(f"total_entries_before_filtering: {debug_snapshot['total_entries']}")
            self.stdout.write(f"bozo: {debug_snapshot.get('bozo')}")
            self.stdout.write(f"http_status: {debug_snapshot.get('status')}")
            self.stdout.write(f"resolved_href: {debug_snapshot.get('href')}")
            if debug_snapshot.get("bozo_exception"):
                self.stdout.write(f"bozo_exception: {debug_snapshot['bozo_exception']}")
            if debug_snapshot.get("skip_reason"):
                self.stdout.write(f"feed_skip_reason: {debug_snapshot['skip_reason']}")

            for index, entry in enumerate(debug_snapshot["entries"], start=1):
                self.stdout.write("")
                self.stdout.write(f"[entry {index}]")
                self.stdout.write(
                    json.dumps(
                        {
                            "available_keys": entry.get("available_keys"),
                            "raw_title": entry.get("raw_title"),
                            "raw_link": entry.get("raw_link"),
                            "raw_id": entry.get("raw_id"),
                            "raw_summary": entry.get("raw_summary"),
                            "raw_description": entry.get("raw_description"),
                            "raw_published": entry.get("raw_published"),
                            "skip_reason": entry.get("skip_reason"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            self.stdout.write("")

        articles = fetch_rss_articles(feed_url)

        self.stdout.write(self.style.SUCCESS(f"Loaded {len(articles)} RSS items from {feed_url}"))

        if not articles:
            self.stdout.write("No valid RSS items found.")
            return

        self.stdout.write("")
        self.stdout.write("=== RSS PREVIEW ===")
        for index, article in enumerate(articles[:5], start=1):
            preview_payload = {
                "title": article.get("title"),
                "url": article.get("url"),
                "source_name": article.get("source_name"),
                "snippet": article.get("snippet"),
                "published_at": article.get("published_at"),
            }
            self.stdout.write(f"[{index}]")
            self.stdout.write(json.dumps(preview_payload, ensure_ascii=False, indent=2))
            self.stdout.write("")
