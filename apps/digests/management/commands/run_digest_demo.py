import logging

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.digests.models import DigestRun
from apps.topics.models import Topic
from services.json_utils import make_json_safe
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import get_demo_articles_for_topic
from services.sources.rss_adapter import fetch_rss_articles

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Создает DigestRun для Topic и выполняет базовый demo pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--topic-id", type=int, required=True, help="ID темы Topic")
        parser.add_argument(
            "--demo",
            action="store_true",
            help="Явно использовать demo source вместо RSS.",
        )
        parser.add_argument(
            "--rss-url",
            type=str,
            help="Явно проверить указанный RSS feed через основной pipeline.",
        )

    def handle(self, *args, **options):
        topic_id = options["topic_id"]
        use_demo = options["demo"]
        rss_url = (options.get("rss_url") or "").strip() or None

        try:
            topic = Topic.objects.get(pk=topic_id)
        except Topic.DoesNotExist as exc:
            raise CommandError(f"Topic with id={topic_id} does not exist.") from exc

        input_mode = "demo" if use_demo else "auto_source"
        if rss_url:
            input_mode = "rss_url_override"

        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot=make_json_safe({
                "mode": input_mode,
                "source": "management_command",
                "rss_url": rss_url or "",
            }),
        )

        if rss_url:
            self.stdout.write(f"Trying RSS feed: {rss_url}")
            raw_items = fetch_rss_articles(rss_url)
            self.stdout.write(f"RSS fetched items count: {len(raw_items)}")
            logger.info("RSS override items count: %s", len(raw_items))

            if raw_items:
                self.stdout.write(
                    self.style.SUCCESS(f"Loaded {len(raw_items)} RSS items from override URL.")
                )
            elif use_demo:
                self.stdout.write(
                    self.style.WARNING(
                        "RSS override returned 0 items. Falling back to demo source because --demo was provided."
                    )
                )
                raw_items = get_demo_articles_for_topic(topic.name)
            else:
                run.status = DigestRun.STATUS_FAILED
                run.error_message = f"RSS override returned no valid items: {rss_url}"
                run.finished_at = timezone.now()
                run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
                self.stdout.write(self.style.WARNING(f"DigestRun {run.id} finished with status={run.status}"))
                self.stdout.write(self.style.WARNING(f"error_message: {run.error_message}"))
                self.stdout.write("")
                self.stdout.write("=== RUN SUMMARY ===")
                self.stdout.write(f"run_id: {run.id}")
                self.stdout.write(f"status: {run.status}")
                self.stdout.write(f"topic: {run.topic.name}")
                self.stdout.write("digest_id: null")
                self.stdout.write("content_package_id: null")
                self.stdout.write("article_ids: []")
                self.stdout.write("total_tokens: null")
                self.stdout.write("total_estimated_cost: null")
                self.stdout.write("used_mock: False")
                self.stdout.write(f"error_message: {run.error_message}")
                return
        elif use_demo:
            raw_items = get_demo_articles_for_topic(topic.name)
        else:
            raw_items = None

        run_digest_pipeline(run.id, raw_items=raw_items)
        run.refresh_from_db()

        digest = getattr(run, "digest", None)
        content_package = getattr(digest, "content_package", None) if digest else None

        digest_stage_metrics = run.metrics.get("digest_stage", {}) if isinstance(run.metrics, dict) else {}
        packaging_stage_metrics = (
            run.metrics.get("packaging_stage", {}) if isinstance(run.metrics, dict) else {}
        )
        source_stage_metrics = run.metrics.get("source_stage", {}) if isinstance(run.metrics, dict) else {}
        digest_tokens = digest_stage_metrics.get("tokens", {}) if isinstance(digest_stage_metrics, dict) else {}
        packaging_tokens = (
            packaging_stage_metrics.get("tokens", {}) if isinstance(packaging_stage_metrics, dict) else {}
        )
        used_mock = bool(
            digest_stage_metrics.get("is_mock") or packaging_stage_metrics.get("is_mock")
        )
        total_tokens = _sum_metric_values(
            digest_tokens.get("total"),
            packaging_tokens.get("total"),
        )
        total_estimated_cost = _sum_metric_values(
            digest_stage_metrics.get("estimated_cost_usd"),
            packaging_stage_metrics.get("estimated_cost_usd"),
        )

        self.stdout.write(self.style.SUCCESS(f"DigestRun {run.id} finished with status={run.status}"))
        if run.error_message:
            self.stdout.write(self.style.WARNING(f"error_message: {run.error_message}"))

        self.stdout.write("")
        self.stdout.write("=== RUN SUMMARY ===")
        self.stdout.write(f"run_id: {run.id}")
        self.stdout.write(f"status: {run.status}")
        self.stdout.write(f"topic: {run.topic.name}")
        self.stdout.write(f"digest_id: {digest.id if digest else 'null'}")
        self.stdout.write(
            f"content_package_id: {content_package.id if content_package else 'null'}"
        )
        self.stdout.write(
            f"article_ids: {source_stage_metrics.get('article_ids', [])}"
        )
        self.stdout.write(f"total_tokens: {total_tokens if total_tokens is not None else 'null'}")
        self.stdout.write(
            "total_estimated_cost: "
            f"{total_estimated_cost if total_estimated_cost is not None else 'null'}"
        )
        self.stdout.write(f"used_mock: {used_mock}")
        self.stdout.write(f"error_message: {run.error_message or 'null'}")
        if digest_stage_metrics:
            self.stdout.write(
                "digest_provider: "
                f"{digest_stage_metrics.get('provider', 'null')}"
            )
        if packaging_stage_metrics:
            self.stdout.write(
                "packaging_provider: "
                f"{packaging_stage_metrics.get('provider', 'null')}"
            )


def _sum_metric_values(*values):
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return round(sum(present_values), 6)
