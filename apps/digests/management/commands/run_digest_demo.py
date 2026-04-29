from django.core.management.base import BaseCommand, CommandError

from apps.digests.models import DigestRun
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import get_demo_articles_for_topic


class Command(BaseCommand):
    help = "Создает DigestRun для Topic и выполняет базовый demo pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--topic-id", type=int, required=True, help="ID темы Topic")

    def handle(self, *args, **options):
        topic_id = options["topic_id"]

        try:
            topic = Topic.objects.get(pk=topic_id)
        except Topic.DoesNotExist as exc:
            raise CommandError(f"Topic with id={topic_id} does not exist.") from exc

        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={
                "mode": "demo",
                "source": "management_command",
            },
        )

        raw_items = get_demo_articles_for_topic(topic.name)

        run_digest_pipeline(run.id, raw_items)
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
