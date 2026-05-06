import json

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.digests.models import DigestRun
from apps.topics.models import Topic
from services.digests import generate_digest_for_run
from services.sources import get_demo_articles_for_topic


class Command(BaseCommand):
    help = "Создает DigestRun и выполняет только AI digest stage без packaging."

    def add_arguments(self, parser):
        parser.add_argument("--topic-id", type=int, required=True, help="ID темы Topic")

    def handle(self, *args, **options):
        topic_id = options["topic_id"]

        try:
            topic = Topic.objects.get(pk=topic_id)
        except Topic.DoesNotExist as exc:
            raise CommandError(f"Topic with id={topic_id} does not exist.") from exc

        articles = get_demo_articles_for_topic(topic.name)
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={
                "mode": "digest_stage_only",
                "source": "management_command",
                "articles_count": len(articles),
            },
            metrics={"source_stage": {"articles_count": len(articles)}},
        )

        self.stdout.write(self.style.SUCCESS(f"[DigestRun {run.id}] Topic loaded: {topic.name}"))
        self.stdout.write(self.style.SUCCESS(f"[DigestRun {run.id}] Demo articles loaded: {len(articles)}"))

        try:
            digest, debug_info = generate_digest_for_run(run, articles)
            run.status = DigestRun.STATUS_COMPLETED
            run.finished_at = timezone.now()
            run.save(update_fields=["status", "finished_at", "updated_at"])
        except Exception as exc:
            run.status = DigestRun.STATUS_FAILED
            run.error_message = str(exc)
            run.finished_at = timezone.now()
            run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
            raise

        articles = digest.get_articles()

        self.stdout.write("")
        self.stdout.write("=== DIGEST ===")
        self.stdout.write(f"title: {digest.get_payload_title()}")
        self.stdout.write(f"articles: {json.dumps(articles, ensure_ascii=False)}")
        self.stdout.write("")
        self.stdout.write("=== DEBUG ===")
        self.stdout.write(f"provider: {debug_info['provider']}")
        self.stdout.write(f"is_mock: {debug_info['is_mock']}")
        if debug_info["fallback_reason"]:
            self.stdout.write(f"fallback_reason: {debug_info['fallback_reason']}")
        self.stdout.write("")
        self.stdout.write("=== PROMPT ===")
        self.stdout.write(debug_info["prompt"])
        self.stdout.write("")
        self.stdout.write("=== RESPONSE TEXT ===")
        self.stdout.write(debug_info["response_text"])
