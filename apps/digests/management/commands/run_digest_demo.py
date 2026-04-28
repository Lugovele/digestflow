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

        self.stdout.write(self.style.SUCCESS(f"DigestRun {run.id} finished with status={run.status}"))
