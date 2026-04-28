import json

from django.core.management.base import BaseCommand, CommandError

from apps.topics.models import Topic
from services.sources import get_demo_articles_for_topic


class Command(BaseCommand):
    help = "Показывает demo articles для выбранной Topic без запуска полного pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--topic-id", type=int, required=True, help="ID темы Topic")

    def handle(self, *args, **options):
        topic_id = options["topic_id"]

        try:
            topic = Topic.objects.get(pk=topic_id)
        except Topic.DoesNotExist as exc:
            raise CommandError(f"Topic with id={topic_id} does not exist.") from exc

        articles = get_demo_articles_for_topic(topic.name)

        self.stdout.write(self.style.SUCCESS(f"Loaded {len(articles)} demo articles for topic '{topic.name}'"))
        self.stdout.write(json.dumps(articles, ensure_ascii=False, indent=2))
