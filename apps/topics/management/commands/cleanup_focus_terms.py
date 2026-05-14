from django.core.management.base import BaseCommand, CommandError

from apps.topics.focus import clean_focus_terms, is_meaningful_focus_term
from apps.topics.models import Topic


DEFAULT_NOISE_TERMS = {
    "pumpumpum",
    "гне",
    "еакноедрп",
    "еакноеарп",
    "ить",
    "итиь",
    "оалро",
    "767ghjb;k",
}


class Command(BaseCommand):
    help = "Remove invalid or explicitly listed noisy focus terms from a topic."

    def add_arguments(self, parser):
        parser.add_argument("--topic-id", type=int, required=True)
        parser.add_argument(
            "--remove",
            action="append",
            default=[],
            help="Exact focus term to remove. Can be passed multiple times.",
        )

    def handle(self, *args, **options):
        topic_id = options["topic_id"]
        topic = Topic.objects.filter(id=topic_id).first()
        if topic is None:
            raise CommandError(f"Topic {topic_id} was not found.")

        explicit_terms = {str(term).strip().casefold() for term in options["remove"] if str(term).strip()}
        removal_terms = DEFAULT_NOISE_TERMS | explicit_terms

        original_terms = clean_focus_terms(topic.keywords if isinstance(topic.keywords, list) else [])
        cleaned_terms = [
            term for term in original_terms
            if term.casefold() not in removal_terms and is_meaningful_focus_term(term)
        ]

        if cleaned_terms == original_terms:
            self.stdout.write(self.style.WARNING("No focus terms needed cleanup."))
            return

        topic.keywords = cleaned_terms
        topic.save(update_fields=["keywords", "updated_at"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Cleaned focus terms for topic {topic.id}. Remaining terms: {cleaned_terms}"
            )
        )
