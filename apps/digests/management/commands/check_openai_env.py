from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Проверяет, читает ли проект OPENAI_API_KEY из .env, не выводя сам ключ."

    def handle(self, *args, **options):
        api_key = (settings.OPENAI_API_KEY or "").strip()

        found = bool(api_key)
        starts_with_sk = api_key.startswith("sk-") if api_key else False
        length = len(api_key)

        self.stdout.write(f"OPENAI_API_KEY found: {found}")
        self.stdout.write(f"starts_with_sk: {starts_with_sk}")
        self.stdout.write(f"length: {length}")
