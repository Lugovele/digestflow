import json

from django.core.management.base import BaseCommand

from services.ai import run_digest_smoke_test


class Command(BaseCommand):
    help = "Запускает ранний AI smoke test для digest generation."

    def add_arguments(self, parser):
        parser.add_argument(
            "--topic",
            type=str,
            default="AI product strategy",
            help="Текстовое имя темы для smoke test.",
        )

    def handle(self, *args, **options):
        topic_name = options["topic"].strip()
        result = run_digest_smoke_test(topic_name)

        if result.validation_passed:
            self.stdout.write(self.style.SUCCESS("AI smoke test completed."))
        else:
            self.stdout.write(self.style.WARNING("AI smoke test completed with controlled failure."))
        self.stdout.write("")
        self.stdout.write("=== PROVIDER ===")
        self.stdout.write(result.provider)
        self.stdout.write("")
        self.stdout.write("=== IS MOCK ===")
        self.stdout.write(str(result.is_mock))

        if result.fallback_reason:
            self.stdout.write("")
            self.stdout.write("=== FALLBACK REASON ===")
            self.stdout.write(result.fallback_reason)

        self.stdout.write("")
        self.stdout.write("=== PROMPT ===")
        self.stdout.write(result.prompt)
        self.stdout.write("")
        self.stdout.write("=== RESPONSE TEXT ===")
        self.stdout.write(result.response_text)
        self.stdout.write("")
        self.stdout.write("=== VALIDATION PASSED ===")
        self.stdout.write(str(result.validation_passed))
        if result.error_message:
            self.stdout.write("")
            self.stdout.write("=== ERROR MESSAGE ===")
            self.stdout.write(result.error_message)
        if result.payload is not None:
            self.stdout.write("")
            self.stdout.write("=== PARSED PAYLOAD ===")
            self.stdout.write(json.dumps(result.payload, ensure_ascii=False, indent=2))
