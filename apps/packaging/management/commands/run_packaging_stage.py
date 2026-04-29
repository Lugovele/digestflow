import json

from django.core.management.base import BaseCommand, CommandError

from apps.digests.models import Digest
from services.packaging import generate_content_package_for_digest


class Command(BaseCommand):
    help = "Выполняет только LinkedIn packaging stage для готового Digest."

    def add_arguments(self, parser):
        parser.add_argument("--digest-id", type=int, required=True, help="ID готового Digest")

    def handle(self, *args, **options):
        digest_id = options["digest_id"]

        try:
            digest = Digest.objects.select_related("run__topic").get(pk=digest_id)
        except Digest.DoesNotExist as exc:
            raise CommandError(f"Digest with id={digest_id} does not exist.") from exc

        self.stdout.write(self.style.SUCCESS(f"[Digest {digest.id}] Digest loaded: {digest.title}"))

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.stdout.write("")
        self.stdout.write("=== CONTENT PACKAGE ===")
        self.stdout.write(f"post_text: {content_package.post_text}")
        self.stdout.write(f"hook_variants: {json.dumps(content_package.hook_variants, ensure_ascii=False)}")
        self.stdout.write(f"cta_variants: {json.dumps(content_package.cta_variants, ensure_ascii=False)}")
        self.stdout.write(f"hashtags: {json.dumps(content_package.hashtags, ensure_ascii=False)}")
        self.stdout.write(
            f"carousel_outline: {json.dumps(content_package.carousel_outline, ensure_ascii=False)}"
        )
        self.stdout.write("")
        self.stdout.write("=== DEBUG ===")
        self.stdout.write(f"provider: {debug_info['provider']}")
        self.stdout.write(f"is_mock: {debug_info['is_mock']}")
        if debug_info["fallback_reason"]:
            self.stdout.write(f"fallback_reason: {debug_info['fallback_reason']}")
        self.stdout.write(
            f"validation_report: {json.dumps(debug_info['validation_report'], ensure_ascii=False)}"
        )
        self.stdout.write("")
        self.stdout.write("=== PROMPT ===")
        self.stdout.write(debug_info["prompt"])
        self.stdout.write("")
        self.stdout.write("=== RESPONSE TEXT ===")
        self.stdout.write(debug_info["response_text"])
