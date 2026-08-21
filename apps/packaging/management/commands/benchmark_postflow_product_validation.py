from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from django.core.management.base import BaseCommand

from services.packaging.postflow_product_validation_corpus import DEFAULT_MANIFEST_PATH
from services.packaging.postflow_product_validation_runner import (
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_OUTPUT_ROOT,
    ProductValidationRequest,
    run_product_validation_benchmark,
)


class Command(BaseCommand):
    help = "Run PostFlow product validation benchmark infrastructure."
    requires_system_checks: list[str] = []

    def add_arguments(self, parser):
        parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
        parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
        parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
        parser.add_argument("--compare-to")
        execution_mode = parser.add_mutually_exclusive_group()
        execution_mode.add_argument(
            "--dry-run",
            action="store_true",
            default=True,
            help="Plan and write artifacts without provider/API calls (default).",
        )
        execution_mode.add_argument(
            "--allow-api",
            action="store_true",
            default=False,
            help="Explicitly enable live provider/API calls for benchmark execution.",
        )
        parser.add_argument(
            "--export-human-review",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Write human review CSV/Markdown artifacts (default: enabled).",
        )
        parser.add_argument(
            "--fail-on-corpus-invalid",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Return a sanitized config_error result for invalid corpus inputs (default: enabled).",
        )

    def handle(self, *args, **options):
        request = ProductValidationRequest(
            experiment_id=options["experiment_id"],
            manifest_path=Path(options["manifest"]),
            output_root=Path(options["output_root"]),
            compare_to=Path(options["compare_to"]) if options.get("compare_to") else None,
            export_human_review=bool(options["export_human_review"]),
            fail_on_corpus_invalid=bool(options["fail_on_corpus_invalid"]),
            allow_api=bool(options["allow_api"]),
            dry_run=not bool(options["allow_api"]),
        )
        result = run_product_validation_benchmark(request)
        self.stdout.write("=== PostFlow PRODUCT VALIDATION BENCHMARK ===")
        self.stdout.write(f"experiment_id: {result.experiment_id}")
        self.stdout.write(f"status: {result.status}")
        self.stdout.write(f"exit_code: {result.exit_code}")
        self.stdout.write(f"corpus_case_count: {result.corpus_case_count}")
        self.stdout.write(f"run_count: {result.run_count}")
        counts = result.metrics.get("provider_invocation_counts", {})
        self.stdout.write(f"provider_calls: {result.provider_call_count}")
        self.stdout.write(
            "candidate_writer_provider_calls: "
            f"{counts.get('candidate_writer_provider_api_calls', 0)}"
        )
        self.stdout.write(
            "semantic_grounding_provider_calls: "
            f"{counts.get('semantic_grounding_provider_api_calls', 0)}"
        )
        self.stdout.write(
            "quality_evaluator_provider_calls: "
            f"{counts.get('quality_evaluator_provider_api_calls', 0)}"
        )
        self.stdout.write(
            "repair_writer_provider_calls: "
            f"{counts.get('repair_writer_provider_api_calls', 0)}"
        )
        self.stdout.write(
            "publication_packaging_invocations: "
            f"{counts.get('publication_packaging_invocations', 0)}"
        )
        if result.safe_failure_code:
            self.stdout.write(f"safe_failure_code: {result.safe_failure_code}")
        if result.safe_failure_message:
            self.stdout.write(f"safe_failure_message: {result.safe_failure_message}")
        if result.artifacts:
            self.stdout.write("")
            self.stdout.write("=== ARTIFACTS ===")
            self.stdout.write(
                json.dumps(
                    result.artifacts.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        self.stdout.write("")
        self.stdout.write("=== SANITIZED RESULT JSON ===")
        self.stdout.write(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        if result.exit_code:
            sys.exit(result.exit_code)
