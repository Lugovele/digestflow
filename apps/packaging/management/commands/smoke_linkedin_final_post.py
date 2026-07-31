from __future__ import annotations

import json
from pathlib import Path
import sys

from django.core.management.base import BaseCommand

from services.packaging.linkedin_post_final_post_smoke_runner import (
    DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
    DEFAULT_QUALITY_MAX_OUTPUT_TOKENS,
    SMOKE_MODE_CONTROLLED_REPAIR,
    SMOKE_MODE_STANDALONE,
    FinalPostSmokeRunRequest,
    run_final_post_smoke,
)


class Command(BaseCommand):
    help = "Manual PostFlow final-post smoke run. No production runtime wiring or persistence."
    requires_system_checks: list[str] = []

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            required=True,
            help="Path to a local final-post smoke input JSON file.",
        )
        parser.add_argument(
            "--mode",
            choices=(SMOKE_MODE_STANDALONE, SMOKE_MODE_CONTROLLED_REPAIR),
            default=SMOKE_MODE_STANDALONE,
            help="Smoke flow to run.",
        )
        parser.add_argument(
            "--allow-api",
            action="store_true",
            help="Allow configured provider calls. Omit for dry-run validation only.",
        )
        parser.add_argument(
            "--save-output",
            action="store_true",
            help="Save sanitized output under debug_outputs/final_post_smoke_runs/.",
        )
        parser.add_argument(
            "--include-raw-responses",
            action="store_true",
            help="Include raw model text only. Provider raw metadata remains hidden.",
        )
        parser.add_argument(
            "--expect-repair",
            action="store_true",
            help="Return scenario-mismatch exit code if controlled repair is not executed.",
        )
        parser.add_argument("--candidate-provider")
        parser.add_argument("--candidate-model")
        parser.add_argument("--grounding-provider")
        parser.add_argument("--grounding-model")
        parser.add_argument("--evaluator-provider")
        parser.add_argument("--evaluator-model")
        parser.add_argument("--repair-provider")
        parser.add_argument("--repair-model")
        parser.add_argument("--candidate-max-output-tokens", type=int, default=1200)
        parser.add_argument(
            "--grounding-max-output-tokens",
            type=int,
            default=DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
        )
        parser.add_argument(
            "--evaluator-max-output-tokens",
            type=int,
            default=DEFAULT_QUALITY_MAX_OUTPUT_TOKENS,
        )
        parser.add_argument("--repair-max-output-tokens", type=int, default=1200)

    def handle(self, *args, **options):
        request = FinalPostSmokeRunRequest(
            input_path=Path(options["input"]),
            mode=options["mode"],
            allow_api=bool(options["allow_api"]),
            save_output=bool(options["save_output"]),
            include_raw_responses=bool(options["include_raw_responses"]),
            expect_repair=bool(options["expect_repair"]),
            candidate_provider=options.get("candidate_provider"),
            candidate_model=options.get("candidate_model"),
            semantic_grounding_provider=options.get("grounding_provider"),
            semantic_grounding_model=options.get("grounding_model"),
            quality_evaluator_provider=options.get("evaluator_provider"),
            quality_evaluator_model=options.get("evaluator_model"),
            repair_provider=options.get("repair_provider"),
            repair_model=options.get("repair_model"),
            candidate_max_output_tokens=options["candidate_max_output_tokens"],
            semantic_grounding_max_output_tokens=options[
                "grounding_max_output_tokens"
            ],
            quality_evaluator_max_output_tokens=options[
                "evaluator_max_output_tokens"
            ],
            repair_max_output_tokens=options["repair_max_output_tokens"],
        )
        result = run_final_post_smoke(request)
        self.stdout.write("=== POSTFLOW FINAL POST SMOKE ===")
        self.stdout.write(f"mode: {result.mode}")
        self.stdout.write(f"status: {result.status}")
        self.stdout.write(f"exit_code: {result.exit_code}")
        self.stdout.write(f"manual_smoke_only: {True}")
        self.stdout.write(f"production_runtime_wiring: {False}")
        self.stdout.write(f"input_path: {result.input_path}")
        self.stdout.write("")
        self.stdout.write("=== PROVIDER / MODEL ===")
        self.stdout.write(
            json.dumps(result.provider_models, indent=2, sort_keys=True, allow_nan=False)
        )
        self.stdout.write("")
        self.stdout.write("=== INVOCATION BUDGET ===")
        self.stdout.write(
            json.dumps(result.invocation_budget, indent=2, sort_keys=True, allow_nan=False)
        )
        self.stdout.write("")
        self.stdout.write("=== INVOCATION COUNTS ===")
        self.stdout.write(
            json.dumps(result.invocation_counts, indent=2, sort_keys=True, allow_nan=False)
        )
        self.stdout.write("")
        self.stdout.write("=== OUTCOME ===")
        self.stdout.write(f"dry_run: {result.dry_run}")
        self.stdout.write(f"repair_enabled: {result.repair_enabled}")
        self.stdout.write(f"repair_executed: {result.repair_executed}")
        self.stdout.write(f"initial_outcome: {result.initial_outcome}")
        self.stdout.write(f"final_outcome: {result.final_outcome}")
        self.stdout.write(f"accepted: {result.accepted}")
        self.stdout.write(f"deterministic_gate_passed: {result.deterministic_gate_passed}")
        self.stdout.write(f"quality_passed: {result.quality_passed}")
        if result.safe_failure_code:
            self.stdout.write(f"safe_failure_code: {result.safe_failure_code}")
        if result.safe_failure_message:
            self.stdout.write(f"safe_failure_message: {result.safe_failure_message}")
        self.stdout.write("")
        self.stdout.write("=== FINAL POST TEXT ===")
        self.stdout.write(result.final_post_text)
        if result.saved_output_path:
            self.stdout.write("")
            self.stdout.write(f"saved_output: {result.saved_output_path}")
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
