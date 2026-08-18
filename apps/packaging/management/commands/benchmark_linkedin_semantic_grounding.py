from __future__ import annotations

import json
from pathlib import Path
import sys

from django.core.management.base import BaseCommand, CommandError

from services.packaging.linkedin_post_semantic_grounding_benchmark import (
    DEFAULT_EXPERIMENT_ID,
    SemanticGroundingBenchmarkPlan,
    SEMANTIC_GROUNDING_EXECUTION_PROFILE_PROVIDER_DEFAULT,
    SemanticGroundingBenchmarkRequest,
    default_semantic_grounding_benchmark_cases,
    default_semantic_grounding_benchmark_plans,
    load_semantic_grounding_benchmark_case,
    run_semantic_grounding_benchmark,
    semantic_grounding_benchmark_max_output_tokens_for_provider,
)


class Command(BaseCommand):
    help = "Run an isolated Semantic Grounding benchmark. Live mode requires --allow-api."
    requires_system_checks: list[str] = []

    def add_arguments(self, parser):
        parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
        parser.add_argument("--case", action="append", default=[])
        parser.add_argument("--plan", action="append", default=[])
        parser.add_argument("--runs-per-plan", type=int, default=1)
        parser.add_argument("--output-root")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument(
            "--allow-api",
            action="store_true",
            help="Explicitly allow live Semantic Grounding provider execution.",
        )

    def handle(self, *args, **options):
        if options["dry_run"] and options["allow_api"]:
            raise CommandError("--dry-run and --allow-api cannot be used together")
        cases = tuple(load_semantic_grounding_benchmark_case(Path(path)) for path in options["case"])
        if not cases:
            cases = default_semantic_grounding_benchmark_cases()
        plans = tuple(_parse_plan(value) for value in options["plan"])
        if not plans:
            plans = default_semantic_grounding_benchmark_plans()

        request = SemanticGroundingBenchmarkRequest(
            experiment_id=options["experiment_id"],
            cases=cases,
            plans=plans,
            runs_per_plan=options["runs_per_plan"],
            allow_api=bool(options["allow_api"]),
            output_root=Path(options["output_root"]) if options.get("output_root") else None,
        )
        result = run_semantic_grounding_benchmark(request)
        self.stdout.write("=== POSTFLOW SEMANTIC GROUNDING BENCHMARK ===")
        self.stdout.write(f"experiment_id: {result.experiment_id}")
        self.stdout.write(f"status: {result.status}")
        self.stdout.write(f"exit_code: {result.exit_code}")
        self.stdout.write(f"run_count: {result.run_count}")
        self.stdout.write(f"dry_run: {not request.allow_api}")
        self.stdout.write(f"allow_api: {request.allow_api}")
        self.stdout.write(f"provider_calls: {result.provider_call_count}")
        self.stdout.write("candidate_writer_invocations: 0")
        self.stdout.write("quality_evaluator_invocations: 0")
        self.stdout.write("repair_writer_invocations: 0")
        self.stdout.write("publication_packaging_invocations: 0")
        if result.safe_failure_code:
            self.stdout.write(f"safe_failure_code: {result.safe_failure_code}")
        if result.safe_failure_message:
            self.stdout.write(f"safe_failure_message: {result.safe_failure_message}")
        if result.artifacts:
            self.stdout.write("")
            self.stdout.write("=== ARTIFACTS ===")
            self.stdout.write(json.dumps(result.artifacts.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        self.stdout.write("")
        self.stdout.write("=== SANITIZED RESULT JSON ===")
        self.stdout.write(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        if result.exit_code:
            sys.exit(result.exit_code)


def _parse_plan(value: str) -> SemanticGroundingBenchmarkPlan:
    if "=" not in value:
        raise CommandError("--plan must use plan_id=provider,model")
    plan_id, raw_fields = value.split("=", 1)
    fields = [field.strip() for field in raw_fields.split(",")]
    if len(fields) not in (2, 3):
        raise CommandError("--plan must include provider,model[,execution_profile]")
    provider = fields[0]
    return SemanticGroundingBenchmarkPlan(
        plan_id=plan_id.strip(),
        provider=provider,
        model=fields[1],
        max_output_tokens=semantic_grounding_benchmark_max_output_tokens_for_provider(
            provider
        ),
        execution_profile=(
            fields[2]
            if len(fields) == 3
            else SEMANTIC_GROUNDING_EXECUTION_PROFILE_PROVIDER_DEFAULT
        ),
    )
