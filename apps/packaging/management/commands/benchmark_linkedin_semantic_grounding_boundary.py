from __future__ import annotations

import json
from pathlib import Path
import sys

from django.core.management.base import BaseCommand, CommandError

from services.packaging.linkedin_post_semantic_grounding_boundary_benchmark import (
    DEFAULT_EXPERIMENT_ID,
    SemanticGroundingBoundaryExecutionPolicy,
    SemanticGroundingBoundaryPlan,
    SemanticGroundingBoundaryRequest,
    default_semantic_grounding_boundary_plans,
    load_semantic_grounding_boundary_cases,
    run_semantic_grounding_boundary_benchmark,
)


class Command(BaseCommand):
    help = (
        "Run the Semantic Grounding boundary benchmark. "
        "Live mode requires --allow-api."
    )
    requires_system_checks: list[str] = []

    def add_arguments(self, parser):
        parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
        parser.add_argument("--fixture")
        parser.add_argument("--plan", action="append", default=[])
        parser.add_argument("--runs-per-plan", type=int, default=1)
        parser.add_argument("--output-root")
        parser.add_argument("--retry-failed-from")
        parser.add_argument("--inter-call-delay-seconds", type=float, default=0.0)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument(
            "--allow-api",
            action="store_true",
            help="Explicitly allow live Semantic Grounding provider execution.",
        )

    def handle(self, *args, **options):
        if options["dry_run"] and options["allow_api"]:
            raise CommandError("--dry-run and --allow-api cannot be used together")

        cases = (
            load_semantic_grounding_boundary_cases(Path(options["fixture"]))
            if options.get("fixture")
            else load_semantic_grounding_boundary_cases()
        )
        plans = tuple(_parse_plan(value) for value in options["plan"])
        if not plans:
            plans = default_semantic_grounding_boundary_plans()

        request = SemanticGroundingBoundaryRequest(
            experiment_id=options["experiment_id"],
            cases=cases,
            plans=plans,
            runs_per_plan=options["runs_per_plan"],
            allow_api=bool(options["allow_api"]),
            output_root=Path(options["output_root"]) if options.get("output_root") else None,
            retry_failed_from=(
                Path(options["retry_failed_from"])
                if options.get("retry_failed_from")
                else None
            ),
            execution_policy=SemanticGroundingBoundaryExecutionPolicy(
                inter_call_delay_seconds=options["inter_call_delay_seconds"],
            ),
        )
        result = run_semantic_grounding_boundary_benchmark(request)
        self.stdout.write("=== POSTFLOW SEMANTIC GROUNDING BOUNDARY BENCHMARK ===")
        self.stdout.write(f"experiment_id: {result.experiment_id}")
        self.stdout.write(f"status: {result.status}")
        self.stdout.write(f"exit_code: {result.exit_code}")
        self.stdout.write(f"run_count: {result.run_count}")
        self.stdout.write(f"dry_run: {not request.allow_api}")
        self.stdout.write(f"allow_api: {request.allow_api}")
        self.stdout.write(f"provider_calls: {result.provider_call_count}")
        self.stdout.write(
            f"planned_live_provider_calls: {result.planned_provider_call_count}"
        )
        if request.retry_failed_from:
            self.stdout.write(f"retry_failed_from: {request.retry_failed_from}")
        self.stdout.write(
            "inter_call_delay_seconds: "
            f"{request.execution_policy.inter_call_delay_seconds}"
        )
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


def _parse_plan(value: str) -> SemanticGroundingBoundaryPlan:
    if "=" not in value:
        raise CommandError("--plan must use plan_id=provider,model,max_output_tokens")
    plan_id, raw_fields = value.split("=", 1)
    fields = [field.strip() for field in raw_fields.split(",")]
    if len(fields) not in {2, 3}:
        raise CommandError("--plan must include provider,model[,max_output_tokens]")
    try:
        max_output_tokens = (
            int(fields[2])
            if len(fields) == 3
            else _default_budget_for_provider(fields[0])
        )
    except ValueError as exc:
        raise CommandError("--plan max_output_tokens must be an integer") from exc
    return SemanticGroundingBoundaryPlan(
        plan_id=plan_id.strip(),
        provider=fields[0],
        model=fields[1],
        max_output_tokens=max_output_tokens,
    )


def _default_budget_for_provider(provider: str) -> int:
    for plan in default_semantic_grounding_boundary_plans():
        if plan.provider == provider:
            return plan.max_output_tokens
    return 2400
