from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.ai.client import OpenAIClient
from services.packaging.linkedin_final_post_diagnostics import (
    FinalPostDiagnostics,
    diagnose_final_post_payload,
)
from services.packaging.linkedin_post_pipeline import (
    FinalPostPayload,
    build_angle_decision_from_contextual_evidence_pack,
    build_article_evidence_pack_from_pipeline_input,
    build_contextual_evidence_pack_from_article_evidence_pack,
    build_pipeline_input_from_digest,
    build_post_brief_from_angle_decision,
    validate_final_post_payload,
)
from services.packaging.validators import validate_content_package_payload


PROMPT_PATH = Path(settings.BASE_DIR) / "prompts" / "linkedin" / "final_post_from_brief.txt"
PLACEHOLDER_API_KEYS = {"", "sk-your-key", "your-openai-api-key-here"}


class Command(BaseCommand):
    help = "Debug-only final post generation from a PostBrief fixture. Does not save runtime data."
    requires_system_checks: list[str] = []

    def add_arguments(self, parser):
        parser.add_argument("--fixture", required=True, help="Path to a LinkedIn post fixture JSON file.")
        parser.add_argument("--allow-api", action="store_true", help="Allow the command to call the OpenAI API.")
        parser.add_argument("--save-output", action="store_true", help="Save debug output under debug_outputs/.")
        parser.add_argument("--max-output-tokens", type=int, default=1200)

    def handle(self, *args, **options):
        fixture_path = Path(options["fixture"]).resolve()
        allow_api = bool(options["allow_api"])
        save_output = bool(options["save_output"])
        max_output_tokens = int(options["max_output_tokens"])

        fixture = _load_fixture(fixture_path)
        angle_decision, post_brief = _build_post_brief_from_fixture(fixture)
        selected_evidence_ids = [item.evidence_id for item in post_brief.evidence_to_use]
        prompt_text = _load_prompt_contract()
        execution_prompt = _build_execution_prompt(
            prompt_text,
            fixture=fixture,
            angle_decision=angle_decision,
            post_brief=post_brief,
            selected_evidence_ids=selected_evidence_ids,
        )

        provider = settings.POSTFLOW_POST_PROVIDER
        model = settings.POSTFLOW_POST_MODEL

        self.stdout.write("=== FINAL POST DEBUG RUN ===")
        self.stdout.write(f"fixture_path: {fixture_path}")
        self.stdout.write(f"provider: {provider}")
        self.stdout.write(f"model: {model}")
        self.stdout.write(f"selected_evidence_ids: {json.dumps(selected_evidence_ids, ensure_ascii=False)}")
        self.stdout.write(f"prompt_path: {PROMPT_PATH}")
        self.stdout.write("")
        self.stdout.write("=== PROMPT / INPUT PREVIEW ===")
        self.stdout.write(_preview_text(execution_prompt))

        if not allow_api:
            self.stdout.write("")
            self.stdout.write("API call skipped. Pass --allow-api to execute the final prompt.")
            return

        _validate_api_guardrails(provider)

        self.stdout.write("")
        self.stdout.write("=== API CALL ===")
        self.stdout.write(f"provider: {provider}")
        self.stdout.write(f"model: {model}")

        response = OpenAIClient(model=model).generate_text(
            prompt=execution_prompt,
            max_output_tokens=max_output_tokens,
            json_mode=True,
        )
        raw_output = response.text.strip()
        parsed_payload: dict[str, Any] | None = None
        final_post_payload_result = "not_run"
        content_package_result = "not_run"
        final_post_text = ""
        diagnostics: FinalPostDiagnostics | None = None

        self.stdout.write("")
        self.stdout.write("=== RAW MODEL OUTPUT ===")
        self.stdout.write(raw_output)

        try:
            parsed_payload = _parse_json_response(raw_output)
            self.stdout.write("")
            self.stdout.write("=== PARSED JSON PAYLOAD ===")
            self.stdout.write(json.dumps(parsed_payload, ensure_ascii=False, indent=2))
        except Exception as exc:  # noqa: BLE001 - debug output should show parse failures.
            self.stdout.write("")
            self.stdout.write("=== PARSED JSON PAYLOAD ===")
            self.stdout.write(f"parse_failed: {exc}")

        if parsed_payload is not None:
            final_post_text = str(parsed_payload.get("post_text") or "")
            final_post_payload_passed, final_post_payload_error = _validate_final_post_payload_dict(parsed_payload)
            final_post_payload_result = _format_validation_result(
                final_post_payload_passed,
                final_post_payload_error,
            )
            content_package_result = _validate_content_package_payload_dict(parsed_payload)
            diagnostics = diagnose_final_post_payload(
                parsed_payload,
                selected_evidence_ids=selected_evidence_ids,
                schema_validation_passed=final_post_payload_passed,
                schema_validation_error=final_post_payload_error,
            )

        self.stdout.write("")
        self.stdout.write("=== VALIDATION ===")
        self.stdout.write(f"FinalPostPayload: {final_post_payload_result}")
        self.stdout.write(f"ContentPackage payload: {content_package_result}")
        self.stdout.write("")
        self.stdout.write("=== FINAL POST TEXT ===")
        self.stdout.write(final_post_text)
        if diagnostics is not None:
            self.stdout.write("")
            _write_diagnostics(self.stdout, diagnostics)

        if save_output:
            output_path = _save_debug_output(
                {
                    "fixture_path": str(fixture_path),
                    "provider": provider,
                    "model": model,
                    "selected_evidence_ids": selected_evidence_ids,
                    "raw_output": raw_output,
                    "parsed_output": parsed_payload,
                    "validation_results": {
                        "final_post_payload": final_post_payload_result,
                        "content_package_payload": content_package_result,
                    },
                    "diagnostics": diagnostics.to_dict() if diagnostics is not None else None,
                    "final_post_text": final_post_text,
                }
            )
            self.stdout.write("")
            self.stdout.write(f"saved_output: {output_path}")


def _load_fixture(fixture_path: Path) -> dict[str, Any]:
    if not fixture_path.exists():
        raise CommandError(f"Fixture does not exist: {fixture_path}")
    with fixture_path.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    if not isinstance(fixture, dict):
        raise CommandError("Fixture must be a JSON object.")
    return fixture


def _load_prompt_contract() -> str:
    if not PROMPT_PATH.exists():
        raise CommandError(f"Prompt contract does not exist: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_post_brief_from_fixture(fixture: dict[str, Any]):
    digest = _FixtureDigest(fixture)
    pipeline_input = build_pipeline_input_from_digest(digest, author_profile={})
    article_evidence_pack = build_article_evidence_pack_from_pipeline_input(pipeline_input)
    contextual_evidence_pack = build_contextual_evidence_pack_from_article_evidence_pack(
        article_evidence_pack
    )
    angle_decision = build_angle_decision_from_contextual_evidence_pack(contextual_evidence_pack)
    post_brief = build_post_brief_from_angle_decision(contextual_evidence_pack, angle_decision)
    return angle_decision, post_brief


class _FixtureDigest:
    def __init__(self, fixture: dict[str, Any]) -> None:
        self.id = fixture["digest"]["id"]
        self.title = fixture["digest"]["title"]
        self.run = SimpleNamespace(topic=SimpleNamespace(name=fixture["topic"]["name"]))
        self._articles = fixture["articles"]

    def get_articles(self) -> list[dict[str, Any]]:
        return list(self._articles)


def _build_execution_prompt(
    prompt_text: str,
    *,
    fixture: dict[str, Any],
    angle_decision,
    post_brief,
    selected_evidence_ids: list[str],
) -> str:
    execution_input = {
        "topic": fixture.get("topic", {}),
        "digest": fixture.get("digest", {}),
        "angle_decision": asdict(angle_decision),
        "post_brief": asdict(post_brief),
        "selected_evidence_ids": selected_evidence_ids,
        "expected_behavior": fixture.get("expected_behavior", {}),
    }
    return (
        f"{prompt_text.strip()}\n\n"
        "## Debug Execution Input\n\n"
        "Use the following JSON input for this debug run:\n\n"
        f"{json.dumps(execution_input, ensure_ascii=False, indent=2)}"
    )


def _validate_api_guardrails(provider: str) -> None:
    if provider != "openai":
        raise CommandError("debug_final_post_from_brief currently supports only POSTFLOW_POST_PROVIDER=openai.")
    api_key = str(settings.OPENAI_API_KEY or "").strip()
    if api_key in PLACEHOLDER_API_KEYS:
        raise CommandError("OPENAI_API_KEY must be configured with a real key before using --allow-api.")


def _parse_json_response(response_text: str) -> dict[str, Any]:
    candidate = _extract_json_candidate(response_text)
    if not candidate:
        raise CommandError("Model output is empty or does not contain a JSON object.")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise CommandError(f"Model output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CommandError("Model output must be a JSON object.")
    return payload


def _extract_json_candidate(response_text: str) -> str:
    normalized = str(response_text or "").strip()
    if not normalized:
        return ""
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    if normalized.startswith("{") and normalized.endswith("}"):
        return normalized
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return normalized[start : end + 1]


def _validate_final_post_payload_dict(payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        final_post_payload = FinalPostPayload(
            post_text=payload.get("post_text", ""),
            hook_variants=payload.get("hook_variants", []),
            cta_variants=payload.get("cta_variants", []),
            hashtags=payload.get("hashtags", []),
            quality_checks=payload.get("quality_checks", {}),
            carousel_outline=payload.get("carousel_outline", []),
        )
        validate_final_post_payload(final_post_payload)
    except Exception as exc:  # noqa: BLE001 - command reports validation failures.
        return False, str(exc)
    return True, ""


def _format_validation_result(validation_passed: bool, validation_error: str) -> str:
    if validation_passed:
        return "passed"
    return f"failed: {validation_error}"


def _validate_content_package_payload_dict(payload: dict[str, Any]) -> str:
    try:
        validate_content_package_payload(payload)
    except Exception as exc:  # noqa: BLE001 - command reports validation failures.
        return f"failed: {exc}"
    return "passed"


def _preview_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated preview: {len(text) - limit} chars omitted]"


def _save_debug_output(payload: dict[str, Any]) -> Path:
    output_dir = Path(settings.BASE_DIR) / "debug_outputs" / "final_post_model_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"final_post_debug_{timestamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _write_diagnostics(stdout, diagnostics: FinalPostDiagnostics) -> None:
    stdout.write("=== DETERMINISTIC DIAGNOSTICS ===")
    stdout.write(f"system_linkedin_ready: {str(diagnostics.system_linkedin_ready).lower()}")
    stdout.write(f"deterministic_checks_passed: {str(diagnostics.deterministic_checks_passed).lower()}")
    stdout.write(f"model_claimed_linkedin_ready: {_format_optional_bool(diagnostics.model_claimed_linkedin_ready)}")
    stdout.write(
        "missing_quality_check_keys: "
        f"{json.dumps(diagnostics.missing_quality_check_keys, ensure_ascii=False)}"
    )
    stdout.write(
        "non_boolean_quality_check_keys: "
        f"{json.dumps(diagnostics.non_boolean_quality_check_keys, ensure_ascii=False)}"
    )
    stdout.write(
        "evidence_id_leaks: "
        f"{json.dumps([leak.to_dict() for leak in diagnostics.evidence_id_leaks], ensure_ascii=False)}"
    )
    stdout.write(
        "scaffold_phrase_leaks: "
        f"{json.dumps([leak.to_dict() for leak in diagnostics.scaffold_phrase_leaks], ensure_ascii=False)}"
    )
    stdout.write(
        "source_summary_phrase_leaks: "
        f"{json.dumps([leak.to_dict() for leak in diagnostics.source_summary_phrase_leaks], ensure_ascii=False)}"
    )
    stdout.write(f"repair_reasons: {json.dumps(diagnostics.repair_reasons, ensure_ascii=False)}")


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "null"
    return str(value).lower()
