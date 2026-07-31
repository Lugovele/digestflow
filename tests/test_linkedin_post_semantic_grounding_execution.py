from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

from django.test import SimpleTestCase

from services.packaging import linkedin_post_semantic_grounding_execution
from services.packaging.linkedin_post_prompt_renderers import SemanticGroundingPromptRender
from services.packaging.linkedin_post_semantic_grounding_execution import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    SemanticGroundingRawResponse,
    build_semantic_grounding_execution_request,
    execute_semantic_grounding_prompt,
)


class LinkedInPostSemanticGroundingExecutionTests(SimpleTestCase):
    def test_request_defaults_to_post_model_role_settings(self) -> None:
        render = _render()
        request = build_semantic_grounding_execution_request(
            render,
            prompt_text="Semantic prompt.",
            provider="openai",
            model="grounding-model",
        )

        self.assertEqual(request.provider, "openai")
        self.assertEqual(request.model, "grounding-model")
        self.assertEqual(request.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertTrue(request.json_mode)

    def test_request_failure_does_not_call_provider(self) -> None:
        client = FakeClient("unused")
        request = build_semantic_grounding_execution_request(
            _render(),
            prompt_text="Semantic prompt.",
            provider="",
            model="grounding-model",
        )

        response = execute_semantic_grounding_prompt(request, client=client)

        self.assertEqual(client.call_count, 0)
        self.assertEqual(response.execution_error, "missing semantic grounding provider")

    def test_fake_provider_response_is_captured_without_json_fallback(self) -> None:
        client = FakeClient('{"pass": true, "claims": [], "failed_claim_ids": []}')
        request = build_semantic_grounding_execution_request(
            _render(),
            prompt_text="Semantic prompt.",
            provider="openai",
            model="grounding-model",
            execution_metadata={"trace": "audit-only"},
        )

        response = execute_semantic_grounding_prompt(request, client=client)

        self.assertIsInstance(response, SemanticGroundingRawResponse)
        self.assertEqual(client.call_count, 1)
        self.assertTrue(client.json_mode)
        self.assertFalse(client.allow_json_mode_fallback)
        self.assertIn("Semantic prompt.", client.prompts[0])
        self.assertNotIn("audit-only", client.prompts[0])
        self.assertEqual(response.raw_text, '{"pass": true, "claims": [], "failed_claim_ids": []}')

    def test_empty_response_is_execution_error(self) -> None:
        client = FakeClient("")
        request = build_semantic_grounding_execution_request(
            _render(),
            prompt_text="Semantic prompt.",
            provider="openai",
            model="grounding-model",
        )

        response = execute_semantic_grounding_prompt(request, client=client)

        self.assertEqual(response.execution_error, "empty provider response")

    def test_execution_module_has_no_runtime_generation_or_repair_imports(self) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_semantic_grounding_execution))
        imported_modules = _imported_modules(tree)
        imported_symbols = _imported_symbols(tree)

        forbidden_modules = {
            "services.packaging.generator",
            "apps.packaging.models",
            "django.db",
            "services.sources",
        }
        forbidden_symbols = {
            "ContentPackage",
            "generate_content_package_for_digest",
            "GeminiClient",
            "RepairAgent",
        }

        self.assertTrue(forbidden_modules.isdisjoint(imported_modules))
        self.assertTrue(forbidden_symbols.isdisjoint(imported_symbols))


class FakeClient:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text
        self.call_count = 0
        self.prompts: list[str] = []

    def generate_text(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
        json_mode: bool,
        allow_json_mode_fallback: bool = True,
    ) -> SimpleNamespace:
        self.call_count += 1
        self.prompts.append(prompt)
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.allow_json_mode_fallback = allow_json_mode_fallback
        return SimpleNamespace(
            text=self.raw_text,
            raw={"id": "resp-1"},
            usage={"total_tokens": 10},
        )


def _render() -> SemanticGroundingPromptRender:
    return SemanticGroundingPromptRender(
        prompt_name="final_post_semantic_grounding_evaluator",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_semantic_grounding_evaluator.txt",
        variables={"candidate_payload_json": "{}"},
        input_text="## CANDIDATE_PAYLOAD_JSON\n{}",
    )


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _imported_symbols(tree: ast.AST) -> set[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            symbols.update(
                alias.asname or alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            symbols.update(alias.asname or alias.name for alias in node.names)
    return symbols
