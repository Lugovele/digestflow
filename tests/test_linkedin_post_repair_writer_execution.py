from __future__ import annotations

import ast
import copy
import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from django.test import override_settings

from services.packaging import linkedin_post_repair_writer_execution
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_prompt_renderers import RepairWriterPromptRender
from services.packaging.linkedin_post_repair_writer_execution import (
    DEFAULT_REPAIR_WRITER_MAX_OUTPUT_TOKENS,
    RepairWriterExecutionRequest,
    RepairWriterRawResponse,
    build_repair_writer_execution_request,
    execute_repair_writer_prompt,
)


class RepairWriterExecutionTests(SimpleTestCase):
    def test_execution_request_construction_uses_postflow_post_settings(self) -> None:
        with override_settings(
            POSTFLOW_POST_PROVIDER="openai",
            POSTFLOW_POST_MODEL="gpt-4.1-2025-04-14",
        ):
            request = build_repair_writer_execution_request(
                _render(),
                prompt_text="Repair writer prompt.",
                execution_metadata={"attempt": 1},
            )

        self.assertIsInstance(request, RepairWriterExecutionRequest)
        self.assertEqual(request.provider, "openai")
        self.assertEqual(request.model, "gpt-4.1-2025-04-14")
        self.assertEqual(request.execution_metadata, {"attempt": 1})

    def test_raw_response_to_dict_preserves_raw_text_and_metadata(self) -> None:
        response = RepairWriterRawResponse(
            raw_text='{"post_text": "Repaired"}',
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_metadata=_prompt_metadata(),
            usage={"total_tokens": 12},
            raw_provider_response={"id": "resp_1"},
            provider_response_metadata={
                "provider": "openai",
                "provider_output_limit_reached": False,
            },
            execution_metadata={"attempt": {"index": 1}},
        )

        serialized = response.to_dict()
        serialized["usage"]["total_tokens"] = 99
        serialized["raw_provider_response"]["id"] = "changed"
        serialized["provider_response_metadata"][
            "provider_output_limit_reached"
        ] = True
        serialized["execution_metadata"]["attempt"]["index"] = 2

        self.assertEqual(serialized["raw_text"], '{"post_text": "Repaired"}')
        self.assertEqual(response.usage, {"total_tokens": 12})
        self.assertEqual(response.raw_provider_response, {"id": "resp_1"})
        self.assertEqual(
            response.provider_response_metadata,
            {"provider": "openai", "provider_output_limit_reached": False},
        )
        self.assertEqual(response.execution_metadata, {"attempt": {"index": 1}})

    @patch("services.packaging.linkedin_post_repair_writer_execution.build_ai_client")
    def test_successful_execution_calls_provider_once_and_returns_raw_response(
        self,
        mock_build_ai_client,
    ) -> None:
        request = _request()
        provider_response = SimpleNamespace(
            text='{"post_text": "Repaired post"}',
            raw={"id": "resp_123"},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            provider_response_metadata={
                "provider": "openai",
                "model": "gpt-4.1-2025-04-14",
                "provider_finish_reason": "stop",
                "provider_max_output_tokens": 1800,
                "provider_reported_output_tokens": 5,
                "provider_output_limit_reached": False,
                "raw_provider_secret": "do-not-serialize",
            },
        )
        mock_build_ai_client.return_value.generate_text.return_value = provider_response

        raw_response = execute_repair_writer_prompt(request)

        mock_build_ai_client.assert_called_once_with(
            provider="openai",
            model="gpt-4.1-2025-04-14",
        )
        mock_build_ai_client.return_value.generate_text.assert_called_once_with(
            prompt=f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}",
            max_output_tokens=request.max_output_tokens,
            json_mode=False,
        )
        self.assertEqual(raw_response.raw_text, provider_response.text)
        self.assertEqual(raw_response.provider, "openai")
        self.assertEqual(raw_response.model, "gpt-4.1-2025-04-14")
        self.assertIsNone(raw_response.execution_error)
        self.assertEqual(
            raw_response.provider_response_metadata,
            {
                "provider": "openai",
                "model": "gpt-4.1-2025-04-14",
                "provider_finish_reason": "stop",
                "provider_max_output_tokens": 1800,
                "provider_reported_output_tokens": 5,
                "provider_output_limit_reached": False,
            },
        )

    @patch("services.packaging.linkedin_post_repair_writer_execution.build_ai_client")
    def test_execution_does_not_mutate_render_request_or_metadata(
        self,
        mock_build_ai_client,
    ) -> None:
        render = _render()
        request = _request(render=render, execution_metadata={"attempt": {"index": 1}})
        render_before = copy.deepcopy(render)
        request_before = copy.deepcopy(request)
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="Raw output",
            raw={},
            usage={},
        )

        execute_repair_writer_prompt(request)

        self.assertEqual(render, render_before)
        self.assertEqual(request, request_before)

    @patch("services.packaging.linkedin_post_repair_writer_execution.build_ai_client")
    def test_empty_response_returns_execution_error_without_parsing(
        self,
        mock_build_ai_client,
    ) -> None:
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=" ",
            raw={"id": "empty"},
            usage={"total_tokens": 3},
            provider_response_metadata={
                "provider": "openai",
                "provider_output_limit_reached": True,
            },
        )

        raw_response = execute_repair_writer_prompt(_request())

        self.assertEqual(raw_response.execution_error, "empty provider response")
        self.assertEqual(raw_response.raw_provider_response, {"id": "empty"})
        self.assertEqual(
            raw_response.provider_response_metadata,
            {"provider": "openai", "provider_output_limit_reached": True},
        )

    @patch("services.packaging.linkedin_post_repair_writer_execution.build_ai_client")
    def test_provider_failure_returns_sanitized_execution_error(
        self,
        mock_build_ai_client,
    ) -> None:
        mock_build_ai_client.return_value.generate_text.side_effect = RuntimeError(
            "secret provider details"
        )

        raw_response = execute_repair_writer_prompt(_request())

        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertNotIn("secret provider details", json.dumps(raw_response.to_dict()))

    @patch("services.packaging.linkedin_post_repair_writer_execution.build_ai_client")
    def test_provider_value_error_returns_sanitized_execution_error(
        self,
        mock_build_ai_client,
    ) -> None:
        mock_build_ai_client.return_value.generate_text.side_effect = ValueError(
            "secret provider details"
        )

        raw_response = execute_repair_writer_prompt(_request())

        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertNotIn("secret provider details", json.dumps(raw_response.to_dict()))

    @patch("services.packaging.linkedin_post_repair_writer_execution.build_ai_client")
    def test_request_errors_do_not_call_provider(self, mock_build_ai_client) -> None:
        invalid_requests = (
            (_request(provider=""), "missing repair writer provider"),
            (_request(model=""), "missing repair writer model"),
            (
                _request(max_output_tokens=0),
                "invalid repair writer max_output_tokens: must be a positive integer",
            ),
            (_request(prompt_text=" "), "missing repair writer prompt text"),
            (
                _request(render=_render(input_text=" ")),
                "missing repair writer rendered input text",
            ),
        )

        for request, expected_error in invalid_requests:
            with self.subTest(expected_error=expected_error):
                raw_response = execute_repair_writer_prompt(request)

                self.assertEqual(raw_response.execution_error, expected_error)

        mock_build_ai_client.assert_not_called()


    @patch("services.packaging.linkedin_post_repair_writer_execution.build_ai_client")
    def test_gemini_repair_writer_config_is_allowed_and_calls_provider(
        self,
        mock_build_ai_client,
    ) -> None:
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text='{"post_text": "Repaired by Gemini."}',
            raw={"id": "gemini_repair"},
            usage={"total_tokens": 19},
            provider_response_metadata={
                "provider": "gemini",
                "model": "gemini-3.6-flash",
                "provider_finish_reason": "stop",
                "provider_stop_reason": None,
                "provider_max_output_tokens": 1800,
                "provider_reported_output_tokens": 19,
                "provider_output_limit_reached": False,
            },
        )
        raw_response = execute_repair_writer_prompt(
            _request(provider="gemini", model="gemini-3.6-flash")
        )

        mock_build_ai_client.assert_called_once_with(
            provider="gemini",
            model="gemini-3.6-flash",
        )
        self.assertEqual(raw_response.raw_text, '{"post_text": "Repaired by Gemini."}')
        self.assertIsNone(raw_response.execution_error)
        self.assertEqual(
            raw_response.provider_response_metadata["provider_output_limit_reached"],
            False,
        )

    @patch("services.packaging.linkedin_post_repair_writer_execution.build_ai_client")
    def test_anthropic_repair_writer_config_is_allowed_and_calls_provider(
        self,
        mock_build_ai_client,
    ) -> None:
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text='{"post_text": "Repaired by Claude."}',
            raw={"id": "anthropic_repair"},
            usage={"total_tokens": 21},
            provider_response_metadata={
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "provider_stop_reason": "end_turn",
                "provider_finish_reason": None,
                "provider_max_output_tokens": 1800,
                "provider_reported_output_tokens": 21,
                "provider_output_limit_reached": False,
            },
        )
        request = _request(provider="anthropic", model="claude-sonnet-5")

        raw_response = execute_repair_writer_prompt(request)

        mock_build_ai_client.assert_called_once_with(
            provider="anthropic",
            model="claude-sonnet-5",
        )
        mock_build_ai_client.return_value.generate_text.assert_called_once_with(
            prompt=f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}",
            max_output_tokens=request.max_output_tokens,
            json_mode=False,
        )
        self.assertEqual(raw_response.raw_text, '{"post_text": "Repaired by Claude."}')
        self.assertIsNone(raw_response.execution_error)

    def test_response_to_dict_is_json_serializable(self) -> None:
        response = RepairWriterRawResponse(
            raw_text='{"post_text": "Repaired"}',
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_metadata=_prompt_metadata(),
            usage={"total_tokens": 10},
            raw_provider_response={"id": "resp"},
            execution_metadata={"attempt": 1},
        )

        serialized = json.dumps(response.to_dict(), sort_keys=True)

        self.assertIn("gpt-4.1-2025-04-14", serialized)

    def test_execution_module_has_no_parser_gate_decision_or_runtime_dependencies(
        self,
    ) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_repair_writer_execution))
        imported_modules = _imported_modules(tree)
        imported_symbols = _imported_symbols(tree)

        forbidden_modules = {
            "apps.packaging.models",
            "django.db",
            "services.packaging.generator",
            "services.packaging.linkedin_post_flow_handoffs",
            "services.packaging.linkedin_post_pipeline",
        }
        forbidden_symbols = {
            "CandidateWriterOutput",
            "ContentPackage",
            "FinalPostDecisionController",
            "FinalPostPayload",
            "QualityEvaluator",
            "TargetedRepairPlan",
            "generate_content_package_for_digest",
            "run_final_post_deterministic_gate",
            "validate_final_post_payload",
        }

        self.assertTrue(forbidden_modules.isdisjoint(imported_modules))
        self.assertTrue(forbidden_symbols.isdisjoint(imported_symbols))


def _request(
    *,
    render: RepairWriterPromptRender | None = None,
    prompt_text: str = "Repair writer prompt.",
    provider: str = "openai",
    model: str = "gpt-4.1-2025-04-14",
    max_output_tokens: object = DEFAULT_REPAIR_WRITER_MAX_OUTPUT_TOKENS,
    execution_metadata: dict | None = None,
) -> RepairWriterExecutionRequest:
    return RepairWriterExecutionRequest(
        rendered_prompt_input=render or _render(),
        prompt_text=prompt_text,
        provider=provider,
        model=model,
        max_output_tokens=max_output_tokens,
        execution_metadata=execution_metadata,
    )


def _render(*, input_text: str = "## REPAIR_INSTRUCTION_JSON\n{}") -> RepairWriterPromptRender:
    return RepairWriterPromptRender(
        prompt_name="final_post_repair_writer",
        prompt_version="1.0",
        prompt_path=None,
        variables={
            "repair_instruction_json": "{}",
        },
        input_text=input_text,
    )


def _prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_name="final_post_repair_writer",
        prompt_version="1.0",
        prompt_path=None,
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
