from __future__ import annotations

import ast
import copy
import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from django.test import override_settings

from apps.ai.client import (
    AI_THINKING_MODE_DISABLED,
    AI_THINKING_MODE_PROVIDER_DEFAULT,
)
from services.packaging import linkedin_post_candidate_writer_execution
from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterExecutionRequest,
    CandidateWriterRawResponse,
    EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT,
    build_candidate_writer_execution_request,
    execute_candidate_writer_prompt,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_prompt_renderers import (
    CandidateWriterPromptRender,
)


class CandidateWriterExecutionTests(SimpleTestCase):
    def test_execution_request_construction_uses_candidate_writer_role_settings(self) -> None:
        with override_settings(
            POSTFLOW_CANDIDATE_WRITER_PROVIDER="anthropic",
            POSTFLOW_CANDIDATE_WRITER_MODEL="claude-sonnet-5",
        ):
            request = build_candidate_writer_execution_request(
                _render(),
                prompt_text="Candidate writer prompt.",
                execution_metadata={"attempt": 1},
            )

        self.assertIsInstance(request, CandidateWriterExecutionRequest)
        self.assertEqual(request.provider, "anthropic")
        self.assertEqual(request.model, "claude-sonnet-5")
        self.assertEqual(request.thinking_mode, AI_THINKING_MODE_DISABLED)
        self.assertEqual(request.execution_metadata, {"attempt": 1})

    def test_execution_request_to_dict_defensively_copies_metadata(self) -> None:
        request = _request(execution_metadata={"attempt": {"index": 1}})

        serialized = request.to_dict()
        serialized["execution_metadata"]["attempt"]["index"] = 2

        self.assertEqual(request.execution_metadata, {"attempt": {"index": 1}})
        self.assertEqual(serialized["thinking_mode"], AI_THINKING_MODE_PROVIDER_DEFAULT)

    def test_raw_response_to_dict_preserves_raw_text_and_metadata(self) -> None:
        response = CandidateWriterRawResponse(
            raw_text='{"post_text": "Draft"}',
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_metadata=_prompt_metadata(),
            usage={"total_tokens": 12},
            raw_provider_response={"id": "resp_1"},
            provider_response_metadata={"stop_reason": "end_turn"},
            empty_text_classification=EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT,
            execution_metadata={"attempt": {"index": 1}},
        )

        serialized = response.to_dict()
        serialized["usage"]["total_tokens"] = 99
        serialized["raw_provider_response"]["id"] = "changed"
        serialized["provider_response_metadata"]["stop_reason"] = "changed"
        serialized["execution_metadata"]["attempt"]["index"] = 2

        self.assertEqual(serialized["raw_text"], '{"post_text": "Draft"}')
        self.assertEqual(
            serialized["empty_text_classification"],
            EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT,
        )
        self.assertEqual(response.usage, {"total_tokens": 12})
        self.assertEqual(response.raw_provider_response, {"id": "resp_1"})
        self.assertEqual(response.provider_response_metadata, {"stop_reason": "end_turn"})
        self.assertEqual(response.execution_metadata, {"attempt": {"index": 1}})

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_successful_execution_calls_provider_once_and_returns_raw_response(
        self,
        mock_build_ai_client,
    ) -> None:
        request = _request()
        provider_response = SimpleNamespace(
            text='{"post_text": "Candidate post"}',
            raw={"id": "resp_123", "status": "completed"},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_build_ai_client.return_value.generate_text.return_value = provider_response

        raw_response = execute_candidate_writer_prompt(request)

        mock_build_ai_client.assert_called_once_with(
            provider="openai",
            model="gpt-4.1-2025-04-14",
        )
        mock_build_ai_client.return_value.generate_text.assert_called_once_with(
            prompt=f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}",
            max_output_tokens=request.max_output_tokens,
            json_mode=False,
            thinking_mode=AI_THINKING_MODE_PROVIDER_DEFAULT,
        )
        self.assertEqual(raw_response.raw_text, provider_response.text)
        self.assertEqual(raw_response.provider, "openai")
        self.assertEqual(raw_response.model, "gpt-4.1-2025-04-14")
        self.assertEqual(raw_response.usage, provider_response.usage)
        self.assertEqual(raw_response.raw_provider_response, provider_response.raw)
        self.assertIsNone(raw_response.execution_error)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_execution_forwards_prompt_text_and_input_text_unchanged(
        self,
        mock_build_ai_client,
    ) -> None:
        unicode_word = (
            "\u0447\u0435\u043b\u043e\u0432\u0435\u0447"
            "\u043d\u043e\u0441\u0442\u044c"
        )
        render = _render(
            input_text=f"## POST_BRIEF_JSON\nUnicode: {unicode_word}\n```json\n{{}}\n```"
        )
        prompt_text = "Resolved candidate prompt text.\nDo not change this."
        request = _request(render=render, prompt_text=prompt_text)
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="Raw output",
            raw={},
            usage={},
        )

        execute_candidate_writer_prompt(request)

        called_prompt = mock_build_ai_client.return_value.generate_text.call_args.kwargs[
            "prompt"
        ]
        self.assertEqual(called_prompt, f"{prompt_text}\n\n{render.input_text}")

    @override_settings(OPENAI_API_KEY="test-key", OPENAI_TIMEOUT_SECONDS=30)
    @patch("apps.ai.client.OpenAI")
    def test_provider_failure_is_not_retried_by_json_mode_fallback(
        self,
        mock_openai,
    ) -> None:
        request = _request()
        create_calls = []

        def create_response(**kwargs):
            create_calls.append(kwargs)
            if len(create_calls) == 1:
                raise RuntimeError("json mode unavailable")
            return SimpleNamespace(
                output_text="fallback success",
                model_dump=lambda: {"id": "retried"},
                usage=SimpleNamespace(
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                ),
            )

        mock_openai.return_value.responses.create.side_effect = create_response

        raw_response = execute_candidate_writer_prompt(request)

        self.assertEqual(len(create_calls), 1)
        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertNotIn("text", create_calls[0])

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
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

        execute_candidate_writer_prompt(request)

        self.assertEqual(render, render_before)
        self.assertEqual(request, request_before)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_empty_response_returns_execution_error_without_parsing(
        self,
        mock_build_ai_client,
    ) -> None:
        empty_values = (None, "", "   ", "\r\n")

        for empty_value in empty_values:
            with self.subTest(raw_text=empty_value):
                mock_build_ai_client.return_value.generate_text.return_value = (
                    SimpleNamespace(
                        text=empty_value,
                        raw={"id": "empty"},
                        usage={"total_tokens": 3},
                    )
                )

                raw_response = execute_candidate_writer_prompt(_request())

                self.assertEqual(raw_response.raw_text, str(empty_value or ""))
                self.assertEqual(
                    raw_response.execution_error,
                    "empty provider response",
                )
                self.assertEqual(raw_response.raw_provider_response, {"id": "empty"})

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_empty_ai_response_maps_to_empty_provider_response_with_safe_metadata(
        self,
        mock_build_ai_client,
    ) -> None:
        request = build_candidate_writer_execution_request(
            _render(),
            prompt_text="Candidate writer prompt.",
            provider="anthropic",
            model="claude-sonnet-5",
        )
        provider_metadata = {
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "stop_reason": "max_tokens",
            "content_block_types": ["thinking"],
            "input_tokens": 11,
            "output_tokens": 13,
            "thinking_tokens": 13,
        }
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="",
            raw={},
            usage={"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24},
            provider_response_metadata=provider_metadata,
        )

        raw_response = execute_candidate_writer_prompt(request)

        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "empty provider response")
        self.assertEqual(raw_response.raw_provider_response, {})
        self.assertEqual(raw_response.provider_response_metadata, provider_metadata)
        self.assertEqual(
            raw_response.empty_text_classification,
            EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT,
        )
        self.assertNotEqual(raw_response.execution_error, "provider invocation failed")
        serialized = json.dumps(raw_response.to_dict(), sort_keys=True)
        self.assertNotIn("secret provider thinking text", serialized)
        mock_build_ai_client.return_value.generate_text.assert_called_once()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_empty_response_classification_requires_supported_metadata(
        self,
        mock_build_ai_client,
    ) -> None:
        cases = (
            (
                "end_turn",
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "stop_reason": "end_turn",
                    "content_block_types": ["thinking"],
                    "output_tokens": 4000,
                    "thinking_tokens": 4000,
                },
            ),
            (
                "no_thinking_block",
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "stop_reason": "max_tokens",
                    "content_block_types": [],
                    "output_tokens": 4000,
                    "thinking_tokens": 4000,
                },
            ),
            (
                "text_block_present",
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "stop_reason": "max_tokens",
                    "content_block_types": ["thinking", "text"],
                    "output_tokens": 4000,
                    "thinking_tokens": 4000,
                },
            ),
            (
                "malformed_metadata",
                {
                    "provider": "anthropic",
                    "stop_reason": "max_tokens",
                    "content_block_types": {"type": "thinking"},
                    "output_tokens": "4000",
                    "thinking_tokens": 4000,
                },
            ),
            (
                "thinking_tokens_do_not_match_output_tokens",
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "stop_reason": "max_tokens",
                    "content_block_types": ["thinking"],
                    "output_tokens": 4000,
                    "thinking_tokens": 3999,
                },
            ),
        )

        for case_name, metadata in cases:
            with self.subTest(case_name=case_name):
                mock_build_ai_client.reset_mock()
                mock_build_ai_client.return_value.generate_text.return_value = (
                    SimpleNamespace(
                        text="",
                        raw={},
                        usage={"total_tokens": 3},
                        provider_response_metadata=metadata,
                    )
                )

                raw_response = execute_candidate_writer_prompt(
                    _request(provider="anthropic", model="claude-sonnet-5")
                )

                self.assertEqual(raw_response.execution_error, "empty provider response")
                self.assertIsNone(raw_response.empty_text_classification)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_provider_failure_returns_sanitized_execution_error(
        self,
        mock_build_ai_client,
    ) -> None:
        request = _request()
        request_before = copy.deepcopy(request)
        mock_build_ai_client.return_value.generate_text.side_effect = RuntimeError(
            "secret provider details"
        )

        raw_response = execute_candidate_writer_prompt(request)

        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertNotIn("secret provider details", raw_response.to_dict().values())
        self.assertEqual(request, request_before)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_provider_value_error_returns_sanitized_execution_error(
        self,
        mock_build_ai_client,
    ) -> None:
        mock_build_ai_client.return_value.generate_text.side_effect = ValueError(
            "secret provider details"
        )

        raw_response = execute_candidate_writer_prompt(_request())

        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertNotIn("secret provider details", json.dumps(raw_response.to_dict()))

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_unsupported_provider_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(
            _request(provider="unknown", model="gpt-4.1-2025-04-14")
        )

        self.assertEqual(
            raw_response.execution_error,
            "unsupported PostFlow final post role/provider/model: role=candidate_writer provider=unknown model=gpt-4.1-2025-04-14",
        )
        mock_build_ai_client.assert_not_called()


    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_gemini_candidate_writer_delegates_to_generic_client_once(
        self,
        mock_build_ai_client,
    ) -> None:
        request = _request(provider="gemini", model="gemini-3.6-flash")
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text='{"post_text": "Gemini candidate"}',
            raw={"id": "gemini-response"},
            usage={"total_tokens": 11},
        )

        raw_response = execute_candidate_writer_prompt(request)

        mock_build_ai_client.assert_called_once_with(
            provider="gemini",
            model="gemini-3.6-flash",
        )
        mock_build_ai_client.return_value.generate_text.assert_called_once_with(
            prompt=f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}",
            max_output_tokens=request.max_output_tokens,
            json_mode=False,
            thinking_mode=AI_THINKING_MODE_PROVIDER_DEFAULT,
        )
        self.assertEqual(raw_response.provider, "gemini")
        self.assertEqual(raw_response.model, "gemini-3.6-flash")
        self.assertIsNone(raw_response.execution_error)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_anthropic_candidate_writer_delegates_to_generic_client_once(
        self,
        mock_build_ai_client,
    ) -> None:
        request = build_candidate_writer_execution_request(
            _render(),
            prompt_text="Candidate writer prompt.",
            provider="anthropic",
            model="claude-sonnet-5",
        )
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text='{"post_text": "Claude candidate"}',
            raw={"id": "claude-response"},
            usage={"total_tokens": 11},
        )

        raw_response = execute_candidate_writer_prompt(request)

        mock_build_ai_client.assert_called_once_with(
            provider="anthropic",
            model="claude-sonnet-5",
        )
        mock_build_ai_client.return_value.generate_text.assert_called_once_with(
            prompt=f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}",
            max_output_tokens=request.max_output_tokens,
            json_mode=False,
            thinking_mode=AI_THINKING_MODE_DISABLED,
        )
        self.assertEqual(raw_response.provider, "anthropic")
        self.assertEqual(raw_response.model, "claude-sonnet-5")
        self.assertIsNone(raw_response.execution_error)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_anthropic_candidate_writer_resolves_disabled_thinking_mode(
        self,
        mock_build_ai_client,
    ) -> None:
        request = build_candidate_writer_execution_request(
            _render(),
            prompt_text="Candidate writer prompt.",
            provider="anthropic",
            model="claude-sonnet-5",
            max_output_tokens=4000,
        )
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text='{"post_text": "Claude candidate"}',
            raw={"id": "claude-response"},
            usage={"total_tokens": 11},
        )

        execute_candidate_writer_prompt(request)

        self.assertEqual(request.thinking_mode, AI_THINKING_MODE_DISABLED)
        mock_build_ai_client.return_value.generate_text.assert_called_once_with(
            prompt=f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}",
            max_output_tokens=4000,
            json_mode=False,
            thinking_mode=AI_THINKING_MODE_DISABLED,
        )

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_non_anthropic_candidate_writer_rejects_non_default_thinking_mode(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(
            _request(provider="openai", thinking_mode=AI_THINKING_MODE_DISABLED)
        )

        self.assertEqual(
            raw_response.execution_error,
            "unsupported candidate writer thinking_mode for provider openai: disabled",
        )
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_missing_provider_returns_distinct_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(
            _request(provider="", model="gpt-4.1-2025-04-14")
        )

        self.assertEqual(
            raw_response.execution_error,
            "missing candidate writer provider",
        )
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_invalid_max_output_tokens_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        invalid_values = (True, False, "1200", None, 0, -1)

        for invalid_value in invalid_values:
            with self.subTest(max_output_tokens=invalid_value):
                raw_response = execute_candidate_writer_prompt(
                    _request(max_output_tokens=invalid_value)
                )

                self.assertEqual(
                    raw_response.execution_error,
                    "invalid candidate writer max_output_tokens: "
                    "must be a positive integer",
                )
                self.assertEqual(raw_response.raw_text, "")

        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_missing_prompt_content_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(_request(prompt_text=" "))

        self.assertEqual(
            raw_response.execution_error,
            "missing candidate writer prompt text",
        )
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_blank_rendered_input_text_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(
            _request(render=_render(input_text=" \n\t "))
        )

        self.assertEqual(
            raw_response.execution_error,
            "missing candidate writer rendered input text",
        )
        self.assertEqual(raw_response.raw_text, "")
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_missing_model_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(_request(model=""))

        self.assertEqual(raw_response.execution_error, "missing candidate writer model")
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_raw_markdown_fenced_json_is_preserved_unparsed(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_text = '```json\n{"post_text": "Candidate"}\n```'
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=raw_text,
            raw={},
            usage={},
        )

        raw_response = execute_candidate_writer_prompt(_request())

        self.assertEqual(raw_response.raw_text, raw_text)
        self.assertNotIn("payload", raw_response.to_dict())

    @patch("services.packaging.linkedin_post_candidate_writer_execution.build_ai_client")
    def test_unicode_raw_response_is_preserved(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_text = (
            '{"post_text": "'
            "\u0447\u0435\u043b\u043e\u0432\u0435\u0447\u043d\u043e\u0441\u0442\u044c "
            "\u0442\u0435\u043a\u0441\u0442\u0430"
            '"}'
        )
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=raw_text,
            raw={},
            usage={},
        )

        raw_response = execute_candidate_writer_prompt(_request())

        self.assertEqual(raw_response.raw_text, raw_text)

    def test_raw_response_to_dict_omits_unavailable_optional_metadata(self) -> None:
        response = CandidateWriterRawResponse(
            raw_text="Raw",
            provider="openai",
            model="gpt-4.1-2025-04-14",
        )

        serialized = response.to_dict()

        self.assertEqual(set(serialized), {"raw_text", "provider", "model"})

    def test_response_to_dict_is_json_serializable(self) -> None:
        response = CandidateWriterRawResponse(
            raw_text='{"post_text": "Candidate"}',
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_metadata=_prompt_metadata(),
            usage={"total_tokens": 10},
            raw_provider_response={"id": "resp"},
            execution_metadata={"attempt": 1},
        )

        serialized = json.dumps(response.to_dict(), sort_keys=True)

        self.assertIn("gpt-4.1-2025-04-14", serialized)

    def test_execution_module_has_no_parser_gate_decision_repair_or_runtime_dependencies(
        self,
    ) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_candidate_writer_execution))
        imported_modules = _imported_modules(tree)
        imported_symbols = _imported_symbols(tree)

        forbidden_modules = {
            "apps.packaging.models",
            "django.db",
            "services.packaging.generator",
            "services.packaging.linkedin_post_flow_handoffs",
            "services.packaging.linkedin_post_pipeline",
            "services.packaging.linkedin_post_quality_evaluator",
        }
        forbidden_symbols = {
            "CandidateWriterOutput",
            "ContentPackage",
            "FinalPostDecisionController",
            "FinalPostPayload",
            "QualityEvaluator",
            "RepairAgent",
            "TargetedRepairPlan",
            "generate_content_package_for_digest",
            "run_final_post_deterministic_gate",
            "validate_final_post_payload",
        }

        self.assertTrue(forbidden_modules.isdisjoint(imported_modules))
        self.assertTrue(forbidden_symbols.isdisjoint(imported_symbols))


def _request(
    *,
    render: CandidateWriterPromptRender | None = None,
    prompt_text: str = "Candidate writer prompt.",
    provider: str = "openai",
    model: str = "gpt-4.1-2025-04-14",
    max_output_tokens: object = 1200,
    thinking_mode: str = AI_THINKING_MODE_PROVIDER_DEFAULT,
    execution_metadata: dict | None = None,
) -> CandidateWriterExecutionRequest:
    return CandidateWriterExecutionRequest(
        rendered_prompt_input=render or _render(),
        prompt_text=prompt_text,
        provider=provider,
        model=model,
        max_output_tokens=max_output_tokens,
        thinking_mode=thinking_mode,
        execution_metadata=execution_metadata,
    )


def _render(
    *,
    input_text: str = "## CANDIDATE_WRITER_INPUT_JSON\n{}",
) -> CandidateWriterPromptRender:
    return CandidateWriterPromptRender(
        prompt_name="final_post_candidate_from_brief",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_from_brief.txt",
        variables={
            "post_brief_json": "{}",
            "angle_decision_json": "{}",
            "selected_evidence_json": "[]",
            "candidate_writer_input_json": "{}",
        },
        input_text=input_text,
    )


def _prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_name="final_post_candidate_from_brief",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_from_brief.txt",
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
