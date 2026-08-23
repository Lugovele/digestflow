from __future__ import annotations

import copy
import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from django.test import SimpleTestCase
from django.test import override_settings

from services.packaging import linkedin_post_quality_evaluator_execution
from services.packaging.linkedin_post_quality_evaluator_execution import (
    DEFAULT_JSON_MODE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MIN_MAX_OUTPUT_TOKENS,
    QualityEvaluatorExecutionRequest,
    QualityEvaluatorRawResponse,
    build_quality_evaluator_execution_request,
    execute_quality_evaluator_prompt,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_prompt_renderers import (
    QualityEvaluatorPromptRender,
)
from services.packaging.linkedin_post_provider_diagnostics import (
    PROVIDER_ERROR_AUTHENTICATION,
)


class QualityEvaluatorExecutionTests(SimpleTestCase):
    def test_execution_request_construction_uses_quality_evaluator_role_settings(self) -> None:
        with override_settings(
            POSTFLOW_QUALITY_EVALUATOR_PROVIDER="openai",
            POSTFLOW_QUALITY_EVALUATOR_MODEL="gpt-4.1-2025-04-14",
        ):
            request = build_quality_evaluator_execution_request(
                _render(),
                prompt_text="Prompt contract.",
                execution_metadata={"attempt": 1},
            )

        self.assertIsInstance(request, QualityEvaluatorExecutionRequest)
        self.assertEqual(request.provider, "openai")
        self.assertEqual(request.model, "gpt-4.1-2025-04-14")
        self.assertEqual(request.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertIs(request.json_mode, DEFAULT_JSON_MODE)
        self.assertEqual(request.execution_metadata, {"attempt": 1})

    def test_execution_request_to_dict_defensively_copies_metadata(self) -> None:
        request = _request(execution_metadata={"attempt": {"index": 1}})

        serialized = request.to_dict()
        serialized["execution_metadata"]["attempt"]["index"] = 2

        self.assertEqual(request.execution_metadata, {"attempt": {"index": 1}})

    def test_raw_response_to_dict_preserves_raw_text_and_metadata(self) -> None:
        response = QualityEvaluatorRawResponse(
            raw_text='{"pass": true}',
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_metadata=_prompt_metadata(),
            usage={"total_tokens": 12},
            raw_provider_response={"id": "resp_1"},
            provider_response_metadata={"provider_finish_reason": "completed"},
            execution_diagnostics={"provider_error_category": "AUTHENTICATION"},
        )

        serialized = response.to_dict()
        serialized["usage"]["total_tokens"] = 99
        serialized["raw_provider_response"]["id"] = "changed"
        serialized["provider_response_metadata"]["provider_finish_reason"] = "changed"
        serialized["execution_diagnostics"]["provider_error_category"] = "OTHER"

        self.assertEqual(serialized["raw_text"], '{"pass": true}')
        self.assertEqual(response.usage, {"total_tokens": 12})
        self.assertEqual(response.raw_provider_response, {"id": "resp_1"})
        self.assertEqual(
            response.provider_response_metadata,
            {"provider_finish_reason": "completed"},
        )
        self.assertEqual(
            response.execution_diagnostics,
            {"provider_error_category": "AUTHENTICATION"},
        )

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_successful_execution_calls_provider_once_and_returns_raw_response(
        self,
        mock_build_ai_client,
    ) -> None:
        request = _request()
        provider_response = SimpleNamespace(
            text='{"scores": {"hook": 4}}',
            raw={"id": "resp_123", "status": "completed"},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            provider_response_metadata={
                "provider": "openai",
                "provider_finish_reason": "completed",
                "provider_max_output_tokens": 2400,
            },
        )
        mock_build_ai_client.return_value.generate_text.return_value = provider_response

        raw_response = execute_quality_evaluator_prompt(request)

        mock_build_ai_client.assert_called_once_with(
            provider="openai",
            model="gpt-4.1-2025-04-14",
        )
        mock_build_ai_client.return_value.generate_text.assert_called_once_with(
            prompt=f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}",
            max_output_tokens=request.max_output_tokens,
            json_mode=True,
            allow_json_mode_fallback=False,
        )
        self.assertEqual(raw_response.raw_text, provider_response.text)
        self.assertEqual(raw_response.provider, "openai")
        self.assertEqual(raw_response.model, "gpt-4.1-2025-04-14")
        self.assertEqual(raw_response.usage, provider_response.usage)
        self.assertEqual(raw_response.raw_provider_response, provider_response.raw)
        self.assertEqual(
            raw_response.provider_response_metadata,
            provider_response.provider_response_metadata,
        )
        self.assertIsNone(raw_response.execution_error)

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_execution_uses_injected_client_without_constructing_openai_client(
        self,
        mock_build_ai_client,
    ) -> None:
        request = _request()
        fake_client = RecordingQualityEvaluatorClient(
            SimpleNamespace(
                text='{"pass": true}',
                raw={"id": "fake-evaluator"},
                usage={"total_tokens": 7},
            )
        )

        raw_response = execute_quality_evaluator_prompt(request, client=fake_client)

        mock_build_ai_client.assert_not_called()
        self.assertEqual(fake_client.call_count, 1)
        self.assertEqual(fake_client.kwargs["max_output_tokens"], DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertIs(fake_client.kwargs["json_mode"], True)
        self.assertIs(fake_client.kwargs["allow_json_mode_fallback"], False)
        self.assertEqual(raw_response.raw_text, '{"pass": true}')
        self.assertEqual(raw_response.raw_provider_response, {"id": "fake-evaluator"})

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_execution_forwards_prompt_text_and_input_text_unchanged(
        self,
        mock_build_ai_client,
    ) -> None:
        unicode_word = (
            "\u0447\u0435\u043b\u043e\u0432\u0435\u0447"
            "\u043d\u043e\u0441\u0442\u044c"
        )
        render = _render(
            input_text=f"## INPUT\nUnicode: {unicode_word}\n```json\n{{}}\n```"
        )
        prompt_text = "Resolved prompt text.\nDo not change this."
        request = _request(render=render, prompt_text=prompt_text)
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="Raw output",
            raw={},
            usage={},
        )

        execute_quality_evaluator_prompt(request)

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
        mock_openai.return_value.responses.create.side_effect = RuntimeError(
            "json mode unavailable"
        )

        raw_response = execute_quality_evaluator_prompt(request)

        mock_openai.return_value.responses.create.assert_called_once()
        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")

    @override_settings(OPENAI_API_KEY="test-key", OPENAI_TIMEOUT_SECONDS=30)
    @patch("apps.ai.client.OpenAI")
    def test_execution_boundary_invokes_client_once_with_json_mode(
        self,
        mock_openai,
    ) -> None:
        request = _request()
        mock_openai.return_value.responses.create.return_value = SimpleNamespace(
            output_text='{"pass": true}',
            model_dump=lambda: {"id": "json-mode"},
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
            ),
        )

        raw_response = execute_quality_evaluator_prompt(request)

        mock_openai.return_value.responses.create.assert_called_once()
        self.assertEqual(
            mock_openai.return_value.responses.create.call_args.kwargs["text"],
            {"format": {"type": "json_object"}},
        )
        self.assertEqual(raw_response.raw_text, '{"pass": true}')
        self.assertIsNone(raw_response.execution_error)

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_execution_does_not_mutate_render_or_request_metadata(
        self,
        mock_build_ai_client,
    ) -> None:
        render = _render()
        request = _request(render=render, execution_metadata={"attempt": {"index": 1}})
        render_before = copy.deepcopy(render)
        metadata_before = copy.deepcopy(request.execution_metadata)
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="Raw output",
            raw={},
            usage={},
        )

        execute_quality_evaluator_prompt(request)

        self.assertEqual(render, render_before)
        self.assertEqual(request.execution_metadata, metadata_before)

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
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
                        provider_response_metadata={
                            "provider_finish_reason": "completed",
                        },
                    )
                )

                raw_response = execute_quality_evaluator_prompt(_request())

                self.assertEqual(raw_response.raw_text, str(empty_value or ""))
                self.assertEqual(
                    raw_response.execution_error,
                    "empty provider response",
                )
                self.assertEqual(raw_response.raw_provider_response, {"id": "empty"})
                self.assertEqual(
                    raw_response.provider_response_metadata,
                    {"provider_finish_reason": "completed"},
                )

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_provider_failure_returns_execution_error_without_downstream_calls(
        self,
        mock_build_ai_client,
    ) -> None:
        request = _request()
        request_before = copy.deepcopy(request)
        mock_build_ai_client.return_value.generate_text.side_effect = _ProviderError(
            "secret auth failure",
            status_code=401,
            code="authentication_error",
        )

        raw_response = execute_quality_evaluator_prompt(request)

        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertEqual(
            raw_response.execution_diagnostics["provider_error_category"],
            PROVIDER_ERROR_AUTHENTICATION,
        )
        self.assertEqual(raw_response.execution_diagnostics["provider_http_status"], 401)
        self.assertFalse(raw_response.execution_diagnostics["provider_error_retryable"])
        self.assertEqual(
            raw_response.execution_diagnostics["provider_endpoint_family"],
            "responses",
        )
        self.assertEqual(request, request_before)

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_provider_value_error_returns_sanitized_execution_error(
        self,
        mock_build_ai_client,
    ) -> None:
        mock_build_ai_client.return_value.generate_text.side_effect = ValueError(
            "secret provider details"
        )

        raw_response = execute_quality_evaluator_prompt(_request())

        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertNotIn("secret provider details", json.dumps(raw_response.to_dict()))

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_unsupported_provider_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_quality_evaluator_prompt(
            _request(provider="gemini", model="gpt-4.1-2025-04-14")
        )

        self.assertEqual(
            raw_response.execution_error,
            "unsupported PostFlow final post role/provider/model: role=quality_evaluator provider=gemini model=gpt-4.1-2025-04-14",
        )
        mock_build_ai_client.assert_not_called()


    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_gemini_quality_evaluator_config_uses_injected_client_without_constructing_provider(
        self,
        mock_build_ai_client,
    ) -> None:
        request = _request(provider="gemini", model="gemini-3.6-flash")
        fake_client = RecordingQualityEvaluatorClient(
            SimpleNamespace(text='{"pass": true}', raw={"id": "gemini-fake"}, usage={"total_tokens": 9})
        )

        raw_response = execute_quality_evaluator_prompt(request, client=fake_client)

        mock_build_ai_client.assert_not_called()
        self.assertIsNone(raw_response.execution_error)
        self.assertEqual(raw_response.provider, "gemini")
        self.assertEqual(raw_response.model, "gemini-3.6-flash")
        self.assertEqual(fake_client.kwargs["json_mode"], True)

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_anthropic_quality_evaluator_config_returns_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_quality_evaluator_prompt(
            _request(provider="anthropic", model="claude-sonnet-5")
        )

        self.assertIn(
            "unsupported PostFlow final post role/provider/model",
            raw_response.execution_error,
        )
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_missing_provider_returns_distinct_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_quality_evaluator_prompt(
            _request(provider="", model="gpt-4.1-2025-04-14")
        )

        self.assertEqual(
            raw_response.execution_error,
            "missing quality evaluator provider",
        )
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_invalid_max_output_tokens_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        invalid_values = (True, False, "2400", None, 0, -1)

        for invalid_value in invalid_values:
            with self.subTest(max_output_tokens=invalid_value):
                raw_response = execute_quality_evaluator_prompt(
                    _request(max_output_tokens=invalid_value)
                )

                self.assertEqual(
                    raw_response.execution_error,
                    "invalid quality evaluator max_output_tokens: "
                    "must be a positive integer",
                )
                self.assertEqual(raw_response.raw_text, "")

        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_too_small_max_output_tokens_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        for invalid_value in (1, 900, MIN_MAX_OUTPUT_TOKENS - 1):
            with self.subTest(max_output_tokens=invalid_value):
                raw_response = execute_quality_evaluator_prompt(
                    _request(max_output_tokens=invalid_value)
                )

                self.assertEqual(
                    raw_response.execution_error,
                    f"invalid quality evaluator max_output_tokens: must be at least {MIN_MAX_OUTPUT_TOKENS}",
                )
                self.assertEqual(raw_response.raw_text, "")

        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_invalid_json_mode_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_quality_evaluator_prompt(_request(json_mode="true"))

        self.assertEqual(
            raw_response.execution_error,
            "invalid quality evaluator json_mode: must be a boolean",
        )
        self.assertEqual(raw_response.raw_text, "")
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_missing_prompt_content_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_quality_evaluator_prompt(_request(prompt_text=" "))

        self.assertEqual(
            raw_response.execution_error,
            "missing quality evaluator prompt text",
        )
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_blank_rendered_input_text_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_quality_evaluator_prompt(
            _request(render=_render(input_text=" \n\t "))
        )

        self.assertEqual(
            raw_response.execution_error,
            "missing quality evaluator rendered input text",
        )
        self.assertEqual(raw_response.raw_text, "")
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_missing_model_returns_execution_error_without_provider_call(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_response = execute_quality_evaluator_prompt(_request(model=""))

        self.assertEqual(raw_response.execution_error, "missing quality evaluator model")
        mock_build_ai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_raw_markdown_fenced_json_is_preserved_unparsed(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_text = '```json\n{"passed": true}\n```'
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=raw_text,
            raw={},
            usage={},
        )

        raw_response = execute_quality_evaluator_prompt(_request())

        self.assertEqual(raw_response.raw_text, raw_text)
        self.assertNotIn("pass", raw_response.to_dict())

    @patch("services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client")
    def test_unicode_raw_response_is_preserved(
        self,
        mock_build_ai_client,
    ) -> None:
        raw_text = (
            '{"notes": ["'
            "\u0447\u0435\u043b\u043e\u0432\u0435\u0447\u043d\u043e\u0441\u0442\u044c "
            "\u0442\u0435\u043a\u0441\u0442\u0430"
            '"]}'
        )
        mock_build_ai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=raw_text,
            raw={},
            usage={},
        )

        raw_response = execute_quality_evaluator_prompt(_request())

        self.assertEqual(raw_response.raw_text, raw_text)

    def test_raw_response_to_dict_omits_unavailable_optional_metadata(self) -> None:
        response = QualityEvaluatorRawResponse(
            raw_text="Raw",
            provider="openai",
            model="gpt-4.1-2025-04-14",
        )

        serialized = response.to_dict()

        self.assertEqual(set(serialized), {"raw_text", "provider", "model"})

    def test_response_to_dict_is_json_serializable(self) -> None:
        response = QualityEvaluatorRawResponse(
            raw_text='{"pass": true}',
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_metadata=_prompt_metadata(),
            usage={"total_tokens": 10},
            raw_provider_response={"id": "resp"},
        )

        serialized = json.dumps(response.to_dict(), sort_keys=True)

        self.assertIn("gpt-4.1-2025-04-14", serialized)

    def test_execution_module_has_no_parser_decision_repair_or_runtime_dependencies(self) -> None:
        source = inspect.getsource(linkedin_post_quality_evaluator_execution)

        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)
        self.assertNotIn("ContentPackage", source)
        self.assertNotIn("django.db", source)
        self.assertNotIn("apps.packaging.models", source)
        self.assertNotIn("normalize_quality_review_result", source)
        self.assertNotIn("FinalPostDecisionController", source)
        self.assertNotIn("TargetedRepairPlan", source)
        self.assertNotIn("RepairAgent", source)
        self.assertNotIn("run_final_post_deterministic_gate", source)


class RecordingQualityEvaluatorClient:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.call_count = 0
        self.kwargs: dict = {}

    def generate_text(self, **kwargs) -> SimpleNamespace:
        self.call_count += 1
        self.kwargs = kwargs
        return self.response


def _request(
    *,
    render: QualityEvaluatorPromptRender | None = None,
    prompt_text: str = "Quality evaluator prompt.",
    provider: str = "openai",
    model: str = "gpt-4.1-2025-04-14",
    max_output_tokens: object = DEFAULT_MAX_OUTPUT_TOKENS,
    json_mode: object = DEFAULT_JSON_MODE,
    execution_metadata: dict | None = None,
) -> QualityEvaluatorExecutionRequest:
    return QualityEvaluatorExecutionRequest(
        rendered_prompt_input=render or _render(),
        prompt_text=prompt_text,
        provider=provider,
        model=model,
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
        execution_metadata=execution_metadata,
    )


def _render(
    *,
    input_text: str = "## CANDIDATE_PAYLOAD_JSON\n{}",
) -> QualityEvaluatorPromptRender:
    return QualityEvaluatorPromptRender(
        prompt_name="final_post_quality_evaluator",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_quality_evaluator.txt",
        variables={
            "candidate_payload_json": "{}",
            "post_brief_json": "{}",
            "angle_decision_json": "{}",
            "selected_evidence_json": "[]",
            "quality_rubric_json": "{}",
        },
        input_text=input_text,
    )


def _prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_name="final_post_quality_evaluator",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_quality_evaluator.txt",
    )


class _ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
