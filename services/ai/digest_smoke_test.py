"""Minimal AI helpers for per-article digest analysis."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from apps.ai.client import OpenAIClient, estimate_cost_usd
from services.ai.prompt_builder import build_prompt as render_prompt
from services.ai.validators import DigestPayloadValidationError, validate_digest_payload
from services.sources import get_demo_articles_for_topic

logger = logging.getLogger(__name__)


class DigestSmokeTestError(DigestPayloadValidationError):
    """Structured validation or smoke test error for digest payloads."""


@dataclass(frozen=True)
class DigestSmokeTestResult:
    prompt: str
    response_text: str
    payload: dict[str, Any] | None
    is_mock: bool
    validation_passed: bool
    provider: str
    fallback_reason: str
    error_message: str


@dataclass(frozen=True)
class DigestGenerationPayload:
    prompt: str
    response_text: str
    payload: dict[str, Any]
    is_mock: bool
    provider: str
    fallback_reason: str
    tokens: dict[str, int | None] | None
    estimated_cost_usd: float | None
    articles: list[dict[str, Any]]


def run_digest_smoke_test(topic_name: str) -> DigestSmokeTestResult:
    """Build prompts, get real or mock responses, and validate the digest payload."""
    articles = get_demo_articles_for_topic(topic_name)
    try:
        generation = generate_digest_payload(topic_name, articles)
        return DigestSmokeTestResult(
            prompt=generation.prompt,
            response_text=generation.response_text,
            payload=generation.payload,
            is_mock=generation.is_mock,
            validation_passed=True,
            provider=generation.provider,
            fallback_reason=generation.fallback_reason,
            error_message="",
        )
    except Exception as exc:  # noqa: BLE001 - smoke test should finish in a controlled way
        return DigestSmokeTestResult(
            prompt="",
            response_text="",
            payload=None,
            is_mock=False,
            validation_passed=False,
            provider="failed",
            fallback_reason="",
            error_message=str(exc),
        )


def generate_digest_payload(topic_name: str, articles: list[dict[str, Any]]) -> DigestGenerationPayload:
    """Analyze each article separately and build the article-based digest payload."""
    prompts: list[str] = []
    response_texts: list[str] = []
    analyzed_articles: list[dict[str, Any]] = []
    provider = "openai"
    is_mock = False
    fallback_reasons: list[str] = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    has_real_usage = False

    for article in articles:
        analysis, debug = analyze_single_article(article)
        analyzed_articles.append(analysis)
        prompts.append(debug["prompt"])
        response_texts.append(debug["response_text"])
        provider = "mock" if debug["is_mock"] else provider
        is_mock = is_mock or debug["is_mock"]
        if debug["fallback_reason"]:
            fallback_reasons.append(debug["fallback_reason"])

        usage = debug.get("tokens")
        if usage:
            for key in token_usage:
                value = usage.get(key)
                if value is not None:
                    token_usage[key] += value
                    has_real_usage = True

    payload = build_digest_payload_from_articles(topic_name, analyzed_articles)
    validate_digest_payload(payload)

    estimated_cost = estimate_cost_usd(
        token_usage["prompt_tokens"] if has_real_usage else None,
        token_usage["completion_tokens"] if has_real_usage else None,
    )

    return DigestGenerationPayload(
        prompt="\n\n---\n\n".join(prompt for prompt in prompts if prompt),
        response_text="\n\n---\n\n".join(text for text in response_texts if text),
        payload=payload,
        is_mock=is_mock,
        provider=provider,
        fallback_reason=" | ".join(reason for reason in fallback_reasons if reason),
        tokens=token_usage if has_real_usage else None,
        estimated_cost_usd=estimated_cost,
        articles=analyzed_articles,
    )


def analyze_single_article(article: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze one article and return structured output plus debug metadata."""
    article_url = str(article.get("url", "")).strip()
    article_title = str(article.get("title", "")).strip()
    article_content = str(article.get("content") or article.get("snippet") or "").strip()
    prompt = build_prompt(article)

    if not article_content:
        analysis = _build_failed_analysis(article_url, article_title)
        return analysis, _build_article_debug(prompt, "", True, "mock", "Empty article content.", None)

    if _should_use_mock():
        response_text = _build_mock_article_response(article)
        parsed = safe_json_loads(response_text)
        if parsed is None:
            logger.warning("Digest analysis mock JSON could not be parsed; fallback analysis used.")
            analysis = _build_failed_analysis(article_url, article_title)
        else:
            analysis = _normalize_article_analysis(parsed, article_url, article_title)
        return analysis, _build_article_debug(
            prompt,
            response_text,
            True,
            "mock",
            "OPENAI_API_KEY не задан или содержит placeholder.",
            None,
        )

    try:
        response_text, usage = call_llm(prompt)
        parsed = safe_json_loads(response_text)
        if parsed is None:
            logger.warning("Digest analysis JSON could not be parsed; fallback analysis used.")
            analysis = _build_failed_analysis(article_url, article_title)
        else:
            analysis = _normalize_article_analysis(parsed, article_url, article_title)
        return analysis, _build_article_debug(prompt, response_text, False, "openai", "", usage)
    except Exception as exc:  # noqa: BLE001 - per-article fallback is intentional
        response_text = _build_mock_article_response(article)
        parsed = safe_json_loads(response_text)
        if parsed is None:
            analysis = _build_failed_analysis(article_url, article_title)
        else:
            analysis = _normalize_article_analysis(parsed, article_url, article_title)
        return analysis, _build_article_debug(
            prompt,
            response_text,
            True,
            "mock",
            f"Fallback на mock из-за ошибки per-article AI call: {exc}",
            None,
        )


def build_prompt(article: dict[str, Any]) -> str:
    """Build a prompt for a single article only."""
    return render_prompt(
        "digest/analyze_single_article.txt",
        article_title=str(article.get("title", "")).strip(),
        article_source=str(article.get("source_name") or article.get("source") or "unknown").strip(),
        article_url=str(article.get("url", "")).strip(),
        article_content=str(article.get("content") or article.get("snippet") or "").strip(),
    )


def call_llm(prompt: str) -> tuple[str, dict[str, int | None] | None]:
    """Call the LLM and return response text plus token usage."""
    response = OpenAIClient().generate_text(
        prompt=prompt,
        max_output_tokens=500,
        json_mode=True,
    )
    response_text = response.text.strip()
    if not response_text:
        raise DigestSmokeTestError("Модель вернула пустой ответ для анализа статьи.")
    return response_text, response.usage


def parse_response(response_text: str) -> dict[str, Any]:
    """Parse article-analysis JSON response."""
    payload = safe_json_loads(response_text)
    if payload is None:
        raise DigestSmokeTestError("Ответ модели пустой или не содержит валидный JSON-объект.")
    return payload


def _should_use_mock() -> bool:
    api_key = settings.OPENAI_API_KEY.strip()
    return not api_key or api_key == "sk-your-key"


def build_digest_payload_from_articles(
    topic_name: str,
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_articles = []
    for analysis in analyses:
        if not isinstance(analysis, dict):
            continue
        url = str(analysis.get("url", "")).strip()
        title = str(analysis.get("title", "")).strip()
        summary = str(analysis.get("summary", "")).strip()
        key_points = ensure_list(analysis.get("key_points"))
        if not url or not summary or summary == "Failed to extract":
            continue
        normalized_articles.append(
            {
                "url": url,
                "title": title,
                "summary": summary,
                "key_points": key_points[:5],
                "content_type": str(analysis.get("content_type", "unknown")).strip() or "unknown",
                "confidence": float(analysis.get("confidence", 0.0) or 0.0),
            }
        )

    if not normalized_articles:
        fallback_url = ""
        if analyses and isinstance(analyses[0], dict):
            fallback_url = str(analyses[0].get("url", "")).strip()
            fallback_title = str(analyses[0].get("title", "")).strip()
        else:
            fallback_title = ""
        normalized_articles.append(
            {
                "url": fallback_url or "https://invalid.local/failed-analysis",
                "title": fallback_title,
                "summary": "Failed to extract",
                "key_points": [],
                "content_type": "unknown",
                "confidence": 0.0,
            }
        )

    return {
        "version": 1,
        "title": f"Digest for {topic_name}",
        "articles": normalized_articles,
    }


def _normalize_article_analysis(payload: dict[str, Any], article_url: str, article_title: str) -> dict[str, Any]:
    validated = validate_article_analysis(payload)
    summary = validated["summary"] or "Failed to extract"
    key_points = validated["key_points"]
    content_type = validated["content_type"]
    confidence = validated["confidence"]

    if summary == "Failed to extract":
        key_points = []
        confidence = 0.0
        content_type = "unknown"

    return {
        "url": article_url,
        "title": article_title,
        "summary": summary,
        "key_points": key_points[:5],
        "content_type": content_type,
        "confidence": confidence,
    }


def _build_failed_analysis(article_url: str, article_title: str = "") -> dict[str, Any]:
    return {
        "url": article_url,
        "title": article_title,
        "summary": "Failed to extract",
        "key_points": [],
        "content_type": "unknown",
        "confidence": 0.0,
    }


def _build_article_debug(
    prompt: str,
    response_text: str,
    is_mock: bool,
    provider: str,
    fallback_reason: str,
    tokens: dict[str, int | None] | None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "response_text": response_text,
        "is_mock": is_mock,
        "provider": provider,
        "fallback_reason": fallback_reason,
        "tokens": tokens,
    }


def _build_mock_article_response(article: dict[str, Any]) -> str:
    snippet = str(article.get("snippet") or article.get("content") or "").strip()
    title = str(article.get("title", "")).strip()
    summary = snippet[:220].strip() or f"Failed to extract facts from {title}."
    if snippet and not summary.endswith("."):
        summary += "."

    key_points = [sentence.strip() for sentence in snippet.replace("!", ".").replace("?", ".").split(".")]
    key_points = [point for point in key_points if point][:3]

    lowered = f"{title.lower()} {snippet.lower()}"
    if any(word in lowered for word in ("guide", "tutorial", "how to", "step")):
        content_type = "tutorial"
    elif any(word in lowered for word in ("opinion", "argues", "thinks", "view")):
        content_type = "opinion"
    else:
        content_type = "news"

    confidence = 0.65 if key_points else 0.2
    payload = {
        "summary": summary or "Failed to extract",
        "key_points": key_points,
        "content_type": content_type,
        "confidence": confidence,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_json_candidate(response_text: str) -> str:
    text = response_text.strip()
    if not text:
        return ""

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return ""
    return text[first_brace : last_brace + 1]


def clean_llm_json(raw_text: str) -> str:
    """Clean common LLM JSON issues before parsing."""
    if not raw_text:
        return ""

    cleaned = raw_text.strip()
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("«", '"').replace("»", '"')
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = _extract_json_candidate(cleaned) or cleaned
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    return cleaned.strip()


def safe_json_loads(text: str) -> dict[str, Any] | None:
    """Try loading JSON, clean and retry if needed."""
    if not text or not text.strip():
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    cleaned = clean_llm_json(text)
    if cleaned != text:
        logger.warning("Digest analysis JSON was cleaned before parsing.")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Digest analysis JSON parsing failed after cleaning: %s", exc)
        return None

    if not isinstance(payload, dict):
        logger.warning("Digest analysis payload is not a JSON object after parsing.")
        return None
    return payload


def validate_article_analysis(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one article-analysis payload."""
    summary = _clean_summary_text(data.get("summary", ""))
    if not summary:
        summary = "Failed to extract"
        logger.warning("Digest analysis summary was missing; fallback used.")

    key_points = ensure_list(data.get("key_points"))
    filtered_key_points: list[str] = []
    inferred_content_type = None

    for point in key_points:
        normalized_point = " ".join(str(point).split()).strip()
        if not normalized_point:
            continue
        lowered = normalized_point.lower()
        if lowered in {"news", "opinion", "tutorial", "unknown"}:
            inferred_content_type = lowered
            logger.warning("Digest analysis key_points contained content_type-like value; corrected.")
            continue
        if _is_duplicate_summary_content(normalized_point, summary):
            logger.warning("Digest analysis key point duplicated summary content; removed.")
            continue
        filtered_key_points.append(normalized_point)

    if len(filtered_key_points) != len(key_points):
        logger.warning("Digest analysis key_points were corrected during normalization.")

    content_type = str(data.get("content_type", "")).strip().lower() or (inferred_content_type or "unknown")
    if content_type not in {"news", "opinion", "tutorial"}:
        logger.warning("Digest analysis content_type was invalid; fallback to unknown.")
        content_type = "unknown"

    raw_confidence = data.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        logger.warning("Digest analysis confidence was invalid; fallback to 0.0.")
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "summary": summary,
        "key_points": filtered_key_points[:5],
        "content_type": content_type,
        "confidence": confidence,
    }


def ensure_list(value: Any) -> list[str]:
    """Normalize key_points into a safe flat list of strings."""
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            if isinstance(item, dict):
                logger.warning("Digest analysis key_points contained nested object; flattening value.")
                flattened = " ".join(str(part).strip() for part in item.values() if str(part).strip())
                if flattened:
                    normalized.append(flattened)
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized[:5]

    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    return []


def _clean_summary_text(value: Any) -> str:
    summary = " ".join(str(value or "").replace("```json", " ").replace("```", " ").split()).strip()
    if not summary:
        return ""

    lowered = summary.lower()
    json_markers = ('"key_points"', '"content_type"', '"confidence"', '{"summary"', "'key_points'")
    if any(marker in lowered for marker in json_markers):
        logger.warning("Digest analysis summary contained embedded JSON markers; fallback used.")
        return ""

    return summary


def _is_duplicate_summary_content(key_point: str, summary: str) -> bool:
    normalized_point = " ".join(key_point.lower().split()).strip(" .")
    normalized_summary = " ".join(summary.lower().split()).strip(" .")
    if not normalized_point or not normalized_summary:
        return False
    return normalized_point == normalized_summary or normalized_point in normalized_summary
