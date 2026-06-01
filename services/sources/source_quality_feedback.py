"""Deterministic source-quality feedback for discovery diagnostics and planning guidance."""

from __future__ import annotations

from collections import Counter
from typing import Any
from urllib.parse import urlparse


MAX_FEEDBACK_REASONS = 3
MAX_FEEDBACK_DOMAINS = 3
MAX_FEEDBACK_TYPES = 4
MAX_FEEDBACK_GUIDANCE = 4

_PREFERRED_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "institutional / analyst report": ("analyst report", "research report", "weekly commentary"),
    "market data / flow analysis": ("ETF flows", "institutional flows", "funding rates", "open interest"),
    "on-chain analysis": ("on-chain analysis", "exchange reserves", "wallet data"),
    "market structure analysis": ("market structure", "liquidity"),
    "research paper": ("research paper",),
    "serious case study": ("case study outcomes", "what worked"),
}

_WEAK_DOMAIN_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("facebook.com", "social_profile_forum", "social/profile/forum"),
    ("instagram.com", "social_profile_forum", "social/profile/forum"),
    ("linkedin.com", "social_profile_forum", "social/profile/forum"),
    ("quora.com", "social_profile_forum", "social/profile/forum"),
    ("reddit.com", "social_profile_forum", "social/profile/forum"),
)

_PREFERRED_DOMAIN_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("blackrock.com", "institutional_analyst_report", "institutional / analyst report"),
    ("ishares.com", "institutional_analyst_report", "institutional / analyst report"),
    ("spglobal.com", "institutional_analyst_report", "institutional / analyst report"),
    ("coinshares.com", "institutional_analyst_report", "institutional / analyst report"),
    ("glassnode.com", "on_chain_analysis", "on-chain analysis"),
    ("cryptoquant.com", "on_chain_analysis", "on-chain analysis"),
    ("papers.ssrn.com", "research_paper", "research paper"),
    ("sciencedirect.com", "research_paper", "research paper"),
    ("pubsonline.informs.org", "research_paper", "research paper"),
)

_WEAK_TEXT_PATTERNS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("price prediction", "forecast", "will btc hit"), "price_prediction_live_price", "price prediction / live price"),
    (("btc price today", "live price", "to usd live price"), "price_prediction_live_price", "price prediction / live price"),
    (("for beginners", "ultimate guide", "how to invest in bitcoin", "best crypto trading strategies"), "beginner_seo_guide", "beginner / SEO guide"),
    (("market cap", "live ticker"), "generic_live_price_page", "live price / ticker page"),
)

_PREFERRED_TEXT_PATTERNS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("research paper", "weekly commentary", "analyst report", "research report"), "institutional_analyst_report", "institutional / analyst report"),
    (("etf flows", "institutional flows", "market data", "funding rates", "open interest"), "market_data_flow_analysis", "market data / flow analysis"),
    (("on-chain", "hashrate", "exchange reserves", "wallet data"), "on_chain_analysis", "on-chain analysis"),
    (("market structure", "liquidity"), "market_structure_analysis", "market structure analysis"),
    (("case study", "what worked"), "case_study", "serious case study"),
)


def build_source_quality_feedback(
    *,
    source_research_result,
    shown_candidates: list[dict[str, Any]],
    known_normalized_urls: set[str],
) -> dict[str, Any]:
    rejection_reasons: Counter[str] = Counter()
    weak_domains: Counter[tuple[str, str]] = Counter()
    weak_material_types: Counter[tuple[str, str]] = Counter()
    preferred_material_types: Counter[tuple[str, str]] = Counter()

    quality_rejected_count = 0
    known_or_duplicate_count = 0

    for candidate in source_research_result.evaluated_candidates:
        normalized_url = str(candidate.normalized_url or "").strip()
        if normalized_url and normalized_url in known_normalized_urls:
            known_or_duplicate_count += 1
        elif bool(candidate.diagnostics.get("duplicate_url")) or bool(candidate.diagnostics.get("duplicate_hostname")):
            known_or_duplicate_count += 1

        classification = classify_source_quality_pattern(
            url=candidate.url,
            title=candidate.title,
            snippet=candidate.snippet,
            source_type=candidate.candidate_type,
            rejection_reason=str(candidate.diagnostics.get("quality_rejection_reason") or "").strip(),
        )

        if candidate.status.value in {"rejected", "weak_content", "low_relevance"}:
            quality_rejected_count += 1
            for reason in candidate.rejection_reasons:
                cleaned_reason = str(reason or "").strip()
                if cleaned_reason:
                    rejection_reasons[cleaned_reason] += 1
            if classification["weak_material_type"]:
                weak_material_types[(classification["weak_material_type"], classification["weak_material_type_label"])] += 1
            if classification["weak_domain"]:
                weak_domains[(classification["weak_domain"], classification["weak_domain_reason"])] += 1
        if classification["preferred_material_type"]:
            preferred_material_types[(classification["preferred_material_type"], classification["preferred_material_type_label"])] += 1

    shown_count = len(shown_candidates)
    main_quality_issue = _derive_main_quality_issue(weak_material_types, weak_domains, rejection_reasons, quality_rejected_count, shown_count)
    planner_quality_guidance = _build_planner_quality_guidance(weak_material_types, preferred_material_types, weak_domains, rejection_reasons)

    return {
        "quality_rejected_count": quality_rejected_count,
        "known_or_duplicate_count": known_or_duplicate_count,
        "shown_count": shown_count,
        "dominant_rejection_reasons": _counter_to_reason_rows(rejection_reasons),
        "weak_domains": _counter_to_domain_rows(weak_domains),
        "weak_material_types": _counter_to_material_type_rows(weak_material_types),
        "preferred_material_types_found": _counter_to_material_type_rows(preferred_material_types),
        "main_quality_issue": main_quality_issue,
        "planner_quality_guidance": planner_quality_guidance[:MAX_FEEDBACK_GUIDANCE],
    }


def classify_source_quality_pattern(
    *,
    url: str,
    title: str = "",
    snippet: str = "",
    source_type: str = "",
    rejection_reason: str = "",
) -> dict[str, str]:
    normalized_url = str(url or "").strip()
    hostname = _normalize_hostname(normalized_url)
    text = " ".join(
        value.strip().casefold()
        for value in (hostname, normalized_url, title, snippet, source_type, rejection_reason)
        if str(value or "").strip()
    )

    weak_domain = ""
    weak_domain_reason = ""
    weak_material_type = ""
    weak_material_type_label = ""
    preferred_material_type = ""
    preferred_material_type_label = ""

    for domain, material_type, label in _WEAK_DOMAIN_PATTERNS:
        if domain in hostname:
            weak_domain = domain
            weak_domain_reason = label
            weak_material_type = material_type
            weak_material_type_label = label
            break

    if not weak_material_type:
        for needles, material_type, label in _WEAK_TEXT_PATTERNS:
            if any(needle in text for needle in needles):
                weak_material_type = material_type
                weak_material_type_label = label
                break

    for domain, material_type, label in _PREFERRED_DOMAIN_PATTERNS:
        if domain in hostname:
            preferred_material_type = material_type
            preferred_material_type_label = label
            break

    if not preferred_material_type:
        for needles, material_type, label in _PREFERRED_TEXT_PATTERNS:
            if any(needle in text for needle in needles):
                preferred_material_type = material_type
                preferred_material_type_label = label
                break

    return {
        "domain": hostname or "unknown",
        "weak_domain": weak_domain,
        "weak_domain_reason": weak_domain_reason,
        "weak_material_type": weak_material_type,
        "weak_material_type_label": weak_material_type_label,
        "preferred_material_type": preferred_material_type,
        "preferred_material_type_label": preferred_material_type_label,
    }


def _normalize_hostname(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    hostname = str(parsed.netloc or "").strip().casefold()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def _counter_to_reason_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"reason": reason, "count": count}
        for reason, count in counter.most_common(MAX_FEEDBACK_REASONS)
        if str(reason).strip()
    ]


def _counter_to_domain_rows(counter: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"domain": domain, "count": count, "reason": reason}
        for (domain, reason), count in counter.most_common(MAX_FEEDBACK_DOMAINS)
        if str(domain).strip()
    ]


def _counter_to_material_type_rows(counter: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"material_type": material_type, "label": label, "count": count}
        for (material_type, label), count in counter.most_common(MAX_FEEDBACK_TYPES)
        if str(material_type).strip()
    ]


def _derive_main_quality_issue(
    weak_material_types: Counter[tuple[str, str]],
    weak_domains: Counter[tuple[str, str]],
    rejection_reasons: Counter[str],
    quality_rejected_count: int,
    shown_count: int,
) -> str:
    if weak_material_types:
        (_, label), _ = weak_material_types.most_common(1)[0]
        return f"{label} results dominate recent rejected candidates"
    if weak_domains:
        (domain, reason), _ = weak_domains.most_common(1)[0]
        return f"{domain} repeatedly appears as low-quality {reason} content"
    if rejection_reasons:
        reason, _ = rejection_reasons.most_common(1)[0]
        return reason
    if quality_rejected_count > shown_count:
        return "Most recent provider results were rejected for quality"
    return "No dominant source-quality issue detected"


def _build_planner_quality_guidance(
    weak_material_types: Counter[tuple[str, str]],
    preferred_material_types: Counter[tuple[str, str]],
    weak_domains: Counter[tuple[str, str]],
    rejection_reasons: Counter[str],
) -> list[str]:
    guidance: list[str] = []
    weak_labels = [label for (_, label), _ in weak_material_types.most_common(MAX_FEEDBACK_TYPES)]
    preferred_labels = [label for (_, label), _ in preferred_material_types.most_common(MAX_FEEDBACK_TYPES)]
    weak_domain_names = [domain for (domain, _), _ in weak_domains.most_common(MAX_FEEDBACK_DOMAINS)]

    if weak_labels:
        if "social/profile/forum" in weak_labels:
            guidance.append("Social/profile/forum domains repeatedly produce low-quality results. Avoid query phrasing that attracts discussion threads or profile pages.")
        if "beginner / SEO guide" in weak_labels:
            guidance.append("Broad beginner or SEO-style guide phrasing is producing weak pages. Avoid 'for beginners', 'ultimate guide', or generic strategy phrasing.")
        if "price prediction / live price" in weak_labels or "live price / ticker page" in weak_labels:
            guidance.append("Price prediction and live-price pages are low-substance for this topic. Prefer analysis, flows, market structure, or report-driven framing.")
    if weak_domain_names:
        guidance.append(f"Weak domains seen repeatedly: {', '.join(weak_domain_names)}. Prefer sources with higher analytical depth.")
    if preferred_labels:
        preferred_terms = _build_preferred_query_term_suggestions(preferred_labels)
        if preferred_terms:
            guidance.append(
                f"Prefer material types like {', '.join(preferred_labels)}. Use query terms such as {', '.join(preferred_terms)}."
            )
        else:
            guidance.append(f"Prefer material types like {', '.join(preferred_labels)} in the next discovery run.")
    elif weak_labels:
        guidance.append(
            "Prefer reports, data pages, institutional flows, ETF flows, market structure, and on-chain analysis over broad social or SEO-driven phrasing."
        )
    if rejection_reasons and not guidance:
        reason, _ = rejection_reasons.most_common(1)[0]
        guidance.append(f"Recent rejected results were dominated by this issue: {reason}. Reframe queries toward higher-substance material.")
    return guidance[:MAX_FEEDBACK_GUIDANCE]


def _build_preferred_query_term_suggestions(preferred_labels: list[str]) -> list[str]:
    suggestions: list[str] = []
    seen: set[str] = set()
    for label in preferred_labels:
        for term in _PREFERRED_QUERY_TERMS.get(label, ()):
            normalized = term.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            suggestions.append(term)
            if len(suggestions) >= 5:
                return suggestions
    return suggestions
