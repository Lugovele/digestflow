"""Simple deterministic ranking for the MVP processing layer."""
from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Iterable


RANKING_KEYWORDS = ("reduced", "increase", "improved", "cut", "growth")
DEFAULT_MIN_QUALITY_SCORE = 0.4
MAX_QUALITY_SCORE = 10
PRACTICAL_TERMS = (
    "deploy",
    "deployment",
    "terraform",
    "cloud run",
    "mcp",
    "adk",
    "architecture",
    "memory",
    "testing",
    "integration",
    "multi-agent",
    "multi agent",
    "long-term memory",
    "tooling",
    "infrastructure",
)
GENERAL_PRACTICAL_TERMS = (
    "guide",
    "tips",
    "how to",
    "step-by-step",
    "step by step",
    "routine",
    "checklist",
    "exercise",
    "exercises",
    "planning",
    "plan",
    "safe",
    "safety",
    "advice",
    "support",
)
GENERAL_CREDIBILITY_TERMS = (
    "study",
    "research",
    "review",
    "evidence",
    "clinical",
    "doctor",
    "pediatrician",
    "obstetrician",
    "midwife",
    "expert",
    "cdc",
    "nih",
    "journal",
)
TECHNICAL_TOPIC_HINTS = (
    "ai",
    "llm",
    "agent",
    "agents",
    "api",
    "python",
    "django",
    "software",
    "developer",
    "devops",
    "cloud",
    "infrastructure",
    "deployment",
    "security engineering",
    "mcp",
    "automation",
    "terraform",
    "testing",
)
TOPIC_EDITORIAL_ALIGNMENT_TAGS: dict[str, tuple[str, ...]] = {
    "ai agents": (
        "ai_agents",
        "multi_agent",
        "mcp",
        "memory",
        "security",
        "auth",
        "oauth",
    ),
}
TOPIC_CORE_TITLE_SPECIFICITY_BONUS_MAP: dict[str, dict[str, float]] = {
    "ai agents": {
        "mcp": 0.55,
        "model context protocol": 0.55,
    },
}
TOPIC_SPECIFICITY_SIGNAL_MAP: dict[str, tuple[str, ...]] = {
    "ai agents": (
        "ai agent",
        "ai agents",
        "agentic workflow",
        "agent orchestration",
        "multi-agent",
        "multi agent",
        "autonomous agent",
        "autonomous agents",
        "tool use",
        "mcp",
        "model context protocol",
        "memory",
        "long-term memory",
        "long term memory",
        "agent framework",
        "agent deployment",
        "agent authorization",
        "agent permissions",
        "agent security",
        "planner/executor",
        "planner executor",
        "task delegation",
        "human-in-the-loop agents",
        "human in the loop agents",
    ),
}
TOPIC_RELEVANCE_STRONG_SIGNAL_MAP: dict[str, tuple[str, ...]] = {
    "ai agents": (
        "ai agents",
        "ai agent",
        "agentic",
        "multi-agent",
        "multi agent",
        "autonomous agent",
        "autonomous agents",
        "agent system",
        "agent systems",
        "agent framework",
        "tool-using agent",
        "tool using agent",
        "agent memory",
        "long-term memory",
        "long term memory",
        "agent orchestration",
        "model context protocol",
        "mcp",
        "agent authorization",
        "agent deployment",
        "agent permissions",
        "agent security",
    ),
}
TOPIC_RELEVANCE_WEAK_SIGNAL_MAP: dict[str, tuple[str, ...]] = {
    "ai agents": (
        "ai",
        "llm",
        "llms",
        "claude",
        "gemini",
        "openai",
        "app",
        "helper",
        "automation",
        "cloud",
        "deployment",
        "workflow",
        "docker",
    ),
}
TOPIC_RELEVANCE_MISSING_HINTS: dict[str, tuple[str, ...]] = {
    "ai agents": ("ai agents", "multi-agent", "agent orchestration"),
}
GENERIC_TOPIC_SIGNAL_MAP: dict[str, tuple[str, ...]] = {
    "ai agents": (
        "ai",
        "llm",
        "llms",
        "chatbot",
        "gemini",
        "openai",
        "automation",
        "helper",
        "app",
        "workflow",
        "cloud",
        "deployment",
    ),
}
NOVELTY_TERMS = (
    "research",
    "report",
    "benchmark",
    "study",
    "framework",
    "capability",
    "capabilities",
    "long-term memory",
    "personalized",
    "google adk",
)
ARTICLE_TYPE_REWARD = 0.5
ARTICLE_TYPE_PENALTY = -0.5
EVENT_TITLE_TERMS = ("winners", "challenge", "congrats", "roundup", "event recap", "event")
ANNOUNCEMENT_TITLE_TERMS = ("introducing", "launching", "release", "released", "new update", "version")
PROMOTION_TERMS = ("book a demo", "pricing", "customer story", "contact sales")
INSTRUCTIONAL_TERMS = ("how to", "guide", "step-by-step", "step by step", "tutorial")
IMPLEMENTATION_TERMS = (
    "build",
    "deploy",
    "deployment",
    "implement",
    "testing",
    "local testing",
    "walkthrough",
    "architect a",
    "architect an",
)
DEEP_TECHNICAL_TERMS = (
    "architecture",
    "protocol",
    "system design",
    "benchmark",
    "performance",
    "internals",
    "mechanism",
    "mechanisms",
)
SYSTEM_DESIGN_TERMS = (
    "designing",
    "team of agents",
    "agent architecture",
    "architecture pattern",
    "architecture patterns",
    "how agents coordinate",
    "agent coordination",
    "coordination patterns",
    "planner",
    "executor",
    "system design",
)
SECURITY_STRONG_TERMS = (
    "oauth",
    "token exchange",
    "authorization",
    "authentication",
    "access control",
    "scoped tokens",
    "ephemeral tokens",
    "static api keys",
    "identity",
    "security policy",
    "governance policy",
)
SECURITY_SUPPORT_TERMS = (
    "authorize",
    "permissions",
    "permission",
    "credentials",
    "credential",
    "governance",
)
CASE_STUDY_ACTION_TERMS = ("uses", "builds", "helps", "reduces", "improves")
CASE_STUDY_SUBJECT_TERMS = ("company", "team", "startup", "platform", "customer", "business")
OPINION_TERMS = ("analysis", "why", "lessons", "tradeoffs", "trend", "trends")
GENERIC_INFRASTRUCTURE_TAGS = {"cloud", "devops", "google_cloud", "testing"}
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "how",
    "in",
    "of",
    "on",
    "the",
    "to",
    "using",
    "with",
}


def rank_source_items(
    items: list[dict],
    *,
    keywords: Iterable[str] | None = None,
    excluded_keywords: Iterable[str] | None = None,
    top_n: int = 3,
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
) -> tuple[list[dict], list[dict]]:
    """Score items deterministically and return selected items plus ranking metadata."""
    normalized_keywords = [term for term in (_normalize_term(k) for k in (keywords or [])) if term]
    normalized_excluded = [
        term for term in (_normalize_term(k) for k in (excluded_keywords or [])) if term
    ]

    scored_items: list[tuple[float, float, int, dict, dict]] = []
    for index, item in enumerate(items):
        scoring_details = _score_item(
            item,
            normalized_keywords,
            normalized_excluded,
        )
        scored_items.append(
            (
                scoring_details["score"],
                scoring_details["quality_score"],
                index,
                item,
                scoring_details,
            )
        )

    scored_items.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))

    ranking_scores = [
        {
            "article_id": item.get("id") or item.get("article_id"),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "source_name": item.get("source_name") or item.get("source") or "",
            "topic_domain": scoring_details["topic_domain"],
            "content_tier": scoring_details["content_tier"],
            "content_length": scoring_details["content_length"],
            "score": raw_score,
            "quality_score": quality_score,
            "final_quality_score": quality_score,
            "topic_relevance_score": scoring_details["topic_relevance_score"],
            "topic_relevance_reason": scoring_details["topic_relevance_reason"],
            "relevance_signals": scoring_details["relevance_signals"],
            "weak_relevance_signals": scoring_details["weak_relevance_signals"],
            "missing_relevance_signals": scoring_details["missing_relevance_signals"],
            "topic_specificity_score": scoring_details["topic_specificity_score"],
            "topic_specificity_reason": scoring_details["topic_specificity_reason"],
            "specificity_signals": scoring_details["specificity_signals"],
            "generic_topic_signals": scoring_details["generic_topic_signals"],
            "evidence_score": scoring_details["evidence_score"],
            "practical_value_score": scoring_details["practical_value_score"],
            "novelty_score": scoring_details["novelty_score"],
            "primary_article_type": scoring_details["primary_article_type"],
            "secondary_article_tags": scoring_details["secondary_article_tags"],
            "weighted_secondary_tags": scoring_details["weighted_secondary_tags"],
            "dominant_tags": scoring_details["dominant_tags"],
            "supporting_tags": scoring_details["supporting_tags"],
            "weak_tags": scoring_details["weak_tags"],
            "article_type": scoring_details["article_type"],
            "article_type_reason": scoring_details["article_type_reason"],
            "article_type_score_modifier": scoring_details["article_type_score_modifier"],
            "classification_signal_summary": scoring_details["classification_signal_summary"],
            "dominant_theme_reason": scoring_details["dominant_theme_reason"],
            "primary_type_override_reason": scoring_details["primary_type_override_reason"],
            "heading_diagnostics": scoring_details["heading_diagnostics"],
            "quality_reasons": scoring_details["quality_reasons"],
            "rejection_reasons": scoring_details["rejection_reasons"],
            "diagnostic_warnings": scoring_details["diagnostic_warnings"],
            "diversity_penalty": 0.0,
            "similarity_reasons": [],
            "diversity_adjusted_score": raw_score,
            "scoring_mode": "rule_based",
        }
        for raw_score, quality_score, _, item, scoring_details in scored_items
    ]

    filtered_entries = [
        (
            item,
            ranking_scores[index],
        )
        for index, (_raw_score, quality_score, _position, item, _scoring_details) in enumerate(scored_items)
        if quality_score >= min_quality_score
    ]
    selected_items = _select_diverse_items(filtered_entries, top_n=top_n)
    return selected_items, ranking_scores


def _score_item(
    item: dict,
    normalized_keywords: list[str],
    normalized_excluded: list[str],
) -> dict:
    title = str(item.get("title", ""))
    snippet = str(item.get("snippet", ""))
    content = str(item.get("content", ""))
    source_name = str(item.get("source_name") or item.get("source") or "")
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    content_lower = content.lower()
    text_blob = f"{title_lower} {snippet_lower} {content_lower}"
    source_lower = source_name.lower()
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    content_tier = str(metadata.get("content_tier") or "unknown")
    content_length = int(metadata.get("content_length") or len(content) or len(snippet) or 0)
    heading_diagnostics = _build_heading_diagnostics(metadata, content)
    headings_lower = " ".join(heading_diagnostics["normalized_headings"])
    intro_lower = snippet_lower or content_lower[:280]
    topic_domain = _detect_topic_domain(normalized_keywords)
    (
        primary_article_type,
        secondary_article_tags,
        weighted_secondary_tags,
        dominant_tags,
        supporting_tags,
        weak_tags,
        article_type_reason,
        article_type_score_modifier,
        classification_signal_summary,
        dominant_theme_reason,
        primary_type_override_reason,
    ) = _classify_article_type(
        title_lower=title_lower,
        text_blob=text_blob,
        intro_lower=intro_lower,
        body_lower=content_lower,
        headings_lower=headings_lower,
        headings_normalized=heading_diagnostics["normalized_headings"],
        snippet_lower=snippet_lower,
        source_lower=source_lower,
        content_tier=content_tier,
        content_length=content_length,
        topic_domain=topic_domain,
    )

    quality_reasons: list[str] = []
    diagnostic_warnings: list[str] = []
    rejection_reasons: list[str] = []

    if not normalized_keywords:
        diagnostic_warnings.append("no topic keywords were provided; relevance is based on article signals only")
    if not snippet.strip() and not content.strip():
        diagnostic_warnings.append("article text is sparse; relevance is based mostly on title")

    (
        topic_relevance_score,
        topic_relevance_reason,
        relevance_signals,
        weak_relevance_signals,
        missing_relevance_signals,
    ) = _score_topic_relevance(title_lower, text_blob, normalized_keywords)
    (
        topic_specificity_score,
        topic_specificity_reason,
        specificity_signals,
        generic_topic_signals,
    ) = _score_topic_specificity(
        title_lower,
        text_blob,
        normalized_keywords,
        headings_lower=headings_lower,
        topic_domain=topic_domain,
    )
    if topic_relevance_score >= 3:
        quality_reasons.append("strong relevance to topic")
    elif topic_relevance_score >= 1:
        quality_reasons.append("relevant topic signal present")
    elif normalized_keywords:
        quality_reasons.append("weak relevance to topic")
        rejection_reasons.append("low relevance")

    excluded_hits = sum(1 for keyword in normalized_excluded if keyword and keyword in text_blob)
    if excluded_hits:
        topic_relevance_score = max(topic_relevance_score - min(excluded_hits, 2), 0.0)
        quality_reasons.append("too narrow for selected topic")
        rejection_reasons.append("low relevance")

    if topic_specificity_score >= 1.5:
        quality_reasons.append("strong topic specificity")
    elif topic_specificity_score >= 0.5:
        quality_reasons.append("moderate topic specificity")
    elif normalized_keywords:
        quality_reasons.append("broad topic match without strong specificity")
        rejection_reasons.append("low specificity")

    evidence_score = _score_evidence(snippet, content_length)
    if evidence_score >= 1.5:
        quality_reasons.append("sufficient evidence/detail")
    elif evidence_score <= 0.5:
        quality_reasons.append("insufficient evidence/detail")
        rejection_reasons.append("insufficient detail")

    practical_value_score = _score_practical_value(text_blob, snippet_lower, content_tier, topic_domain=topic_domain)
    if practical_value_score >= 1.5:
        quality_reasons.append("good practical guidance" if topic_domain == "general" else "good technical/practical article")
    elif practical_value_score <= 0.5:
        quality_reasons.append("low practical value")
        rejection_reasons.append("low practical value")

    novelty_score = _score_novelty(text_blob, source_lower, topic_domain=topic_domain)
    if novelty_score >= 1.5:
        quality_reasons.append("strong evidence or source credibility" if topic_domain == "general" else "high novelty or strategic value")
    elif novelty_score <= 0.5:
        quality_reasons.append("low novelty")
        rejection_reasons.append("low novelty")

    if primary_article_type in {
        "tutorial",
        "deep_technical",
        "system_design",
        "architecture_security",
        "case_study",
        "opinion_analysis",
    }:
        quality_reasons.append(f"{primary_article_type.replace('_', ' ')} article")
    elif primary_article_type in {"community_update", "marketing", "lightweight_post"}:
        rejection_reasons.append(primary_article_type.replace("_", " "))

    if normalized_keywords and secondary_article_tags:
        keyword_tag_matches = _count_matching_tags(normalized_keywords, secondary_article_tags)
        if keyword_tag_matches:
            quality_reasons.append("topic-specific domain tags detected")

    editorial_alignment_bonus, editorial_alignment_reason = _score_editorial_alignment(
        normalized_keywords,
        dominant_tags,
        supporting_tags,
    )
    if editorial_alignment_reason:
        quality_reasons.append(editorial_alignment_reason)

    if _looks_promotional(title_lower, snippet_lower):
        quality_reasons.append("promotional/product announcement")
        rejection_reasons.append("low practical value")
    if _looks_opinionated(title_lower, snippet_lower):
        quality_reasons.append("mostly personal opinion")
        rejection_reasons.append("insufficient detail")

    score = round(
        max(topic_relevance_score, 0.0)
        + max(topic_specificity_score, 0.0)
        + max(evidence_score, 0.0)
        + max(practical_value_score, 0.0)
        + max(novelty_score, 0.0),
        2,
    )
    score = round(score + editorial_alignment_bonus, 2)
    score = round(max(score + article_type_score_modifier, 0.0), 2)
    normalized_score = round(max(score, 0.0) / MAX_QUALITY_SCORE, 2)
    heading_diagnostics["matched_heading_tags"] = _build_matched_heading_tags(
        heading_diagnostics,
        weighted_secondary_tags,
    )
    return {
        "score": score,
        "quality_score": min(1.0, normalized_score),
        "topic_relevance_score": round(topic_relevance_score, 2),
        "topic_relevance_reason": topic_relevance_reason,
        "relevance_signals": relevance_signals,
        "weak_relevance_signals": weak_relevance_signals,
        "missing_relevance_signals": missing_relevance_signals,
        "topic_specificity_score": round(topic_specificity_score, 2),
        "topic_specificity_reason": topic_specificity_reason,
        "specificity_signals": specificity_signals,
        "generic_topic_signals": generic_topic_signals,
        "evidence_score": round(evidence_score, 2),
        "practical_value_score": round(practical_value_score, 2),
        "novelty_score": round(novelty_score, 2),
        "primary_article_type": primary_article_type,
        "secondary_article_tags": secondary_article_tags,
        "weighted_secondary_tags": weighted_secondary_tags,
        "dominant_tags": dominant_tags,
        "supporting_tags": supporting_tags,
        "weak_tags": weak_tags,
        "article_type": primary_article_type,
        "article_type_reason": article_type_reason,
        "article_type_score_modifier": article_type_score_modifier,
        "classification_signal_summary": classification_signal_summary,
        "dominant_theme_reason": dominant_theme_reason,
        "primary_type_override_reason": primary_type_override_reason,
        "heading_diagnostics": heading_diagnostics,
        "topic_domain": topic_domain,
        "quality_reasons": _unique_reasons(quality_reasons),
        "rejection_reasons": _unique_reasons(rejection_reasons),
        "diagnostic_warnings": _unique_reasons(diagnostic_warnings),
        "content_tier": content_tier,
        "content_length": content_length,
    }


def _score_topic_relevance(
    title_lower: str,
    text_blob: str,
    normalized_keywords: list[str],
) -> tuple[float, str, list[str], list[str], list[str]]:
    if not normalized_keywords:
        return 0.0, "no topic keywords were provided", [], [], []

    title_tokens = _tokenize_text(title_lower)
    blob_tokens = _tokenize_text(text_blob)
    keyword_blob = " ".join(normalized_keywords)
    strongest_keyword = next((keyword for keyword in TOPIC_RELEVANCE_STRONG_SIGNAL_MAP if keyword in keyword_blob), "")

    if strongest_keyword:
        strong_terms = TOPIC_RELEVANCE_STRONG_SIGNAL_MAP.get(strongest_keyword, ())
        weak_terms = TOPIC_RELEVANCE_WEAK_SIGNAL_MAP.get(strongest_keyword, ())
        missing_hints = list(TOPIC_RELEVANCE_MISSING_HINTS.get(strongest_keyword, ()))
        explicit_strong_title_hits = _collect_matches(title_lower, strong_terms)
        explicit_strong_body_hits = _collect_repeated_body_matches(
            text_blob,
            (term for term in strong_terms if _is_explicit_specificity_phrase(term)),
            min_hits=1,
        )
        supporting_strong_body_hits = _collect_repeated_body_matches(
            text_blob,
            (term for term in strong_terms if not _is_explicit_specificity_phrase(term)),
            min_hits=2,
        )
        relevance_signals = _unique_reasons(
            explicit_strong_title_hits + explicit_strong_body_hits + supporting_strong_body_hits
        )
        weak_relevance_signals = _collect_matches(title_lower, weak_terms)
        weak_relevance_signals.extend(
            signal
            for signal in _collect_repeated_body_matches(text_blob, weak_terms, min_hits=2)
            if signal not in weak_relevance_signals
        )

        if strongest_keyword in title_lower:
            return 4.0, "matched the topic phrase directly in the title", [strongest_keyword], weak_relevance_signals, []
        if strongest_keyword in text_blob:
            return 4.0, "matched the topic phrase directly in the article text", [strongest_keyword], weak_relevance_signals, []
        if len(explicit_strong_title_hits) >= 1:
            score = 4.0 if len(explicit_strong_title_hits) >= 2 else 3.0
            reason = "matched strong agent-system signals in the title"
            return score, reason, relevance_signals, weak_relevance_signals, []
        if len(explicit_strong_body_hits) + len(supporting_strong_body_hits) >= 2:
            return 3.0, "matched multiple strong agent-related signals in the article text", relevance_signals, weak_relevance_signals, []
        if relevance_signals:
            return 2.0, "matched some agent-related signals but not a central topic phrase", relevance_signals, weak_relevance_signals, []
        if weak_relevance_signals:
            return 1.0, "matched only broad AI-adjacent signals", [], weak_relevance_signals, missing_hints
        return 0.0, "did not match meaningful topic signals", [], [], missing_hints

    best_score = 0.0

    for keyword in normalized_keywords:
        keyword_tokens = _tokenize_text(keyword)
        if not keyword_tokens:
            continue
        keyword_phrase = " ".join(keyword_tokens)
        title_phrase = " ".join(title_tokens)
        blob_phrase = " ".join(blob_tokens)
        overlap = len(set(keyword_tokens) & set(blob_tokens))
        overlap_ratio = overlap / len(keyword_tokens)
        is_single_token_keyword = len(keyword_tokens) == 1

        if is_single_token_keyword and keyword_phrase and keyword_phrase in title_phrase:
            best_score = max(best_score, 2.0)
        elif is_single_token_keyword and keyword_phrase and keyword_phrase in blob_phrase:
            best_score = max(best_score, 1.5)
        elif keyword_phrase and keyword_phrase in title_phrase:
            best_score = max(best_score, 4.0)
        elif keyword_phrase and keyword_phrase in blob_phrase:
            best_score = max(best_score, 3.0)
        elif overlap_ratio >= 1.0:
            best_score = max(best_score, 3.0)
        elif overlap_ratio >= 0.5:
            best_score = max(best_score, 2.0)
        elif overlap > 0:
            best_score = max(best_score, 1.0)

    if best_score >= 3.0:
        reason = "matched the topic strongly"
        missing = []
    elif best_score >= 2.0:
        reason = "matched the topic at a moderate level"
        missing = []
    elif best_score > 0.0:
        reason = "matched the topic weakly"
        missing = []
    else:
        reason = "did not match meaningful topic signals"
        missing = normalized_keywords[:3]

    return best_score, reason, [], [], missing


def _score_topic_specificity(
    title_lower: str,
    text_blob: str,
    normalized_keywords: list[str],
    *,
    headings_lower: str,
    topic_domain: str,
) -> tuple[float, str, list[str], list[str]]:
    if not normalized_keywords:
        return 0.0, "no topic keywords were provided", [], []

    if topic_domain != "technical":
        return _score_general_topic_specificity(
            title_lower,
            text_blob,
            normalized_keywords,
            headings_lower=headings_lower,
        )

    keyword_blob = " ".join(normalized_keywords)
    strongest_keyword = next((keyword for keyword in TOPIC_SPECIFICITY_SIGNAL_MAP if keyword in keyword_blob), "")
    if not strongest_keyword:
        return 0.0, "no topic-specific signal model is configured for these keywords", [], []

    specificity_terms = TOPIC_SPECIFICITY_SIGNAL_MAP.get(strongest_keyword, ())
    generic_terms = GENERIC_TOPIC_SIGNAL_MAP.get(strongest_keyword, ())

    title_specific_hits = _collect_matches(title_lower, specificity_terms)
    body_explicit_hits = _collect_repeated_body_matches(
        text_blob,
        (term for term in specificity_terms if _is_explicit_specificity_phrase(term)),
        min_hits=1,
    )
    body_supporting_hits = _collect_repeated_body_matches(
        text_blob,
        (term for term in specificity_terms if not _is_explicit_specificity_phrase(term)),
        min_hits=2,
    )
    specificity_signals = _unique_reasons(title_specific_hits + body_explicit_hits + body_supporting_hits)

    generic_topic_signals = _collect_matches(title_lower, generic_terms)
    generic_topic_signals.extend(
        signal
        for signal in _collect_repeated_body_matches(text_blob, generic_terms, min_hits=2)
        if signal not in generic_topic_signals
    )

    score = 0.0
    if title_specific_hits:
        score += min(len(title_specific_hits) * 1.5, 2.0)
    elif body_explicit_hits:
        score += min(len(body_explicit_hits), 2) * 0.75

    if body_supporting_hits:
        score += min(len(body_supporting_hits), 2) * 0.5

    if len(specificity_signals) >= 3:
        score += 0.5

    core_title_bonus_map = TOPIC_CORE_TITLE_SPECIFICITY_BONUS_MAP.get(strongest_keyword, {})
    if core_title_bonus_map and title_specific_hits:
        score += max(
            (float(core_title_bonus_map.get(hit, 0.0)) for hit in title_specific_hits),
            default=0.0,
        )

    if generic_topic_signals and not specificity_signals:
        score = max(score - 0.5, 0.0)
    elif generic_topic_signals and not title_specific_hits and specificity_signals:
        score = max(score - 0.25, 0.0)

    if score >= 2.0:
        reason = "matched multiple strong topic-specific agent signals"
    elif score >= 1.0:
        reason = "matched clear topic-specific agent signals"
    elif generic_topic_signals:
        reason = "matched broad AI signals without strong agent specificity"
    else:
        reason = "no strong topic-specific signals were detected"

    return min(score, 2.5), reason, _unique_reasons(specificity_signals), _unique_reasons(generic_topic_signals)


def _score_general_topic_specificity(
    title_lower: str,
    text_blob: str,
    normalized_keywords: list[str],
    *,
    headings_lower: str,
) -> tuple[float, str, list[str], list[str]]:
    exact_title_hits = _collect_matches(title_lower, normalized_keywords)
    exact_heading_hits = _collect_matches(headings_lower, normalized_keywords)
    exact_body_hits = _collect_repeated_body_matches(text_blob, normalized_keywords, min_hits=1)

    support_terms = _general_specificity_support_terms(normalized_keywords)
    support_title_hits = _collect_matches(title_lower, support_terms)
    support_heading_hits = _collect_matches(headings_lower, support_terms)
    support_body_hits = _collect_repeated_body_matches(text_blob, support_terms, min_hits=2)

    specificity_signals = _unique_reasons(
        exact_title_hits
        + exact_heading_hits
        + exact_body_hits
        + support_title_hits
        + support_heading_hits
        + support_body_hits
    )
    generic_topic_signals = _unique_reasons(support_title_hits + support_heading_hits + support_body_hits)

    score = 0.0
    if exact_title_hits:
        score += min(len(exact_title_hits) * 1.25, 1.5)
    if exact_heading_hits:
        score += min(len(exact_heading_hits) * 0.75, 1.0)
    if exact_body_hits:
        score += min(len(exact_body_hits) * 0.5, 0.75)
    if support_title_hits or support_heading_hits:
        score += 0.5
    if len(support_body_hits) >= 2:
        score += 0.5
    elif support_body_hits:
        score += 0.25

    if len(specificity_signals) >= 3:
        score += 0.25

    if score >= 2.0:
        reason = "matched multiple clear topic or focus terms"
    elif score >= 1.0:
        reason = "matched clear topic or focus terms"
    elif generic_topic_signals:
        reason = "matched broad topic wording with limited focus specificity"
    else:
        reason = "no strong topic-specific signals were detected"

    return min(score, 2.5), reason, specificity_signals, generic_topic_signals


def _score_evidence(snippet: str, content_length: int) -> float:
    if content_length >= 900:
        return 2.0
    if content_length >= 400:
        return 1.5
    if content_length >= 200 or len(snippet) >= 180:
        return 1.0
    if content_length >= 120 or len(snippet) >= 120:
        return 0.5
    return 0.0


def _score_practical_value(text_blob: str, snippet_lower: str, content_tier: str, *, topic_domain: str) -> float:
    score = 0.0
    practical_terms = PRACTICAL_TERMS if topic_domain == "technical" else GENERAL_PRACTICAL_TERMS
    if any(keyword in text_blob for keyword in practical_terms):
        score += 1.0
    if re.search(r"\d|%", snippet_lower) or any(keyword in snippet_lower for keyword in RANKING_KEYWORDS):
        score += 1.0
    if content_tier == "full_article":
        score += 0.5
    return min(score, 2.0)


def _score_novelty(text_blob: str, source_lower: str, *, topic_domain: str) -> float:
    score = 0.0
    if "research" in source_lower or "report" in source_lower or (topic_domain == "general" and any(term in source_lower for term in ("health", "medical", "journal", "cdc", "nih"))):
        score += 1.0
    novelty_terms = NOVELTY_TERMS if topic_domain == "technical" else GENERAL_CREDIBILITY_TERMS
    if any(keyword in text_blob for keyword in novelty_terms):
        score += 1.0
    return min(score, 2.0)


def _classify_article_type(
    *,
    title_lower: str,
    text_blob: str,
    intro_lower: str,
    body_lower: str,
    headings_lower: str,
    headings_normalized: list[str],
    snippet_lower: str,
    source_lower: str,
    content_tier: str,
    content_length: int,
    topic_domain: str,
) -> tuple[
    str,
    list[str],
    dict[str, dict[str, object]],
    list[str],
    list[str],
    list[str],
    str,
    float,
    dict[str, list[str] | str],
    str,
    str | None,
]:
    if topic_domain != "technical":
        return _classify_general_article_type(
            title_lower=title_lower,
            text_blob=text_blob,
            intro_lower=intro_lower,
            headings_lower=headings_lower,
            content_tier=content_tier,
            content_length=content_length,
            source_lower=source_lower,
        )

    (
        secondary_tags,
        weighted_secondary_tags,
        dominant_tags,
        supporting_tags,
        weak_tags,
        tag_signals,
    ) = _extract_secondary_article_tags(
        title_lower,
        intro_lower,
        headings_lower,
        headings_normalized,
        body_lower,
        text_blob,
    )
    instructional_hits = _collect_matches(text_blob, INSTRUCTIONAL_TERMS)
    implementation_hits = _collect_matches(text_blob, IMPLEMENTATION_TERMS)
    strong_security_hits = _collect_matches(text_blob, SECURITY_STRONG_TERMS)
    support_security_hits = _collect_matches(text_blob, SECURITY_SUPPORT_TERMS)
    deep_technical_hits = _collect_matches(text_blob, DEEP_TECHNICAL_TERMS)
    system_design_hits = _collect_matches(text_blob, SYSTEM_DESIGN_TERMS)
    case_study_hits = _collect_matches(text_blob, CASE_STUDY_ACTION_TERMS)
    case_subject_hits = _collect_matches(title_lower, CASE_STUDY_SUBJECT_TERMS)
    opinion_hits = _collect_matches(text_blob, OPINION_TERMS)
    title_instructional_hits = _collect_matches(title_lower, INSTRUCTIONAL_TERMS)
    title_implementation_hits = _collect_matches(title_lower, IMPLEMENTATION_TERMS)
    title_deep_technical_hits = _collect_matches(title_lower, DEEP_TECHNICAL_TERMS)
    title_system_design_hits = _collect_matches(title_lower, SYSTEM_DESIGN_TERMS)
    title_strong_security_hits = _collect_matches(title_lower, SECURITY_STRONG_TERMS)
    title_support_security_hits = _collect_matches(title_lower, SECURITY_SUPPORT_TERMS)
    intro_strong_security_hits = _collect_matches(snippet_lower, SECURITY_STRONG_TERMS)
    intro_support_security_hits = _collect_matches(snippet_lower, SECURITY_SUPPORT_TERMS)

    classification_signal_summary: dict[str, list[str] | str] = {
        "primary_signals": [],
        "tag_signals": [f"{tag}: {signal}" for tag, signal in tag_signals.items()],
    }

    title_security_hits = _unique_reasons(title_strong_security_hits + title_support_security_hits)
    intro_security_hits = _unique_reasons(intro_strong_security_hits + intro_support_security_hits)
    security_signal_count = len(_unique_reasons(strong_security_hits + support_security_hits))
    tutorial_signal_count = len(_unique_reasons(instructional_hits + implementation_hits))
    tutorial_title_focus = bool(title_instructional_hits or title_implementation_hits)
    security_title_focus = bool(title_security_hits)
    security_intro_focus = bool(intro_strong_security_hits) or len(intro_security_hits) >= 2
    explicit_security_focus = bool(
        {"oauth", "token exchange", "access control"} & set(title_strong_security_hits + intro_strong_security_hits)
    )
    primary_type_override_reason: str | None = None

    def build_result(
        primary_type: str,
        article_type_reason_value: str,
        article_type_score_modifier_value: float,
        primary_signals: list[str],
        *,
        dominant_theme_reason: str,
        override_reason: str | None = None,
    ) -> tuple[
        str,
        list[str],
        dict[str, dict[str, object]],
        list[str],
        list[str],
        list[str],
        str,
        float,
        dict[str, list[str] | str],
        str,
        str | None,
    ]:
        classification_signal_summary["primary_signals"] = primary_signals
        return (
            primary_type,
            secondary_tags,
            weighted_secondary_tags,
            dominant_tags,
            supporting_tags,
            weak_tags,
            article_type_reason_value,
            article_type_score_modifier_value,
            classification_signal_summary,
            dominant_theme_reason,
            override_reason,
        )

    if any(term in text_blob for term in PROMOTION_TERMS):
        return build_result(
            "marketing",
            "matched explicit product-promotion language",
            ARTICLE_TYPE_PENALTY,
            ["promotion language"],
            dominant_theme_reason="promotional language dominates the article framing more than technical depth",
        )

    event_hits = _collect_matches(title_lower, EVENT_TITLE_TERMS)
    if event_hits:
        return build_result(
            "community_update",
            "matched community/event wording in the title",
            ARTICLE_TYPE_PENALTY,
            [f"title event signals: {', '.join(event_hits)}"],
            dominant_theme_reason="community or event framing dominates the title and editorial purpose",
        )

    announcement_hits = _collect_matches(title_lower, ANNOUNCEMENT_TITLE_TERMS)
    if announcement_hits:
        return build_result(
            "announcement",
            "matched product/update announcement wording",
            0.0,
            [f"title announcement signals: {', '.join(announcement_hits)}"],
            dominant_theme_reason="release or update wording is the article's main editorial frame",
        )

    security_dominates = explicit_security_focus or (
        security_title_focus and security_intro_focus
    ) or (
        security_title_focus and security_signal_count >= 3
    ) or (
        security_intro_focus
        and security_signal_count >= 4
        and not tutorial_title_focus
        and not title_deep_technical_hits
        and not title_system_design_hits
    )
    tutorial_dominates = tutorial_signal_count >= 1 and (
        tutorial_title_focus
        or any(tag in dominant_tags for tag in ("terraform", "cloud", "devops", "google_cloud", "testing"))
    )
    system_design_dominates = bool(title_system_design_hits) or (
        len(_unique_reasons(system_design_hits + title_deep_technical_hits)) >= 2
        and not tutorial_dominates
    )

    if security_dominates and not (tutorial_dominates and not explicit_security_focus):
        signals = _unique_reasons(title_security_hits + intro_security_hits + strong_security_hits + support_security_hits)
        return build_result(
            "architecture_security",
            "matched authorization, security, or access-control language as the main editorial focus",
            ARTICLE_TYPE_REWARD,
            [f"security signals: {', '.join(signals[:5])}"],
            dominant_theme_reason="security and authorization concerns are central in the title and/or introduction",
        )

    if tutorial_dominates:
        signals = _unique_reasons(title_instructional_hits + title_implementation_hits + instructional_hits + implementation_hits)
        if security_dominates and not explicit_security_focus:
            primary_type_override_reason = (
                "security/auth terms were present, but deployment/tutorial framing is the article's main editorial center"
            )
        elif security_signal_count >= 2 and not security_title_focus:
            primary_type_override_reason = (
                "security/auth concerns are supporting guidance here, but the editorial center remains deployment/tutorial execution"
            )
        return build_result(
            "tutorial",
            "matched instructional or implementation-oriented language",
            ARTICLE_TYPE_REWARD,
            [f"instructional signals: {', '.join(signals[:5])}"],
            dominant_theme_reason="instructional, deployment, or testing framing dominates the article structure and title intent",
            override_reason=primary_type_override_reason,
        )

    if system_design_dominates:
        signals = _unique_reasons(title_system_design_hits + system_design_hits + title_deep_technical_hits)
        return build_result(
            "system_design",
            "matched conceptual system-design or coordination framing",
            ARTICLE_TYPE_REWARD,
            [f"system design signals: {', '.join(signals[:5])}"],
            dominant_theme_reason="conceptual design, coordination, or architecture framing is central to the article purpose",
        )

    if deep_technical_hits or title_deep_technical_hits:
        signals = _unique_reasons(title_deep_technical_hits + deep_technical_hits)
        return build_result(
            "deep_technical",
            "matched deep technical/system-design language",
            ARTICLE_TYPE_REWARD,
            [f"deep technical signals: {', '.join(signals[:5])}"],
            dominant_theme_reason="technical mechanisms and system-design discussion are more central than instructional framing",
        )

    if case_study_hits and case_subject_hits:
        return build_result(
            "case_study",
            "matched applied outcome language that looks like a case study",
            ARTICLE_TYPE_REWARD,
            [f"case study signals: {', '.join((case_study_hits + case_subject_hits)[:5])}"],
            dominant_theme_reason="outcome-focused company or customer language makes the article read like a case study",
        )

    if opinion_hits and not instructional_hits:
        return build_result(
            "opinion_analysis",
            "matched analysis/tradeoff framing",
            ARTICLE_TYPE_REWARD,
            [f"analysis signals: {', '.join(opinion_hits[:5])}"],
            dominant_theme_reason="analysis and tradeoff framing are more central than implementation detail",
        )

    if content_tier == "weak_snippet" or content_length < 180:
        return build_result(
            "lightweight_post",
            "content is short and low-structure",
            ARTICLE_TYPE_PENALTY,
            ["short or low-structure content"],
            dominant_theme_reason="the article does not show enough structure to infer a stronger editorial center",
        )

    if "blog" in source_lower and content_tier == "rich_summary":
        return build_result(
            "lightweight_post",
            "summary-level blog item without stronger structural signals",
            ARTICLE_TYPE_PENALTY,
            ["summary-level blog content"],
            dominant_theme_reason="summary-style content does not show a stronger editorial format",
        )

    return build_result(
        "unknown",
        "no strong article-type signal was detected",
        0.0,
        ["no strong format signal"],
        dominant_theme_reason="no single editorial center of gravity clearly dominated the article",
    )


def _classify_general_article_type(
    *,
    title_lower: str,
    text_blob: str,
    intro_lower: str,
    headings_lower: str,
    content_tier: str,
    content_length: int,
    source_lower: str,
) -> tuple[
    str,
    list[str],
    dict[str, dict[str, object]],
    list[str],
    list[str],
    list[str],
    str,
    float,
    dict[str, list[str] | str],
    str,
    str | None,
]:
    classification_signal_summary: dict[str, list[str] | str] = {
        "primary_signals": [],
        "tag_signals": [],
    }

    def build_general_result(
        primary_type: str,
        article_type_reason_value: str,
        article_type_score_modifier_value: float,
        primary_signals: list[str],
        *,
        dominant_theme_reason: str,
    ) -> tuple[
        str,
        list[str],
        dict[str, dict[str, object]],
        list[str],
        list[str],
        list[str],
        str,
        float,
        dict[str, list[str] | str],
        str,
        str | None,
    ]:
        classification_signal_summary["primary_signals"] = primary_signals
        return (
            primary_type,
            [],
            {},
            [],
            [],
            [],
            article_type_reason_value,
            article_type_score_modifier_value,
            classification_signal_summary,
            dominant_theme_reason,
            None,
        )

    safety_hits = _collect_matches(text_blob, ("safe sleep", "safe", "safety", "warning", "prevent", "avoid", "risk", "guidance"))
    scientific_hits = _collect_matches(text_blob, ("study", "studies", "research", "review", "evidence", "clinical", "journal"))
    expert_hits = _collect_matches(text_blob, ("doctor", "pediatrician", "obstetrician", "midwife", "expert", "specialist"))
    practical_hits = _collect_matches(text_blob, ("how to", "guide", "tips", "routine", "steps", "exercise", "exercises", "planning", "plan"))
    resource_hits = _collect_matches(text_blob, ("resources", "resource", "faq", "checklist", "toolkit"))
    overview_hits = _collect_matches(text_blob, ("overview", "basics", "what is", "introduction", "understanding"))
    opinion_hits = _collect_matches(text_blob, ("perspective", "opinion", "experience", "lessons"))

    if safety_hits and (title_lower or intro_lower or headings_lower):
        return build_general_result(
            "safety_guidance",
            "matched clear safety-oriented guidance",
            ARTICLE_TYPE_REWARD,
            [f"safety signals: {', '.join(safety_hits[:5])}"],
            dominant_theme_reason="safety recommendations and prevention guidance are central to the article",
        )
    if scientific_hits:
        return build_general_result(
            "scientific_review",
            "matched research, review, or evidence-oriented language",
            ARTICLE_TYPE_REWARD,
            [f"evidence signals: {', '.join(scientific_hits[:5])}"],
            dominant_theme_reason="evidence and research framing are central to the article",
        )
    if practical_hits:
        return build_general_result(
            "practical_guide",
            "matched practical how-to or step-by-step guidance",
            ARTICLE_TYPE_REWARD,
            [f"practical signals: {', '.join(practical_hits[:5])}"],
            dominant_theme_reason="practical instructions or usable tips are the article's main purpose",
        )
    if expert_hits:
        return build_general_result(
            "expert_advice",
            "matched expert or specialist guidance",
            ARTICLE_TYPE_REWARD,
            [f"expert signals: {', '.join(expert_hits[:5])}"],
            dominant_theme_reason="expert guidance is the article's main framing",
        )
    if resource_hits:
        return build_general_result(
            "resource_page",
            "matched resource or checklist-style language",
            0.0,
            [f"resource signals: {', '.join(resource_hits[:5])}"],
            dominant_theme_reason="the article is organized more like a resource page than a narrative article",
        )
    if opinion_hits:
        return build_general_result(
            "opinion_or_perspective",
            "matched perspective or reflective language",
            0.0,
            [f"perspective signals: {', '.join(opinion_hits[:5])}"],
            dominant_theme_reason="personal perspective or reflective framing dominates the article",
        )
    if overview_hits:
        return build_general_result(
            "general_overview",
            "matched overview or basics-oriented language",
            0.0,
            [f"overview signals: {', '.join(overview_hits[:5])}"],
            dominant_theme_reason="the article reads like a general overview of the topic",
        )
    if content_tier == "weak_snippet" or content_length < 180:
        return build_general_result(
            "informational_article",
            "content is short and low-structure",
            ARTICLE_TYPE_PENALTY,
            ["short or low-structure content"],
            dominant_theme_reason="the article does not show enough structure to infer a stronger general-purpose type",
        )
    if any(term in source_lower for term in ("health", "medical", "clinic", "hospital", "cdc", "nih")):
        return build_general_result(
            "informational_article",
            "matched a domain source with general informational content",
            0.0,
            ["credible informational source"],
            dominant_theme_reason="the article is primarily informational without a narrower editorial subtype",
        )
    return build_general_result(
        "informational_article",
        "matched general informational content",
        0.0,
        ["general informational framing"],
        dominant_theme_reason="the article is mainly informative without a stronger specialized framing",
    )


def _extract_secondary_article_tags(
    title_lower: str,
    intro_lower: str,
    headings_lower: str,
    headings_normalized: list[str],
    body_lower: str,
    text_blob: str,
) -> tuple[list[str], dict[str, dict[str, object]], list[str], list[str], list[str], dict[str, str]]:
    weighted_tags: dict[str, dict[str, object]] = {}
    tag_signals: dict[str, str] = {}
    body_text = body_lower or text_blob.replace(title_lower, "", 1).replace(intro_lower, "", 1).strip()

    def add_weighted_tag(
        tag: str,
        *,
        title_terms: tuple[str, ...] = (),
        intro_terms: tuple[str, ...] = (),
        heading_terms: tuple[str, ...] = (),
        body_terms: tuple[str, ...] = (),
        weak_body_terms: tuple[str, ...] = (),
        title_dominant: bool = False,
        weak_only_on_title: bool = False,
    ) -> None:
        title_hits = _collect_matches(title_lower, title_terms)
        intro_hits = _collect_matches(intro_lower, intro_terms or body_terms or title_terms)
        heading_hits = _collect_heading_matches(
            headings_normalized,
            heading_terms or intro_terms or body_terms or title_terms,
        )
        strong_body_hits = _collect_repeated_body_matches(body_text, body_terms, min_hits=2)
        weak_body_hits = _collect_matches(body_text, weak_body_terms or body_terms)
        body_match_count = _count_term_hits(body_text, weak_body_terms or body_terms)
        signals = _unique_reasons(title_hits + intro_hits + heading_hits + strong_body_hits + weak_body_hits)
        if not signals:
            return

        editorial_weight = 0.0
        heading_weight_component, heading_boost_capped = _heading_weight_for_tag(
            tag,
            heading_hits,
            title_hits=title_hits,
            intro_hits=intro_hits,
        )
        body_weight_component = _body_weight_from_count(body_match_count)
        body_saturation_applied = body_match_count >= 6
        if title_hits:
            editorial_weight += 2.0
        if intro_hits:
            editorial_weight += 1.5
        if heading_weight_component:
            editorial_weight += heading_weight_component
        editorial_weight += body_weight_component
        editorial_weight = min(editorial_weight, 5.5)

        strength = 0.5
        if title_hits:
            strength = 2.0
        elif (
            heading_hits
            and editorial_weight >= 2.2
            and (
                intro_hits
                or len(heading_hits) >= 2
                or body_match_count >= 4
            )
        ):
            strength = 2.0
        elif intro_hits and body_match_count >= 2 and editorial_weight >= 3.0:
            strength = 2.0
        elif editorial_weight >= 4.25 and (title_hits or intro_hits or len(heading_hits) >= 2):
            strength = 2.0
        elif intro_hits or heading_hits or body_match_count >= 2 or editorial_weight >= 1.5:
            strength = 1.0
        elif weak_body_hits:
            strength = 0.5

        if strength >= 2.0 and title_hits and body_match_count >= 2:
            reason = "matched title intent and repeated body signals"
            centrality_reason = "tag appears in the title and is reinforced through the article body"
        elif strength >= 2.0 and heading_hits and body_match_count >= 2:
            reason = "matched section framing with repeated body support"
            centrality_reason = "tag is emphasized in section headings and reinforced through the article body"
        elif strength >= 2.0 and (intro_hits or heading_hits):
            reason = "matched intro or section framing with repeated body support"
            centrality_reason = "tag is emphasized in the intro or section framing and reinforced through the body"
        elif strength >= 2.0:
            reason = "matched strong structural signals"
            centrality_reason = "tag is structurally central to the article, not just a repeated mention"
        elif strength >= 1.0 and (intro_hits or heading_hits):
            reason = "matched as a supporting structural theme"
            centrality_reason = "tag appears in the intro or sections, but does not dominate the article purpose"
        elif strength >= 1.0:
            reason = "matched as a supporting theme"
            centrality_reason = "tag is relevant across the article but reads as supporting rather than central"
        else:
            reason = "single incidental mention"
            centrality_reason = "tag appears only incidentally in the body without strong framing support"

        if weak_only_on_title and not title_hits and strength >= 2.0:
            strength = 1.0
            reason = "matched as a supporting theme"
            centrality_reason = "tag is repeated in the body but not framed as a dominant theme in the title or intro"

        weighted_tags[tag] = {
            "strength": strength,
            "reason": reason,
            "signals": signals[:4],
            "title_matches": title_hits[:4],
            "intro_matches": intro_hits[:4],
            "heading_matches": heading_hits[:4],
            "body_match_count": body_match_count,
            "editorial_weight": round(editorial_weight, 2),
            "body_weight_component": round(body_weight_component, 2),
            "body_saturation_applied": body_saturation_applied,
            "heading_weight_component": round(heading_weight_component, 2),
            "heading_boost_capped": heading_boost_capped,
            "dominant_signal_sources": _build_signal_sources(
                title_hits=title_hits,
                intro_hits=intro_hits,
                heading_hits=heading_hits,
                body_match_count=body_match_count,
            ),
            "centrality_reason": centrality_reason,
        }
        tag_signals[tag] = f"{reason} ({', '.join(signals[:4])})"

    add_weighted_tag(
        "ai_agents",
        title_terms=("ai agent", "ai agents"),
        body_terms=("ai agent", "ai agents", "agentic ai", "agentic system"),
        weak_body_terms=("ai agent", "ai agents"),
        title_dominant=True,
    )
    add_weighted_tag(
        "multi_agent",
        title_terms=("multi-agent", "multi agent", "agents team", "system of agents"),
        body_terms=("multi-agent", "multi agent", "agents team", "system of agents"),
        weak_body_terms=("multi-agent", "multi agent"),
        title_dominant=True,
    )
    add_weighted_tag(
        "security",
        title_terms=("security", "authorization", "authentication", "access control", "credentials", "token exchange"),
        body_terms=(
            "security",
            "authorization",
            "authentication",
            "access control",
            "credentials",
            "credential",
            "permissions",
            "least privilege",
            "service account",
            "service accounts",
        ),
        weak_body_terms=("security", "credentials", "credential", "least privilege", "service account", "service accounts"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "auth",
        title_terms=("authorize", "authorization", "authentication", "access control", "identity"),
        body_terms=(
            "authorize",
            "authorization",
            "authentication",
            "access control",
            "identity",
            "permissions",
            "permission",
            "least privilege",
            "service account",
            "service accounts",
        ),
        weak_body_terms=("authorize", "authorization", "identity", "permissions", "permission", "least privilege", "service account", "service accounts"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "oauth",
        title_terms=("oauth", "token exchange"),
        body_terms=("oauth", "token exchange", "openid", "oidc"),
        weak_body_terms=("oauth", "token exchange"),
        title_dominant=True,
    )
    add_weighted_tag(
        "cloud",
        title_terms=("cloud", "cloud run", "cloud build"),
        body_terms=("cloud", "cloud run", "cloud build"),
        weak_body_terms=("cloud", "cloud run"),
        title_dominant=True,
    )
    add_weighted_tag(
        "devops",
        title_terms=("deploy", "deployment", "infrastructure", "ci/cd"),
        body_terms=("deploy", "deployment", "infra", "infrastructure", "ci/cd", "pipeline"),
        weak_body_terms=("deploy", "deployment", "pipeline"),
        title_dominant=True,
    )
    add_weighted_tag(
        "terraform",
        title_terms=("terraform",),
        body_terms=("terraform",),
        weak_body_terms=("terraform",),
        title_dominant=True,
    )
    add_weighted_tag(
        "google_cloud",
        title_terms=("google cloud", "cloud run", "vertex ai", "gcp"),
        body_terms=("google cloud", "cloud run", "vertex ai", "gcp"),
        weak_body_terms=("google cloud", "cloud run", "gcp"),
        title_dominant=True,
    )
    add_weighted_tag(
        "testing",
        title_terms=("testing", "test", "local testing"),
        body_terms=("testing", "test", "local testing", "test harness"),
        weak_body_terms=("testing", "test", "local testing"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "memory",
        title_terms=("memory", "long-term memory", "memory bank", "vertex ai memory"),
        body_terms=("memory", "long-term memory", "memory bank", "vertex ai memory"),
        weak_body_terms=("memory", "long-term memory"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "mcp",
        title_terms=("mcp", "model context protocol"),
        body_terms=("mcp", "model context protocol"),
        weak_body_terms=("mcp", "model context protocol"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "adk",
        title_terms=("adk", "google adk"),
        body_terms=("adk", "google adk"),
        weak_body_terms=("adk", "google adk"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "workflow",
        title_terms=("workflow", "workflows", "orchestration"),
        body_terms=("workflow", "workflows", "orchestration"),
        weak_body_terms=("workflow", "workflows", "orchestration"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "product_update",
        title_terms=ANNOUNCEMENT_TITLE_TERMS,
        body_terms=ANNOUNCEMENT_TITLE_TERMS,
        weak_body_terms=ANNOUNCEMENT_TITLE_TERMS,
        title_dominant=True,
    )
    add_weighted_tag(
        "event",
        title_terms=EVENT_TITLE_TERMS,
        body_terms=EVENT_TITLE_TERMS,
        weak_body_terms=EVENT_TITLE_TERMS,
        title_dominant=True,
    )
    add_weighted_tag(
        "benchmark",
        title_terms=("benchmark", "benchmarks"),
        body_terms=("benchmark", "benchmarks"),
        weak_body_terms=("benchmark", "benchmarks"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "performance",
        title_terms=("performance", "latency", "throughput"),
        body_terms=("performance", "latency", "throughput"),
        weak_body_terms=("performance", "latency", "throughput"),
        weak_only_on_title=True,
    )
    add_weighted_tag(
        "personal_project",
        title_terms=("weekend project", "side project", "personal project"),
        body_terms=("weekend project", "side project", "personal project", "built this for myself"),
        weak_body_terms=("weekend project", "side project", "personal project", "built this for myself"),
        weak_only_on_title=True,
    )

    _calibrate_weighted_tags(title_lower, body_text, weighted_tags)

    dominant_tags = [tag for tag, payload in weighted_tags.items() if float(payload["strength"]) >= 2.0]
    supporting_tags = [tag for tag, payload in weighted_tags.items() if float(payload["strength"]) == 1.0]
    weak_tags = [tag for tag, payload in weighted_tags.items() if float(payload["strength"]) == 0.5]
    for tag, payload in weighted_tags.items():
        tag_signals[tag] = f"{payload['reason']} ({', '.join(payload['signals'][:4])})"
    selected_tags = dominant_tags + supporting_tags
    return selected_tags, weighted_tags, dominant_tags, supporting_tags, weak_tags, tag_signals


def _calibrate_weighted_tags(
    title_lower: str,
    body_text: str,
    weighted_tags: dict[str, dict[str, object]],
) -> None:
    security_terms = SECURITY_STRONG_TERMS + SECURITY_SUPPORT_TERMS
    security_title_hits = _collect_matches(title_lower, security_terms)
    security_body_repeats = _collect_repeated_body_matches(body_text, security_terms, min_hits=2)
    security_body_mentions = _collect_matches(body_text, security_terms)
    explicit_security_focus = bool({"oauth", "token exchange", "access control"} & set(security_title_hits + security_body_repeats))
    tutorial_title_hits = _collect_matches(title_lower, INSTRUCTIONAL_TERMS + IMPLEMENTATION_TERMS)
    deployment_title_hits = _collect_matches(
        title_lower,
        ("deploy", "deployment", "terraform", "cloud run", "testing", "local testing"),
    )
    practical_security_hits = _collect_matches(
        body_text,
        ("permissions", "permission", "credentials", "credential", "least privilege", "service account", "service accounts"),
    )

    for tag in ("security", "auth"):
        payload = weighted_tags.get(tag)
        if not payload:
            continue
        strength = float(payload.get("strength", 0.0))
        intro_or_heading_matches = list(payload.get("intro_matches", [])) + list(payload.get("heading_matches", []))
        if explicit_security_focus or security_title_hits:
            continue
        if tutorial_title_hits or deployment_title_hits:
            if len(practical_security_hits) >= 2 or len(intro_or_heading_matches) >= 3 or strength >= 2.0:
                payload["strength"] = 1.0
                payload["reason"] = "supporting concern within a tutorial/deployment article"
                payload["centrality_reason"] = (
                    "security appears as a supporting deployment concern, not the article's main focus"
                )
            elif strength >= 1.0:
                payload["strength"] = 0.5
                payload["reason"] = "single incidental mention"
                payload["centrality_reason"] = (
                    "security/auth is mentioned, but not with enough structural emphasis to count as a supporting theme"
                )

    for tag in ("testing", "memory", "product_update"):
        payload = weighted_tags.get(tag)
        if not payload or float(payload.get("strength", 0.0)) < 2.0:
            continue
        if payload.get("title_matches") or payload.get("intro_matches"):
            continue
        payload["strength"] = 1.0
        payload["reason"] = "supporting theme without title or intro framing"
        payload["centrality_reason"] = (
            "the tag is repeated in the body, but it is not framed as a dominant theme in the title or intro"
        )

    for tag in ("cloud", "devops", "google_cloud"):
        payload = weighted_tags.get(tag)
        if not payload or float(payload.get("strength", 0.0)) < 2.0:
            continue
        if payload.get("title_matches"):
            continue
        payload["strength"] = 1.0
        payload["reason"] = "supporting infrastructure theme without title framing"
        payload["centrality_reason"] = (
            "infrastructure appears in the article structure, but it is not framed in the title as the main editorial center"
        )

    workflow_payload = weighted_tags.get("workflow")
    if workflow_payload and not workflow_payload.get("title_matches") and not workflow_payload.get("heading_matches"):
        workflow_payload["strength"] = 0.5
        workflow_payload["reason"] = "single incidental mention"
        workflow_payload["centrality_reason"] = (
            "workflow language is present, but it is too generic to count as a dominant or supporting theme here"
        )


def _build_heading_diagnostics(metadata: dict, content: str) -> dict[str, object]:
    explicit_headings = metadata.get("headings") if isinstance(metadata.get("headings"), list) else []
    detected_headings: list[str] = []
    heading_source = "none"
    raw_html_heading_count = int(metadata.get("raw_html_heading_count") or 0)
    heading_extraction_strategy = str(metadata.get("heading_extraction_strategy") or "none")
    sample_detected_headings = metadata.get("sample_detected_headings")

    if explicit_headings:
        detected_headings = [str(value).strip() for value in explicit_headings if str(value).strip()]
        heading_source = "explicit" if detected_headings else "none"
    elif content:
        normalized_lines = [line.strip() for line in str(content).splitlines() if line.strip()]
        inferred_headings: list[str] = []
        for line in normalized_lines:
            markdown_match = re.match(r"^#{1,6}\s+(.*)$", line)
            if markdown_match:
                inferred_headings.append(markdown_match.group(1).strip())
                continue
            if line.endswith(":") and len(line) <= 80:
                inferred_headings.append(line.rstrip(":").strip())
                continue
            if len(line) <= 80 and line == line.title() and len(line.split()) <= 8:
                inferred_headings.append(line)
        detected_headings = [value for value in inferred_headings if value][:12]
        heading_source = "inferred" if detected_headings else "none"

    normalized_headings = [_normalize_term(value).replace("-", " ") for value in detected_headings if value]
    if not isinstance(sample_detected_headings, list):
        sample_detected_headings = detected_headings[:5]
    return {
        "detected_headings": detected_headings[:12],
        "normalized_headings": normalized_headings[:12],
        "heading_count": len(detected_headings),
        "raw_html_heading_count": raw_html_heading_count,
        "extracted_heading_count": int(metadata.get("extracted_heading_count") or len(detected_headings)),
        "heading_extraction_strategy": heading_extraction_strategy,
        "sample_detected_headings": [str(value).strip() for value in sample_detected_headings if str(value).strip()][:5],
        "heading_source": heading_source,
        "matched_heading_tags": {},
    }


def _build_matched_heading_tags(
    heading_diagnostics: dict[str, object],
    weighted_secondary_tags: dict[str, dict[str, object]],
) -> dict[str, dict[str, list[str]]]:
    detected_headings = heading_diagnostics.get("detected_headings", [])
    normalized_headings = heading_diagnostics.get("normalized_headings", [])
    if not isinstance(detected_headings, list) or not isinstance(normalized_headings, list):
        return {}

    normalized_to_original: dict[str, list[str]] = {}
    for original, normalized in zip(detected_headings, normalized_headings):
        if not normalized:
            continue
        normalized_to_original.setdefault(str(normalized), []).append(str(original))

    matched_heading_tags: dict[str, dict[str, list[str]]] = {}
    for tag, payload in weighted_secondary_tags.items():
        if not isinstance(payload, dict):
            continue
        normalized_matches = [
            str(value) for value in payload.get("heading_matches", [])
            if str(value)
        ]
        if not normalized_matches:
            continue
        original_matches: list[str] = []
        for normalized_match in normalized_matches:
            for original in normalized_to_original.get(normalized_match, []):
                if original not in original_matches:
                    original_matches.append(original)
        matched_heading_tags[tag] = {
            "matches": original_matches,
            "normalized_matches": _unique_reasons(normalized_matches),
        }
    return matched_heading_tags


def _body_weight_from_count(body_match_count: int) -> float:
    if body_match_count <= 0:
        return 0.0
    if body_match_count == 1:
        return 0.25
    if body_match_count <= 3:
        return 0.6
    if body_match_count <= 5:
        return 0.85
    if body_match_count <= 10:
        return 1.1
    return 1.3


def _heading_weight_for_tag(
    tag: str,
    heading_hits: list[str],
    *,
    title_hits: list[str],
    intro_hits: list[str],
) -> tuple[float, bool]:
    if not heading_hits:
        return 0.0, False

    base_weight = 1.1
    if len(heading_hits) >= 2:
        base_weight += 0.25

    capped_weight = base_weight
    if tag in GENERIC_INFRASTRUCTURE_TAGS and not title_hits:
        capped_weight = 0.75 if not intro_hits else 0.9
    else:
        capped_weight = min(capped_weight, 1.35)

    return round(capped_weight, 2), capped_weight < base_weight


def _build_signal_sources(
    *,
    title_hits: list[str],
    intro_hits: list[str],
    heading_hits: list[str],
    body_match_count: int,
) -> list[str]:
    sources: list[str] = []
    if title_hits:
        sources.append("title")
    if intro_hits:
        sources.append("intro")
    if heading_hits:
        sources.append("heading")
    if body_match_count >= 2:
        sources.append("repeated_body")
    elif body_match_count == 1:
        sources.append("body")
    return sources


def _score_editorial_alignment(
    normalized_keywords: list[str],
    dominant_tags: list[str],
    supporting_tags: list[str],
) -> tuple[float, str | None]:
    if not normalized_keywords:
        return 0.0, None

    keyword_blob = " ".join(normalized_keywords)
    topic_key = next((keyword for keyword in TOPIC_EDITORIAL_ALIGNMENT_TAGS if keyword in keyword_blob), "")
    if not topic_key:
        return 0.0, None

    aligned_tags = set(TOPIC_EDITORIAL_ALIGNMENT_TAGS.get(topic_key, ()))
    if aligned_tags & set(dominant_tags):
        return 0.55, "editorial center aligns with topic"
    if aligned_tags & set(supporting_tags):
        return 0.08, "supporting editorial theme aligns with topic"
    return 0.0, None


def _select_diverse_items(filtered_entries: list[tuple[dict, dict]], *, top_n: int) -> list[dict]:
    if top_n <= 0 or not filtered_entries:
        return []

    remaining = list(filtered_entries)
    selected_items: list[dict] = []
    selected_scores: list[dict] = []

    while remaining and len(selected_items) < top_n:
        best_index = 0
        best_adjusted_score = None
        best_penalty = 0.0
        best_reasons: list[str] = []

        for index, (_item, score_entry) in enumerate(remaining):
            penalty, reasons = _calculate_diversity_penalty(score_entry, selected_scores)
            raw_score = float(score_entry.get("score") or 0.0)
            adjusted_score = round(max(raw_score - penalty, 0.0), 2)

            score_entry["diversity_penalty"] = round(penalty, 2)
            score_entry["similarity_reasons"] = reasons
            score_entry["diversity_adjusted_score"] = adjusted_score

            candidate_key = (adjusted_score, raw_score, -index)
            best_key = None if best_adjusted_score is None else (best_adjusted_score, float(remaining[best_index][1].get("score") or 0.0), -best_index)
            if best_key is None or candidate_key > best_key:
                best_index = index
                best_adjusted_score = adjusted_score
                best_penalty = penalty
                best_reasons = reasons

        selected_item, selected_score = remaining.pop(best_index)
        selected_score["diversity_penalty"] = round(best_penalty, 2)
        selected_score["similarity_reasons"] = best_reasons
        selected_score["diversity_adjusted_score"] = round(best_adjusted_score or float(selected_score.get("score") or 0.0), 2)
        selected_items.append(selected_item)
        selected_scores.append(selected_score)

    return selected_items


def _calculate_diversity_penalty(candidate: dict, selected_scores: list[dict]) -> tuple[float, list[str]]:
    if not selected_scores:
        return 0.0, []

    strongest_penalty = 0.0
    strongest_reasons: list[str] = []
    for selected in selected_scores:
        penalty, reasons = _compare_editorial_similarity(candidate, selected)
        if penalty > strongest_penalty:
            strongest_penalty = penalty
            strongest_reasons = reasons

    if strongest_penalty <= 0:
        return 0.0, []
    max_penalty = 2.1 if any("near-duplicate editorial cluster" in reason for reason in strongest_reasons) else 1.2
    return round(min(strongest_penalty, max_penalty), 2), strongest_reasons


def _compare_editorial_similarity(candidate: dict, selected: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    penalty = 0.0

    candidate_dominant = set(candidate.get("dominant_tags") or [])
    selected_dominant = set(selected.get("dominant_tags") or [])
    dominant_overlap = sorted(candidate_dominant & selected_dominant)
    if len(dominant_overlap) >= 2:
        penalty += 0.45
        reasons.append(f"shared dominant tags with selected article ({', '.join(dominant_overlap[:4])})")
    elif len(dominant_overlap) == 1:
        penalty += 0.18
        reasons.append(f"shared dominant theme with selected article ({dominant_overlap[0]})")

    candidate_supporting = set(candidate.get("supporting_tags") or [])
    selected_supporting = set(selected.get("supporting_tags") or [])
    supporting_overlap = sorted((candidate_dominant | candidate_supporting) & (selected_dominant | selected_supporting))
    if len(supporting_overlap) >= 4:
        penalty += 0.24
        reasons.append("high supporting-tag overlap with an already selected article")
    elif len(supporting_overlap) >= 2:
        penalty += 0.12
        reasons.append("moderate supporting-tag overlap with an already selected article")

    if candidate.get("primary_article_type") == selected.get("primary_article_type"):
        candidate_type = str(candidate.get("primary_article_type") or "")
        if candidate_type in {"tutorial", "deep_technical", "system_design", "architecture_security"}:
            penalty += 0.16
            reasons.append(f"same editorial framing as selected article ({candidate_type.replace('_', ' ')})")

    if _same_source_family(candidate, selected):
        penalty += 0.16
        reasons.append("same source or publication family as selected article")

    title_overlap = _significant_title_overlap(
        str(candidate.get("title") or ""),
        str(selected.get("title") or ""),
    )
    if title_overlap >= 4:
        penalty += 0.32
        reasons.append("very similar title phrasing to an already selected article")
    elif title_overlap >= 2:
        penalty += 0.14
        reasons.append("similar title phrasing to an already selected article")

    if _shared_named_system(candidate, selected):
        penalty += 0.18
        reasons.append("repeats the same named system or platform focus")

    if _is_near_duplicate_editorial_cluster(
        candidate,
        selected,
        dominant_overlap=dominant_overlap,
        supporting_overlap=supporting_overlap,
        title_overlap=title_overlap,
    ):
        penalty += 0.6
        reasons.append("near-duplicate editorial cluster with an already selected article")

    return round(penalty, 2), _unique_reasons(reasons)


def _same_source_family(candidate: dict, selected: dict) -> bool:
    candidate_source = _normalize_term(str(candidate.get("source_name") or ""))
    selected_source = _normalize_term(str(selected.get("source_name") or ""))
    if candidate_source and candidate_source == selected_source:
        return True

    candidate_domain = _extract_domain(str(candidate.get("url") or ""))
    selected_domain = _extract_domain(str(selected.get("url") or ""))
    return bool(candidate_domain and candidate_domain == selected_domain)


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.netloc or "").lower()


def _significant_title_overlap(first_title: str, second_title: str) -> int:
    first_tokens = {
        token
        for token in _tokenize_text(first_title)
        if len(token) > 2 and token not in TITLE_STOPWORDS
    }
    second_tokens = {
        token
        for token in _tokenize_text(second_title)
        if len(token) > 2 and token not in TITLE_STOPWORDS
    }
    return len(first_tokens & second_tokens)


def _shared_named_system(candidate: dict, selected: dict) -> bool:
    candidate_title = _similarity_marker_text(candidate)
    selected_title = _similarity_marker_text(selected)
    system_markers = (
        "google adk",
        "adk",
        "cloud run",
        "mcp",
        "dev signal",
        "terraform",
        "gemma",
        "vertex ai",
    )
    shared = [marker for marker in system_markers if marker in candidate_title and marker in selected_title]
    return len(shared) >= 2


def _similarity_marker_text(score_entry: dict) -> str:
    title = str(score_entry.get("title") or "")
    heading_diagnostics = score_entry.get("heading_diagnostics") or {}
    headings = heading_diagnostics.get("detected_headings") or []
    combined = " ".join([title, *[str(heading) for heading in headings]])
    return _normalize_term(combined)


def _is_near_duplicate_editorial_cluster(
    candidate: dict,
    selected: dict,
    *,
    dominant_overlap: list[str],
    supporting_overlap: list[str],
    title_overlap: int,
) -> bool:
    if candidate.get("primary_article_type") != "tutorial" or selected.get("primary_article_type") != "tutorial":
        return False
    if not _same_source_family(candidate, selected):
        return False
    if len(dominant_overlap) < 3:
        return False
    if len(supporting_overlap) < 5:
        return False
    if title_overlap < 4:
        return False
    return True


def _count_matching_tags(normalized_keywords: list[str], secondary_tags: list[str]) -> int:
    keyword_blob = " ".join(normalized_keywords)
    matches = 0
    for tag in secondary_tags:
        if any(part in keyword_blob for part in tag.split("_")):
            matches += 1
    return matches


def _collect_matches(text: str, terms: Iterable[str]) -> list[str]:
    return [term for term in terms if term in text]


def _collect_heading_matches(headings: list[str], terms: Iterable[str]) -> list[str]:
    matches: list[str] = []
    for heading in headings:
        normalized_heading = _normalize_term(heading).replace("-", " ")
        for term in terms:
            normalized_term = _normalize_term(term).replace("-", " ")
            if normalized_term and normalized_term in normalized_heading and normalized_heading not in matches:
                matches.append(normalized_heading)
    return matches


def _collect_repeated_body_matches(text: str, terms: Iterable[str], *, min_hits: int) -> list[str]:
    matches: list[str] = []
    for term in terms:
        if text.count(term) >= min_hits:
            matches.append(term)
    return matches


def _count_term_hits(text: str, terms: Iterable[str]) -> int:
    return sum(text.count(term) for term in terms)


def _is_explicit_specificity_phrase(term: str) -> bool:
    explicit_terms = {
        "ai agent",
        "ai agents",
        "agentic workflow",
        "agent orchestration",
        "multi-agent",
        "multi agent",
        "autonomous agent",
        "autonomous agents",
        "model context protocol",
        "agent framework",
        "agent deployment",
        "agent authorization",
        "agent permissions",
        "agent security",
        "planner/executor",
        "planner executor",
        "task delegation",
        "human-in-the-loop agents",
        "human in the loop agents",
        "long-term memory",
        "long term memory",
        "tool use",
        "token exchange",
    }
    return term in explicit_terms


def _looks_promotional(title_lower: str, snippet_lower: str) -> bool:
    promo_terms = (
        "launch",
        "released",
        "release",
        "introducing",
        "update",
        "version",
        "v7",
        "announcement",
    )
    text_blob = f"{title_lower} {snippet_lower}"
    return any(term in text_blob for term in promo_terms)


def _looks_opinionated(title_lower: str, snippet_lower: str) -> bool:
    opinion_terms = (
        "i read",
        "i tried",
        "my take",
        "my thoughts",
        "lessons",
        "opinion",
        "what i learned",
    )
    text_blob = f"{title_lower} {snippet_lower}"
    return any(term in text_blob for term in opinion_terms)


def _detect_topic_domain(normalized_keywords: list[str]) -> str:
    if not normalized_keywords:
        return "technical"
    keyword_blob = " ".join(normalized_keywords)
    if any(term in keyword_blob for term in TECHNICAL_TOPIC_HINTS):
        return "technical"
    return "general"


def _general_specificity_support_terms(normalized_keywords: list[str]) -> tuple[str, ...]:
    tokens: list[str] = []
    for keyword in normalized_keywords:
        for token in _tokenize_text(keyword):
            if len(token) <= 2 and not token.isdigit():
                continue
            if token in TITLE_STOPWORDS:
                continue
            if token not in tokens:
                tokens.append(token)
    return tuple(tokens)


def _unique_reasons(reasons: list[str]) -> list[str]:
    unique: list[str] = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(reason)
    return unique


def _normalize_term(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _tokenize_text(value: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalize_term(value).replace("-", " "))
    normalized_tokens: list[str] = []
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            token = token[:-1]
        normalized_tokens.append(token)
    return normalized_tokens
