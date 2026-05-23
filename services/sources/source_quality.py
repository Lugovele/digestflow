"""Topic-agnostic heuristics for judging search-result source quality."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from urllib.parse import urlparse
from django.conf import settings


_COMMERCIAL_PATTERNS = (
    (re.compile(r"\b(book a demo|request a demo|schedule (a )?(call|consultation)|contact sales)\b"), "sales_demo_contact_page"),
    (re.compile(r"\b(request a quote|get started( today)?|talk to sales|contact us)\b"), "lead_generation_article"),
    (re.compile(r"\b(consulting services|consulting offer|our services|sleep consulting|consultant program)\b"), "agency_or_consulting_offer"),
    (re.compile(r"\b(our platform|platform for|solutions for your business|why choose us)\b"), "product_landing_page"),
)

_GENERIC_LISTICLE_PATTERNS = (
    (re.compile(r"\btop\s+\d+\s+(benefits|reasons)\b"), "generic_benefits_listicle"),
    (re.compile(r"\bbenefits of\b"), "generic_benefits_listicle"),
    (re.compile(r"\bwhy (your|every) [a-z0-9\s]+ needs\b"), "generic_benefits_listicle"),
    (re.compile(r"\bultimate guide\b"), "shallow_ultimate_guide"),
    (re.compile(r"\b(unlock the power of|boost your|transform your|streamline your)\b"), "vague_promotional_article"),
)

_URL_REJECTION_PATTERNS = (
    (re.compile(r"/(pricing|demo|contact|quote|consult|services|solutions|platform)(/|$)"), "product_landing_page"),
)

_HIGH_QUALITY_TYPES = {
    "research_report",
    "scientific_study",
    "survey_or_data_report",
    "news_article",
    "expert_interview",
    "expert_opinion",
    "debate_or_perspective",
    "concrete_case_study",
    "institutional_guideline",
    "policy_or_regulatory_source",
    "methodological_article",
    "substantive_longform_article",
    "practical_guide_with_concrete_steps",
}

_LOW_QUALITY_TYPES = {
    "generic_benefits_listicle",
    "vague_how_to_seo_article",
    "thin_affiliate_article",
    "service_landing_page",
    "product_landing_page",
    "sales_demo_contact_page",
    "agency_or_consulting_offer",
    "lead_generation_article",
    "generic_company_blog_without_substance",
    "shallow_ultimate_guide",
    "vague_promotional_article",
}

_RESEARCH_TERMS = {
    "study",
    "research",
    "report",
    "survey",
    "data",
    "findings",
    "evidence",
    "methodology",
    "limitations",
    "regulation",
    "regulatory",
    "policy",
    "guideline",
    "guidance",
}

_NEWS_TERMS = {
    "news",
    "reported",
    "reporting",
    "latest",
    "new findings",
    "announced",
    "says",
    "highlights",
    "investigation",
    "according to",
    "coverage",
    "current developments",
    "update",
    "breaking",
    "recent",
}

_EXPERT_TERMS = {
    "expert",
    "interview",
    "opinion",
    "analysis",
    "perspective",
    "debate",
    "controversy",
    "tradeoff",
    "tradeoffs",
    "risk",
    "risks",
}

_CASE_STUDY_TERMS = {
    "case study",
    "case studies",
    "implementation",
    "lessons learned",
    "examples",
    "comparison",
    "comparisons",
    "real-world",
    "real world",
    "implementation details",
}

_PRACTICAL_GUIDE_TERMS = {
    "step-by-step",
    "step by step",
    "concrete steps",
    "checklist",
    "how to",
    "playbook",
    "framework",
    "practical guide",
}

_PROMOTIONAL_TERMS = {
    "boost",
    "transform",
    "streamline",
    "get started",
    "our platform",
    "our services",
    "why choose us",
    "sales",
    "demo",
    "consultation",
}


@dataclass(frozen=True)
class SourceQualityAssessment:
    source_content_type: str
    quality_score: int
    commercial_intent_score: int
    substance_score: int
    freshness_status: str
    detected_publication_date: str
    detected_publication_year: int | None
    freshness_score: int
    freshness_rejection_reason: str
    freshness_signals: tuple[str, ...]
    quality_tags: tuple[str, ...]
    rejection_reason: str
    accepted: bool
    accepted_reason: str


def assess_source_quality(*, title: str, url: str, snippet: str, provider_published_at: str = "") -> SourceQualityAssessment:
    title_text = _normalize_text(title)
    snippet_text = _normalize_text(snippet)
    combined_text = " ".join(part for part in (title_text, snippet_text) if part)
    normalized_url = str(url or "").strip().lower()

    quality_tags: list[str] = []
    rejection_reason = ""
    accepted_reason = ""

    url_rejection = _match_pattern_group(normalized_url, _URL_REJECTION_PATTERNS)
    commercial_match = _match_pattern_group(combined_text, _COMMERCIAL_PATTERNS)
    generic_match = _match_pattern_group(combined_text, _GENERIC_LISTICLE_PATTERNS)

    research_hits = _count_terms(combined_text, _RESEARCH_TERMS)
    news_hits = _count_terms(combined_text, _NEWS_TERMS)
    expert_hits = _count_terms(combined_text, _EXPERT_TERMS)
    case_hits = _count_terms(combined_text, _CASE_STUDY_TERMS)
    practical_hits = _count_terms(combined_text, _PRACTICAL_GUIDE_TERMS)
    promotional_hits = _count_terms(combined_text, _PROMOTIONAL_TERMS)

    if research_hits:
        quality_tags.append("research_signals")
    if news_hits:
        quality_tags.append("news_signals")
    if expert_hits:
        quality_tags.append("expert_signals")
    if case_hits:
        quality_tags.append("case_study_signals")
    if practical_hits:
        quality_tags.append("practical_guide_signals")

    freshness = assess_source_freshness(
        title=title,
        url=normalized_url or url,
        snippet=snippet,
        provider_published_at=provider_published_at,
    )
    quality_tags.extend(freshness.freshness_signals)

    substance_score = (research_hits * 3) + (expert_hits * 2) + (case_hits * 3) + (practical_hits * 2)
    commercial_intent_score = (2 if url_rejection else 0) + (3 if commercial_match else 0) + promotional_hits
    quality_score = substance_score - (commercial_intent_score * 2) - (2 if generic_match else 0) + freshness.freshness_score

    hostname = str(urlparse(normalized_url).netloc or "").strip().lower()
    blog_like_url = "/blog/" in normalized_url or hostname.startswith("blog.")
    if blog_like_url and substance_score == 0 and commercial_intent_score > 0:
        generic_match = generic_match or "generic_company_blog_without_substance"

    source_content_type = _pick_source_content_type(
        title_text=title_text,
        snippet_text=snippet_text,
        research_hits=research_hits,
        news_hits=news_hits,
        expert_hits=expert_hits,
        case_hits=case_hits,
        practical_hits=practical_hits,
        commercial_match=commercial_match,
        generic_match=generic_match,
        url_rejection=url_rejection,
        substance_score=substance_score,
    )

    if url_rejection:
        rejection_reason = "rejected because: product/demo/pricing intent"
    elif commercial_match:
        rejection_reason = "rejected because: commercial service-page signals"
    elif generic_match:
        if generic_match == "generic_company_blog_without_substance":
            rejection_reason = "rejected because: generic company blog without substantive signals"
        else:
            rejection_reason = "rejected because: generic benefits/listicle SEO pattern"
    elif substance_score == 0 and commercial_intent_score >= 2:
        rejection_reason = "rejected because: vague promotional language without substantive signals"
    elif freshness.rejection_reason:
        rejection_reason = freshness.rejection_reason

    accepted = False
    if rejection_reason:
        accepted = False
    elif source_content_type in _HIGH_QUALITY_TYPES and quality_score >= 2:
        accepted = True
    elif source_content_type == "practical_guide_with_concrete_steps" and quality_score >= 2:
        accepted = True
    elif source_content_type == "substantive_longform_article" and substance_score >= 3 and commercial_intent_score <= 1:
        accepted = True

    if accepted:
        if source_content_type in {"research_report", "scientific_study", "survey_or_data_report"}:
            accepted_reason = "accepted because: research/report signals"
        elif source_content_type == "news_article":
            accepted_reason = (
                "accepted because: fresh news/reporting signals"
                if freshness.status in {"fresh", "acceptable"}
                else "accepted with unknown date because: news/reporting signals"
            )
        elif source_content_type in {"expert_interview", "expert_opinion", "debate_or_perspective"}:
            accepted_reason = (
                "accepted because: expert opinion/debate signals"
                if freshness.status != "unknown"
                else "accepted with unknown date because: expert opinion/debate signals"
            )
        elif source_content_type == "concrete_case_study":
            accepted_reason = (
                "accepted because: concrete case-study signals"
                if freshness.status != "unknown"
                else "accepted with unknown date because: concrete case-study signals"
            )
        elif source_content_type == "practical_guide_with_concrete_steps":
            accepted_reason = (
                "accepted because: practical guide includes concrete implementation signals"
                if freshness.status != "unknown"
                else "accepted with unknown date because: practical guide includes concrete implementation signals"
            )
        else:
            accepted_reason = (
                "accepted because: substantive source signals"
                if freshness.status != "unknown"
                else "accepted with unknown date because: strong substantive signals"
            )
    elif not rejection_reason and quality_score < 2:
        rejection_reason = "rejected because: not enough substantive signals"

    quality_tags.append(f"source_content_type:{source_content_type}")

    return SourceQualityAssessment(
        source_content_type=source_content_type,
        quality_score=quality_score,
        commercial_intent_score=commercial_intent_score,
        substance_score=substance_score,
        freshness_status=freshness.status,
        detected_publication_date=freshness.detected_publication_date,
        detected_publication_year=freshness.detected_publication_year,
        freshness_score=freshness.freshness_score,
        freshness_rejection_reason=freshness.rejection_reason,
        freshness_signals=tuple(freshness.freshness_signals),
        quality_tags=tuple(dict.fromkeys(quality_tags)),
        rejection_reason=rejection_reason,
        accepted=accepted,
        accepted_reason=accepted_reason,
    )


@dataclass(frozen=True)
class SourceFreshnessAssessment:
    status: str
    detected_publication_date: str
    detected_publication_year: int | None
    freshness_score: int
    rejection_reason: str
    freshness_signals: tuple[str, ...]


def assess_source_freshness(*, title: str, url: str, snippet: str, provider_published_at: str = "") -> SourceFreshnessAssessment:
    months = _get_search_recency_months()
    current_year = datetime.now(timezone.utc).year
    candidate_text = " ".join(str(value or "") for value in (provider_published_at, title, snippet, url))
    published_date = _extract_publication_date(candidate_text)
    detected_date = published_date.isoformat() if published_date else ""
    detected_year = published_date.year if published_date else _extract_publication_year(candidate_text)
    freshness_signals: list[str] = []

    if provider_published_at:
        freshness_signals.append("freshness:provider_date")
    if published_date:
        freshness_signals.append(f"freshness:date:{published_date.isoformat()}")
    elif detected_year:
        freshness_signals.append(f"freshness:year:{detected_year}")
    else:
        freshness_signals.append("freshness:unknown")

    if published_date:
        month_delta = (current_year - published_date.year) * 12 + (datetime.now(timezone.utc).month - published_date.month)
        if month_delta <= max(0, months - 1):
            return SourceFreshnessAssessment("fresh", detected_date, detected_year, 4, "", tuple(freshness_signals))
        if published_date.year == current_year:
            return SourceFreshnessAssessment("acceptable", detected_date, detected_year, 2, "", tuple(freshness_signals))
        if current_year - published_date.year >= 3:
            return SourceFreshnessAssessment(
                "very_stale",
                detected_date,
                detected_year,
                -6,
                "rejected because: very stale source",
                tuple(freshness_signals),
            )
        return SourceFreshnessAssessment(
            "stale",
            detected_date,
            detected_year,
            -3,
            "rejected because: stale source outside recency window",
            tuple(freshness_signals),
        )

    if detected_year is not None:
        if detected_year == current_year:
            return SourceFreshnessAssessment("acceptable", "", detected_year, 1, "", tuple(freshness_signals))
        if current_year - detected_year >= 3:
            return SourceFreshnessAssessment(
                "very_stale",
                "",
                detected_year,
                -6,
                "rejected because: very stale source",
                tuple(freshness_signals),
            )
        return SourceFreshnessAssessment(
            "stale",
            "",
            detected_year,
            -3,
            "rejected because: stale source outside recency window",
            tuple(freshness_signals),
        )

    return SourceFreshnessAssessment("unknown", "", None, 0, "", tuple(freshness_signals))


def _pick_source_content_type(
    *,
    title_text: str,
    snippet_text: str,
    research_hits: int,
    news_hits: int,
    expert_hits: int,
    case_hits: int,
    practical_hits: int,
    commercial_match: str,
    generic_match: str,
    url_rejection: str,
    substance_score: int,
) -> str:
    if url_rejection or commercial_match:
        return commercial_match or url_rejection or "service_landing_page"
    if generic_match:
        return generic_match
    if "case study" in title_text or "case study" in snippet_text or "lessons learned" in title_text or "lessons learned" in snippet_text:
        return "concrete_case_study"
    if "study" in title_text or "study" in snippet_text:
        return "scientific_study"
    if "guideline" in title_text or "guideline" in snippet_text or "policy" in title_text or "regulation" in snippet_text:
        return "institutional_guideline"
    if research_hits >= 3 and ("survey" in title_text or "report" in title_text or "data" in snippet_text):
        return "survey_or_data_report"
    if news_hits >= 2 and "survey" not in title_text and "survey" not in snippet_text and "study" not in title_text and "study" not in snippet_text:
        return "news_article"
    if news_hits >= 1 and "news" in title_text:
        return "news_article"
    if news_hits >= 1 and "report" in title_text and "survey" not in title_text and "study" not in title_text:
        return "news_article"
    if research_hits >= 2:
        return "research_report"
    if case_hits >= 1:
        return "concrete_case_study"
    if expert_hits >= 2:
        return "debate_or_perspective"
    if practical_hits >= 1 and substance_score >= 3:
        return "practical_guide_with_concrete_steps"
    if "practical guide" in title_text or "practical guide" in snippet_text:
        return "practical_guide_with_concrete_steps"
    if substance_score >= 3:
        return "substantive_longform_article"
    if practical_hits >= 1:
        return "vague_how_to_seo_article"
    return "generic_company_blog_without_substance"


def _match_pattern_group(text: str, patterns: tuple[tuple[re.Pattern[str], str], ...]) -> str:
    for pattern, label in patterns:
        if pattern.search(text):
            return label
    return ""


def _count_terms(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _get_search_recency_months() -> int:
    raw_value = getattr(settings, "SEARCH_RECENCY_MONTHS", 1)
    try:
        months = int(raw_value)
    except (TypeError, ValueError):
        months = 1
    return max(1, months)


def _extract_publication_date(text: str):
    normalized = str(text or "")
    iso_match = re.search(r"\b(20\d{2})[-/](\d{2})[-/](\d{2})\b", normalized)
    if not iso_match:
        return None
    year, month, day = (int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
    try:
        return datetime(year, month, day, tzinfo=timezone.utc).date()
    except ValueError:
        return None


def _extract_publication_year(text: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", str(text or ""))
    if not match:
        return None
    year = int(match.group(1))
    if year < 2000 or year > datetime.now(timezone.utc).year + 1:
        return None
    return year
