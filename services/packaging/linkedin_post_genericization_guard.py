"""Provider-free selection guard for visibly generic final-post prose.

The guard is intentionally narrow: it does not score quality, repair text,
inspect benchmark labels, or call providers. It only detects when an otherwise
accepted post has enough runtime-visible generic/template signals to block
selection under the documented Human Voice automatic-fail semantics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


SIGNAL_FORMULAIC_POLISHED_FRAMING = "formulaic_polished_framing"
SIGNAL_ABSTRACT_CORPORATE_DENSITY = "abstract_corporate_density"
SIGNAL_WEAK_DISTINCTIVE_SENTENCE_SHAPE = "weak_distinctive_sentence_shape"
SIGNAL_GENERIC_CLOSING_OR_CTA = "generic_closing_or_cta"

MIN_BLOCKING_SIGNALS = 3

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*|\d+(?:\.\d+)?%?")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")

_FORMULAIC_FRAMING_MARKERS = (
    "before drawing conclusions",
    "before treating",
    "before viewing",
    "beneath the surface",
    "clean growth narrative",
    "essential for seeing",
    "full value",
    "genuinely effective",
    "in every environment",
    "it's worth asking",
    "it is worth asking",
    "lasting confidence",
    "meaningful learner adoption",
    "must reconsider",
    "real measure",
    "settled story",
    "the full value",
    "the true shape",
    "underlying fragility",
    "unlock those promised benefits",
    "what stands out is the underlying",
)

_GENERIC_CLOSING_MARKERS = (
    "are you checking whether",
    "before drawing conclusions",
    "before viewing",
    "essential for seeing the real picture",
    "it's worth asking",
    "it is worth asking",
    "must reconsider",
    "truly usable for every employee",
)

_ABSTRACT_CORPORATE_TERMS = {
    "adoption",
    "align",
    "broader",
    "confidence",
    "constraints",
    "durable",
    "effective",
    "frameworks",
    "growth",
    "impact",
    "infrastructure",
    "initiatives",
    "meaningful",
    "momentum",
    "narrative",
    "outcomes",
    "productivity",
    "projections",
    "recognition",
    "resolved",
    "signals",
    "solution",
    "stability",
    "structures",
    "support",
    "systems",
    "value",
}


@dataclass(frozen=True)
class GenericizationGuardResult:
    blocked: bool
    reason: str
    signals: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "signals": list(self.signals),
        }


def evaluate_genericization_selection_blocker(post_text: object) -> GenericizationGuardResult:
    """Return whether post_text is too generic/template-like to select."""

    if not isinstance(post_text, str) or not post_text.strip():
        return GenericizationGuardResult(blocked=False, reason="", signals=())

    normalized = _normalize_text(post_text)
    words = _words(normalized)
    sentences = _sentences(post_text)
    signals: list[str] = []

    formulaic_count = _phrase_count(normalized, _FORMULAIC_FRAMING_MARKERS)
    if formulaic_count >= 2:
        signals.append(SIGNAL_FORMULAIC_POLISHED_FRAMING)

    if _abstract_corporate_density(words) >= 0.11:
        signals.append(SIGNAL_ABSTRACT_CORPORATE_DENSITY)

    if _weak_distinctive_sentence_shape(sentences):
        signals.append(SIGNAL_WEAK_DISTINCTIVE_SENTENCE_SHAPE)

    if _has_generic_closing_or_cta(normalized):
        signals.append(SIGNAL_GENERIC_CLOSING_OR_CTA)

    blocked = (
        SIGNAL_WEAK_DISTINCTIVE_SENTENCE_SHAPE in signals
        and len(signals) >= MIN_BLOCKING_SIGNALS
    )
    reason = (
        "material generic/template-like prose blocks selection: "
        + ", ".join(signals)
        if blocked
        else ""
    )
    return GenericizationGuardResult(
        blocked=blocked,
        reason=reason,
        signals=tuple(signals),
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def _words(text: str) -> list[str]:
    return [word.lower().strip("'") for word in _WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    return [
        " ".join(match.group(0).split())
        for match in _SENTENCE_RE.finditer(text)
        if match.group(0).strip()
    ]


def _phrase_count(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


def _abstract_corporate_density(words: list[str]) -> float:
    if len(words) < 80:
        return 0.0
    abstract_count = sum(1 for word in words if word in _ABSTRACT_CORPORATE_TERMS)
    return abstract_count / len(words)


def _weak_distinctive_sentence_shape(sentences: list[str]) -> bool:
    if len(sentences) < 4:
        return False
    short_sentences = [sentence for sentence in sentences if len(sentence) <= 50]
    return not short_sentences


def _has_generic_closing_or_cta(text: str) -> bool:
    closing = text[-360:]
    return any(marker in closing for marker in _GENERIC_CLOSING_MARKERS)
