import re
from dataclasses import dataclass


FOCUS_VALIDATION_MESSAGE = "Enter a focus word or phrase."
FOCUS_TOO_SHORT_MESSAGE = "This is too short to use as a focus."
FOCUS_NUMBER_ONLY_MESSAGE = "Add a word so this focus is understandable."
FOCUS_DUPLICATE_MESSAGE = "This focus is already added."
FOCUS_TOO_LONG_MESSAGE = "Try a shorter word or phrase."
MAX_FOCUS_TERM_LENGTH = 80

_FOCUS_ABBREVIATIONS = {
    "ai", "api", "llm", "ml", "nlp", "rag", "mcp", "etl", "seo", "crm", "erp",
    "saas", "b2b", "b2c", "ui", "ux", "qa", "devops",
}
_KNOWN_NOISE_TERMS = {
    "asdfasdf",
    "qwerty",
    "pumpumpum",
    "767ghjb;k",
    "фывафыва",
    "олололо",
}
_CYRILLIC_SHORT_ALLOWLIST = set()
_ALPHA_ONLY_RE = re.compile(r"[^A-Za-zА-Яа-яЁё]")
_INVALID_CHAR_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё+\-\/& ]")
_KEYBOARD_MASH_RE = re.compile(
    r"(qwerty|asdf|zxcv|ghjb|jkl|йцук|фыва|ячсм)",
    re.IGNORECASE,
)
_REPEATED_CHUNK_RE = re.compile(r"^([A-Za-zА-Яа-яЁё]{1,4})\1+$", re.IGNORECASE)
_CYRILLIC_VOWELS = set("аеёиоуыэюя")


@dataclass(frozen=True)
class FocusValidationIssue:
    term: str
    message: str


def clean_focus_terms(raw_terms: list[str]) -> list[str]:
    cleaned_terms: list[str] = []
    seen_terms: set[str] = set()
    for raw_term in raw_terms:
        term = " ".join(str(raw_term or "").strip().split())
        if not term:
            continue
        normalized_term = term.casefold()
        if normalized_term in seen_terms:
            continue
        seen_terms.add(normalized_term)
        cleaned_terms.append(term)
    return cleaned_terms


def validate_new_focus_terms(existing_terms: list[str], submitted_terms: list[str]) -> FocusValidationIssue | None:
    existing_lookup = {str(term).casefold() for term in existing_terms}
    for term in submitted_terms:
        if term.casefold() in existing_lookup:
            continue
        if not is_meaningful_focus_term(term):
            return FocusValidationIssue(term=term, message=get_focus_validation_message(term))
    return None


def get_focus_validation_message(term: str) -> str:
    normalized = " ".join(str(term or "").strip().split())
    if not normalized or not any(char.isalnum() for char in normalized):
        return FOCUS_VALIDATION_MESSAGE
    if len(normalized) > MAX_FOCUS_TERM_LENGTH:
        return FOCUS_TOO_LONG_MESSAGE
    if all(part.isdigit() for part in normalized.split()):
        return FOCUS_NUMBER_ONLY_MESSAGE
    return FOCUS_TOO_SHORT_MESSAGE


def is_meaningful_focus_term(term: str) -> bool:
    normalized = " ".join(str(term or "").strip().split())
    if not normalized:
        return False
    if len(normalized) > MAX_FOCUS_TERM_LENGTH:
        return False

    lowered = normalized.casefold()
    if lowered in _KNOWN_NOISE_TERMS:
        return False
    if lowered in _FOCUS_ABBREVIATIONS:
        return True
    if _INVALID_CHAR_RE.search(normalized):
        return False
    if _KEYBOARD_MASH_RE.search(normalized):
        return False
    if re.search(r"(.)\1{2,}", lowered):
        return False

    if " " in normalized:
        parts = normalized.split()
        return len(parts) >= 2 and all(_is_meaningful_focus_phrase_token(part) for part in parts)

    return _is_meaningful_focus_token(normalized)


def _is_meaningful_focus_phrase_token(token: str) -> bool:
    lowered = token.casefold()
    if lowered in _FOCUS_ABBREVIATIONS:
        return True
    if _is_contextual_numeric_token(token):
        return True

    alpha_only = _alpha_only(lowered)
    if _is_short_cyrillic_fragment(alpha_only):
        return False
    if len(alpha_only) < 2:
        return False
    if _KEYBOARD_MASH_RE.search(token):
        return False
    if _REPEATED_CHUNK_RE.match(alpha_only):
        return False
    if _has_too_little_signal(alpha_only):
        return False
    return True


def _is_meaningful_focus_token(token: str) -> bool:
    lowered = token.casefold()
    if lowered in _FOCUS_ABBREVIATIONS:
        return True

    alpha_only = _alpha_only(lowered)
    if _is_short_cyrillic_fragment(alpha_only):
        return False
    if len(alpha_only) < 2:
        return False
    if _KEYBOARD_MASH_RE.search(token):
        return False
    if _REPEATED_CHUNK_RE.match(alpha_only):
        return False
    if _has_too_little_signal(alpha_only):
        return False
    return True


def _alpha_only(value: str) -> str:
    return _ALPHA_ONLY_RE.sub("", value)


def _is_contextual_numeric_token(token: str) -> bool:
    stripped = str(token or "").strip()
    return stripped.isdigit() and 1 <= len(stripped) <= 2


def _has_too_little_signal(alpha_only: str) -> bool:
    if len(set(alpha_only)) <= 2 and len(alpha_only) >= 4:
        return True

    if _looks_like_repeated_syllable(alpha_only):
        return True

    if _looks_like_cyrillic_gibberish(alpha_only):
        return True

    return False


def _is_short_cyrillic_fragment(alpha_only: str) -> bool:
    if not alpha_only:
        return False

    is_cyrillic = all("а" <= ch <= "я" or ch == "ё" for ch in alpha_only)
    if not is_cyrillic:
        return False

    if alpha_only in _CYRILLIC_SHORT_ALLOWLIST:
        return False

    return len(alpha_only) <= 3


def _looks_like_repeated_syllable(alpha_only: str) -> bool:
    if len(alpha_only) < 6:
        return False

    for chunk_len in range(2, 5):
        if len(alpha_only) % chunk_len != 0:
            continue
        chunk = alpha_only[:chunk_len]
        if chunk * (len(alpha_only) // chunk_len) == alpha_only:
            return True
    return False


def _looks_like_cyrillic_gibberish(alpha_only: str) -> bool:
    if not alpha_only:
        return False

    is_cyrillic = all("а" <= ch <= "я" or ch == "ё" for ch in alpha_only)
    if not is_cyrillic:
        return False

    if len(alpha_only) <= 3:
        return False

    vowel_count = sum(1 for ch in alpha_only if ch in _CYRILLIC_VOWELS)
    consonant_count = len(alpha_only) - vowel_count

    if vowel_count == 0:
        return True

    if vowel_count == 1 and len(alpha_only) >= 5:
        return True

    if consonant_count >= len(alpha_only) - 1 and len(alpha_only) >= 6:
        return True

    return False
