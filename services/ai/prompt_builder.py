"""Загрузка prompt template с явной подстановкой переменных."""
from __future__ import annotations

from pathlib import Path

from django.conf import settings


class PromptTemplateError(ValueError):
    pass


def build_prompt(template_path: str, **context: object) -> str:
    """Загрузить prompt из prompts/ и подставить переданный context."""
    full_path = (settings.BASE_DIR / "prompts" / template_path).resolve()
    prompts_root = (settings.BASE_DIR / "prompts").resolve()

    if prompts_root not in full_path.parents:
        raise PromptTemplateError("Template path must stay inside prompts directory.")
    if not full_path.exists():
        raise PromptTemplateError(f"Prompt template not found: {template_path}")

    template = Path(full_path).read_text(encoding="utf-8")
    try:
        return template.format(**context)
    except KeyError as exc:
        raise PromptTemplateError(f"Missing prompt variable: {exc}") from exc
