from __future__ import annotations

import json

from django.conf import settings


DEFAULT_AUTHOR_PROFILE = {
    "role": "AI Automation Specialist",
    "background": "Builds and improves workflow systems.",
    "focus": "workflow design, validation, reusable systems",
    "voice": "analytical",
    "style_constraints": [
        "avoid generic marketing language",
        "focus on systems, not tools",
        "connect facts into insights",
    ],
}


def load_author_profile() -> dict:
    """Load author profile config for packaging personalization."""
    profile_path = settings.BASE_DIR / "configs" / "author_profile.json"
    try:
        if not profile_path.exists():
            return dict(DEFAULT_AUTHOR_PROFILE)

        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return dict(DEFAULT_AUTHOR_PROFILE)
    except Exception:
        return dict(DEFAULT_AUTHOR_PROFILE)

    style_constraints = payload.get("style_constraints")
    if not isinstance(style_constraints, list) or len(style_constraints) < 3:
        style_constraints = DEFAULT_AUTHOR_PROFILE["style_constraints"]

    return {
        "role": str(payload.get("role") or DEFAULT_AUTHOR_PROFILE["role"]),
        "background": str(payload.get("background") or DEFAULT_AUTHOR_PROFILE["background"]),
        "focus": str(payload.get("focus") or DEFAULT_AUTHOR_PROFILE["focus"]),
        "voice": str(payload.get("voice") or DEFAULT_AUTHOR_PROFILE["voice"]),
        "style_constraints": [str(item) for item in style_constraints[:3]],
    }
