"""Минимальные настройки Django для локальной разработки DigestFlow."""
from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _postflow_model_setting(env_name: str, default: str) -> str:
    return os.getenv(env_name) or os.getenv("OPENAI_MODEL") or default


def _postflow_provider_setting(env_name: str) -> str:
    return os.getenv(env_name, "openai").strip().lower() or "openai"


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-insecure-secret-key")
DEBUG = os.getenv("DJANGO_DEBUG", "True").lower() == "true"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "apps.topics",
    "apps.sources",
    "apps.digests",
    "apps.packaging",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "ru-ru"
TIME_ZONE = "Europe/Moscow"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Legacy shared model setting. PostFlow stage-specific settings below should be
# preferred by new staged pipeline code; keep this as a compatibility fallback.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))
ANTHROPIC_TIMEOUT_SECONDS = int(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", str(OPENAI_TIMEOUT_SECONDS)))
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", str(OPENAI_TIMEOUT_SECONDS)))
AI_DAILY_TOKEN_BUDGET = int(os.getenv("AI_DAILY_TOKEN_BUDGET", "100000"))
POSTFLOW_RESEARCH_PROVIDER = _postflow_provider_setting("POSTFLOW_RESEARCH_PROVIDER")
POSTFLOW_POST_PROVIDER = _postflow_provider_setting("POSTFLOW_POST_PROVIDER")
POSTFLOW_RESEARCH_MODEL = _postflow_model_setting("POSTFLOW_RESEARCH_MODEL", "gpt-4o-mini-2024-07-18")
POSTFLOW_POST_MODEL = _postflow_model_setting("POSTFLOW_POST_MODEL", "gpt-4.1-2025-04-14")
SEARCH_PROVIDER_ENABLED = os.getenv("SEARCH_PROVIDER_ENABLED", "False").lower() == "true"
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "").strip().lower()
SEARCH_PROVIDER_API_KEY = os.getenv("SEARCH_PROVIDER_API_KEY", "")
SEARCH_RECENCY_MONTHS = int(os.getenv("SEARCH_RECENCY_MONTHS", "1"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "digestflow": {
            "format": "[%(asctime)s] %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "digestflow",
        },
    },
    "loggers": {
        "services": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
