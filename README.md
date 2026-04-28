# DigestFlow

DigestFlow - local-first Django-система для сбора информации по пользовательским темам, генерации структурированных дайджестов и подготовки LinkedIn-ready пакетов контента для ручной публикации.

## Требования

- Python 3.10+
- SQLite для локальной разработки
- OpenAI API key для AI-этапов

## Локальный запуск

```bash
cd digestflow
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Проверка healthcheck:

```bash
curl http://127.0.0.1:8000/health/
```

## MVP-архитектура

MVP строится как pipeline-first:

1. сбор источников
2. очистка источников
3. дедупликация источников
4. ранжирование и валидация источников
5. генерация дайджеста
6. упаковка дайджеста для LinkedIn
7. валидация пакета перед ручной публикацией

В MVP нет автопостинга, Celery, async-оркестрации и multi-agent workflow.

## Ключевая сущность

`DigestRun` - запись выполнения каждого запуска pipeline. Она хранит статус, snapshot входных данных, метрики, ошибки и timing.

## Структура проекта

- `apps/` - Django-приложения и модели БД
- `services/` - сервисы pipeline, processing, AI, sources и packaging
- `prompts/` - структурированные prompt templates
- `tests/` - unit и integration tests

## Следующие шаги разработки

1. Добавить migrations для стартовых моделей.
2. Зарегистрировать Topic, DigestRun, Digest и ContentPackage в Django Admin.
3. Реализовать search adapters для источников.
4. Заменить placeholder-генерацию дайджеста и пакета на AI generation с JSON validation.
5. Добавить тесты для cleaning, deduplication, prompt building и переходов состояния pipeline.
