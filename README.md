# DigestFlow

AI-powered local-first Django pipeline for turning a topic into:

- a structured digest
- a LinkedIn-ready content package

Given a topic, DigestFlow produces:

- a structured factual digest
- a publish-ready LinkedIn post with hooks, CTAs, and hashtags

## What it does

DigestFlow currently supports a full MVP pipeline:

`Topic / user input -> demo source items -> cleaner -> dedupe -> ranking -> AI digest -> LinkedIn package -> result page`

DigestFlow separates:

- synthesis (`Digest`)
- packaging (`ContentPackage`)

This makes the system easier to validate, debug, and extend.

The system is designed to be debug-friendly and safe to iterate on locally:

- demo source instead of real external ingestion
- structured validation for AI output
- mock fallback when AI is unavailable or returns invalid output
- `DigestRun` metrics and console logging
- minimal Django UI for creating a topic, running the pipeline, and viewing the result

## Current MVP status

Implemented:

- Django backend and admin
- Topic model and Topic-based runs
- demo source stage
- cleaner
- dedupe by URL and normalized title
- deterministic topic-aware ranking
- Article storage
- AI digest stage
- LinkedIn packaging stage
- structured validation for digest and package output
- mock fallback for AI failures / invalid JSON
- token and estimated cost tracking
- `DigestRun` metrics
- console logging
- minimal web UI
- integration tests covering:
  - completed
  - partial_failed
  - failed
- unit tests for:
  - cleaner
  - deduper
  - ranker
  - digest validators
  - packaging validators
  - token / cost helpers

Current limitations:

- source ingestion is demo-only
- URLs are synthetic
- real AI calls may fallback to mock
- UI is minimal and not production-ready

## Tech stack

- Python 3.13
- Django 5.2
- SQLite
- OpenAI API (optional)

## Project flow

```text
Topic / user input
  ->
Demo source items
  ->
Cleaner
  ->
Dedupe
  ->
Ranking
  ->
AI Digest
  ->
LinkedIn ContentPackage
  ->
Result page / admin / metrics
```

## Local setup

```powershell
cd C:\Users\Елена\Documents\DigestFlow
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

Open:

- UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- admin: [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/)
- health: [http://127.0.0.1:8000/health/](http://127.0.0.1:8000/health/)

## How to use (UI)

1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
2. Enter a topic, for example: `"AI automation for operations"`
3. Click `Generate digest`
4. You will be redirected to `/runs/<id>/`

On the result page you can:

- read the structured digest
- copy the LinkedIn post
- review hooks, CTAs, and hashtags
- inspect validation and metrics

## Environment

Create `.env` in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=45
AI_DAILY_TOKEN_BUDGET=100000
```

If `OPENAI_API_KEY` is missing, placeholder, or the model returns unusable output, DigestFlow falls back to mock responses.

## Useful commands

Check project health:

```powershell
.\.venv\Scripts\python.exe manage.py check
```

Run tests:

```powershell
.\.venv\Scripts\python.exe manage.py test
```

Preview demo source items:

```powershell
.\.venv\Scripts\python.exe manage.py preview_demo_sources --topic-id 2
```

Run digest smoke test:

```powershell
.\.venv\Scripts\python.exe manage.py ai_digest_smoke_test --topic "AI automation"
```

Run digest stage only:

```powershell
.\.venv\Scripts\python.exe manage.py run_digest_stage --topic-id 2
```

Run packaging stage only:

```powershell
.\.venv\Scripts\python.exe manage.py run_packaging_stage --digest-id 1
```

Run full demo pipeline:

```powershell
.\.venv\Scripts\python.exe manage.py run_digest_demo --topic-id 2
```

## Observability

Where to look:

- console logs while running commands or the web app
- `DigestRun.metrics` in admin
- result page `/runs/<id>/`

Metrics currently include:

- raw items count
- count after cleaning / dedupe / ranking
- selected articles for prompt
- used `Article.id` list
- digest / packaging status
- provider (`openai` / `mock`)
- token usage when available
- estimated cost when available

## Design principles

- pipeline-first
- deterministic preprocessing before AI
- structured validation before accepting AI output
- simple local observability
- minimal UI before product polish
- no agents
- no async orchestration
- no external monitoring stack

## Why this project exists

DigestFlow is built to explore a production-style AI pipeline:

- deterministic preprocessing before LLM
- strict output validation
- controlled failure handling
- traceable execution via metrics

It is intentionally local-first and minimal to make iteration fast and debugging transparent.
