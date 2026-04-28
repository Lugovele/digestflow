# DigestFlow

**AI-powered pipeline for generating structured content digests from topic-based sources**

DigestFlow is a local-first Django-based system designed to collect topic-specific information, generate structured digests using AI, and prepare content for further publishing (e.g. LinkedIn posts).

---

## 🔍 What problem it solves

Content creators, analysts, and founders spend significant time:

* monitoring multiple sources
* filtering relevant information
* summarizing insights
* preparing structured content

DigestFlow automates this workflow:

> **Topic → Articles → AI Digest → (future) Content Packaging**

---

## 🧠 Core idea

The system is built around a modular pipeline architecture:

1. **Source stage**
   Collects articles for a given topic (currently demo/mock data)

2. **Digest stage (AI)**
   Converts articles into a structured digest using prompt templates and AI (or mock fallback)

3. **(Planned) Packaging stage**
   Transforms digest into platform-ready content (e.g. LinkedIn posts)

---

## ⚙️ Current status (MVP)

✅ Django admin interface
✅ Topic management
✅ Demo article source
✅ AI digest generation (mock-based)
✅ Structured pipeline execution (`DigestRun`)

🚧 Real external sources (RSS / APIs)
🚧 Real AI integration (requires API key)
🚧 LinkedIn-ready content packaging

---

## 📦 Data flow

```
Topic
  ↓
Demo Source (articles)
  ↓
AI Digest Generator
  ↓
Digest (title, summary, key_points, sources)
```

---

## 🛠 Tech stack

* Python 3.13
* Django 5
* OpenAI API (optional / planned)
* Local-first architecture

---

## ▶️ How to run

```
cd digestflow
python manage.py migrate
python manage.py runserver
```

Admin panel:
http://127.0.0.1:8000/admin/

---

## 🧪 Run demo pipeline

Preview demo articles:

```
python manage.py preview_demo_sources --topic-id 1
```

Run digest stage:

```
python manage.py run_digest_stage --topic-id 1
```

Run AI smoke test:

```
python manage.py ai_digest_smoke_test --topic "AI automation"
```

---

## 🔐 Environment

Create `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

If not set, the system automatically uses mock responses.

---

## 🧩 Architecture principles

* Pipeline-first design
* Clear separation of stages
* AI as a replaceable component
* Debug-friendly execution
* No overengineering (no agents, no orchestration yet)

---

## 📈 Why this project

This project demonstrates:

* designing AI-driven workflows
* building modular backend systems
* working with prompt templates
* integrating (or mocking) LLM APIs
* structuring data pipelines end-to-end

---

## 👤 Author

Elena Lugovaya
AI Automation / Workflow Systems
