# DigestFlow Entities v1

## Purpose

This document defines the core product entities needed to support the UX flows, run states, and state transitions already described in:

- [docs/ux-flows-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/ux-flows-v1.md)
- [docs/run-states-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/run-states-v1.md)
- [docs/state-transitions-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/state-transitions-v1.md)

This is a planning document only.

It does not change code, models, migrations, or UI.

The goal is to identify the conceptual product entities that the system either already has or will likely need as DigestFlow evolves from a simple digest generator into an editorial system.

---

## Entity Principles

The entity model should help us separate:

- topic setup from run execution
- source strategy from article outcomes
- product data from technical telemetry
- single-topic flows from multi-stream flows
- digest output from packaging output

The point of this document is not to lock the schema. The point is to create a stable shared vocabulary.

---

## 1. User

### Purpose

Represents the person or account using DigestFlow to define topics, sources, and digest runs.

### Key fields

- `id`
- `name` or `username`
- `email` or authentication identity
- preferences
- created/updated timestamps

### Relationships

- one `User` can own many `Topic`
- one `User` can own many `Run`
- one `User` may own many `Source`
- one `User` may define one or more `AudienceProfile`

### Current status

Already exists

### Why it is needed

Needed for ownership, saved topics, saved source sets, and future personalization of editorial behavior.

---

## 2. Topic

### Purpose

Represents the user’s broad or specific subject of interest before or during refinement.

### Key fields

- `id`
- `user_id`
- `name`
- `description`
- `keywords`
- `excluded_keywords`
- `source_url` or source seed
- active/inactive status
- created/updated timestamps

### Relationships

- belongs to one `User`
- may have one or more `DigestStream`
- may have many `Run`
- may reference one preferred `AudienceProfile`
- may produce many `RefinementSuggestion`

### Current status

Already exists, but broad-topic and multi-direction use is only partial

### Why it is needed

Needed because the product starts with a topic, but that topic is not always directly usable as a digest stream.

---

## 3. DigestStream

### Purpose

Represents a refined editorial direction derived from a topic.

Examples:

- Topic: AI
- Digest streams:
  - AI engineering
  - AI business
  - AI tools

### Key fields

- `id`
- `topic_id`
- `name`
- `direction_label`
- normalized keywords
- excluded keywords
- source strategy reference
- default threshold
- audience reference
- active/inactive state

### Relationships

- belongs to one `Topic`
- may have many `Run`
- may use many `Source`
- may target one `AudienceProfile`

### Current status

Future/planned

### Why it is needed

Needed to support the multiple-directions flow without forcing unrelated content into one digest.

---

## 4. Source

### Purpose

Represents a user-provided or system-selected content source.

This is broader than “RSS feed.” A source may be:

- RSS feed
- tag page
- blog
- company news page
- website
- newsletter landing page
- API-backed source

### Key fields

- `id`
- original URL
- normalized URL
- source type
- platform
- validation status
- source mode compatibility
- source metadata
- created/updated timestamps

### Relationships

- may belong to one `User`
- may be attached to one `Topic`
- may be attached to one `DigestStream`
- may produce many `Article`
- may be referenced by many `Run`

### Current status

Partially exists conceptually in normalization and source ingestion, but not yet as a first-class persistent entity

### Why it is needed

Needed for user-curated sources, source validation UX, source mismatch warnings, and source strategy traceability.

---

## 5. SourceMode

### Purpose

Represents the strategy for how sources are chosen for a run.

Examples:

- Automatic
- Custom only
- Hybrid

### Key fields

- `mode_name`
- behavioral flags:
  - allow_auto_discovery
  - allow_user_sources
  - require_valid_custom_sources

### Relationships

- may be attached to a `Topic`
- may be attached to a `DigestStream`
- may be attached to a `Run`

### Current status

Future/planned as a first-class product concept

### Why it is needed

Needed because source selection is part of the product UX, not just a hidden backend choice.

---

## 6. Article

### Purpose

Represents a collected content item that can be ranked and potentially used in a digest.

### Key fields

- `id`
- `topic_id` or stream reference
- title
- URL
- source URL
- source API URL
- source name
- published timestamp
- content
- metadata
- raw payload

### Relationships

- may belong to one `Topic`
- may later belong more specifically to one `DigestStream`
- belongs to one `Source` conceptually
- may be linked to one or more `Run`
- may contribute to one `Digest`

### Current status

Already exists, with partial source-trace fields now present in payload/raw data

### Why it is needed

Needed because the system ranks and reasons over articles, not just sources or summaries.

---

## 7. Run

### Purpose

Represents one execution attempt of the digest pipeline under a specific configuration.

### Key fields

- `id`
- topic or stream reference
- run status
- input snapshot
- started_at
- finished_at
- error message
- metrics
- created/updated timestamps

### Relationships

- belongs to one `Topic`
- may later belong more explicitly to one `DigestStream`
- may include many `RunStage`
- may have one `RunQualityReport`
- may produce one `Digest`

### Current status

Already exists

### Why it is needed

Needed to represent execution lifecycle and user-visible outcomes separately from saved topics and from digest payloads.

---

## 8. RunStage

### Purpose

Represents a distinct execution stage inside a run.

Examples:

- collecting sources
- ranking articles
- generating digest
- generating content package

### Key fields

- `id`
- `run_id`
- stage name
- stage status
- started_at
- finished_at
- stage-level diagnostics
- retry metadata

### Relationships

- belongs to one `Run`

### Current status

Future/planned as a first-class entity

### Why it is needed

Needed if stage-level status, retry, and diagnostics become product-visible rather than living only inside JSON metrics.

---

## 9. RunQualityReport

### Purpose

Represents the editorial quality interpretation of a run outcome.

It explains not only whether the run succeeded technically, but whether the result was actually good enough.

### Key fields

- `id`
- `run_id`
- threshold used
- articles above threshold
- rejected count
- quality distribution
- top rejected articles
- explanation text
- recommended next actions

### Relationships

- belongs to one `Run`
- may reference many `Article`

### Current status

Partially exists inside ranking metrics and diagnostics, but not yet as a first-class entity

### Why it is needed

Needed for insufficient-quality UX, explainable ranking, and future editorial feedback loops.

---

## 10. Digest

### Purpose

Represents the structured editorial output built from selected articles.

### Key fields

- `id`
- `run_id`
- title
- payload
- quality score
- generated timestamp

### Relationships

- belongs to one `Run`
- contains many digest article payloads conceptually
- may have one `ContentPackage`

### Current status

Already exists and is now the canonical product output container through `payload`

### Why it is needed

Needed because digest output must be stored as product data and remain distinct from metrics and packaging.

---

## 11. ContentPackage

### Purpose

Represents audience-facing packaged output derived from a digest.

Examples:

- LinkedIn post
- hook variants
- CTA variants
- hashtags
- carousel outline

### Key fields

- `id`
- `digest_id`
- post text
- hook variants
- CTA variants
- hashtags
- carousel outline
- validation report

### Relationships

- belongs to one `Digest`
- may later reference one `AudienceProfile`

### Current status

Already exists

### Why it is needed

Needed because packaging is a separate outcome layer, not part of the digest itself.

---

## 12. AudienceProfile

### Purpose

Represents the intended audience for ranking and packaging decisions.

Examples:

- AI engineers
- founders
- marketers
- creators
- researchers
- investors
- automation specialists

### Key fields

- `id`
- name
- description
- ranking preferences
- packaging preferences
- default tone or framing rules

### Relationships

- may belong to one `User`
- may be attached to one `Topic`
- may be attached to one `DigestStream`
- may be attached to one `Run`
- may influence one `ContentPackage`

### Current status

Future/planned

### Why it is needed

Needed because article relevance and packaging quality depend strongly on audience context.

---

## 13. RefinementSuggestion

### Purpose

Represents a suggested topic direction proposed when a topic is broad or ambiguous.

### Key fields

- `id`
- parent topic reference
- suggestion label
- suggestion description
- keywords or direction metadata
- confidence or priority
- suggestion source

### Relationships

- belongs to one `Topic`
- may later be turned into one `DigestStream`

### Current status

Future/planned

### Why it is needed

Needed for the broad-topic refinement flow and for creating a clean bridge between topic setup and digest stream creation.

---

## Minimal MVP Entities

These are the entities that are either already present or most likely required first for a stable functional product architecture.

### Existing or near-existing MVP core

- `User`
- `Topic`
- `Article`
- `Run`
- `Digest`
- `ContentPackage`

### High-priority additions or conceptual formalizations

- `DigestStream`
- `Source`
- `SourceMode`
- `RunQualityReport`

Why these matter first:

- `DigestStream` solves broad-topic splitting
- `Source` and `SourceMode` solve source strategy UX
- `RunQualityReport` solves explainable insufficient-quality outcomes

---

## Future Entities

These are important, but may not need to become first-class data entities immediately.

- `RunStage`
- `AudienceProfile`
- `RefinementSuggestion`

They may first appear as structured configuration or derived objects before becoming persistent models.

---

## Open Questions

### 1. Should `DigestStream` be a true model or a derived topic variant?

This is one of the most important architectural choices.

### 2. Should `SourceMode` be an enum field or its own entity?

A simple field may be enough at first, but a separate entity may become useful if source behavior becomes more configurable.

### 3. Should `RunStage` be stored as structured JSON or as a real model?

If retries, timing, and stage-level UX become important, a real model may be the cleaner path.

### 4. How should `AudienceProfile` be represented initially?

Options:

- simple enum
- saved per-user preset
- richer configuration object

### 5. Where should `RefinementSuggestion` live?

It may be:

- generated on the fly
- stored temporarily
- saved permanently for topic history

### 6. How should `Article` ownership evolve?

Today article association is topic-oriented. Multi-stream behavior may require more explicit stream-level ownership or per-run selection relationships.

---

## Implementation Notes

- This document does not require immediate implementation.
- It should guide future model and service design.
- The next implementation step should identify which of these entities need to become actual Django models first.
- The likely first practical design pass should focus on:
  - `DigestStream`
  - `Source`
  - `SourceMode`
  - `RunQualityReport`
- Avoid over-modeling too early.
- Prefer introducing new entities only when they clearly solve:
  - a user-facing flow problem
  - a state-management problem
  - a traceability problem

The main architectural goal is to keep DigestFlow understandable as the product grows:

- topics become directions
- directions become streams
- streams use sources
- sources produce articles
- runs select and rank articles
- digests store editorial output
- content packages adapt that output for an audience

