# DigestFlow Entity Lifecycle v1

## Purpose

This document classifies DigestFlow entities by lifecycle before deciding which ones should become Django models.

It is based on:

- [docs/ux-flows-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/ux-flows-v1.md)
- [docs/run-states-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/run-states-v1.md)
- [docs/state-transitions-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/state-transitions-v1.md)
- [docs/entities-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/entities-v1.md)

This is a planning document only.

It does not change code, create migrations, or redesign UI.

The goal is to answer a practical product-engineering question:

- which entities should be stored now
- which entities should stay ephemeral for now
- which entities are derived outputs rather than first-class stored models
- which future entities should wait until the UX is more stable

---

## Lifecycle Categories

### 1. Persistent entities

Entities that should usually be stored because they represent stable product state, ownership, or reusable output.

### 2. Ephemeral runtime entities

Entities that exist mainly while a run is executing or while the user is stepping through a flow. They may be logged or summarized, but do not necessarily need full first-class persistence yet.

### 3. Derived/generated entities

Entities that are produced from other stored inputs and may be regenerated or represented as structured payloads rather than normalized models.

### 4. Future/planned entities

Entities that are useful conceptually, but should not be modeled too early unless the product flow clearly requires them.

---

## Entity Classification

## 1. User

- **lifecycle type:** Persistent entity
- **should it be stored in the database now?:** Yes
- **why / why not:** User ownership is fundamental to topics, runs, preferences, and future collaboration or personalization.
- **what creates it:** Authentication or local setup flow
- **what updates it:** Account changes, preference updates, future editorial settings
- **what depends on it:** Topic, Run, Source ownership, AudienceProfile ownership
- **MVP relevance:** High

---

## 2. Topic

- **lifecycle type:** Persistent entity
- **should it be stored in the database now?:** Yes
- **why / why not:** A topic is a stable user-owned concept that anchors runs, source defaults, and future refinement history.
- **what creates it:** User topic creation flow
- **what updates it:** User edits, source updates, keyword adjustments, future refinement choices
- **what depends on it:** Run, DigestStream, Source strategy, RefinementSuggestion
- **MVP relevance:** High

---

## 3. DigestStream

- **lifecycle type:** Future/planned entity
- **should it be stored in the database now?:** Not yet
- **why / why not:** It is conceptually important, but it should become persistent only when multi-direction and multi-stream UX are actually implemented. Before that, it risks becoming a premature abstraction.
- **what creates it:** Topic refinement and multi-direction flow
- **what updates it:** User direction changes, source strategy changes, audience selection changes
- **what depends on it:** Stream-level Run, Source assignment, Digest grouping
- **MVP relevance:** Medium now, high later

---

## 4. Source

- **lifecycle type:** Persistent entity
- **should it be stored in the database now?:** Probably not as a full model yet
- **why / why not:** Source behavior matters, but a full stored Source model may be premature until source curation, validation history, and source mode UX are stable. For now, source details can remain attached to Topic, Run input, and Article metadata.
- **what creates it:** User source entry flow or automatic source selection
- **what updates it:** Validation, normalization, future manual edits
- **what depends on it:** Article traceability, source mismatch detection, source mode behavior
- **MVP relevance:** Medium

---

## 5. SourceMode

- **lifecycle type:** Ephemeral runtime entity at first, possibly persistent later
- **should it be stored in the database now?:** Not as its own model yet
- **why / why not:** It is primarily configuration. A simple field on Topic or Run is likely enough before introducing a separate entity.
- **what creates it:** User source mode selection or default system strategy
- **what updates it:** User configuration changes
- **what depends on it:** Source collection behavior, source validation logic, retry paths
- **MVP relevance:** Medium

---

## 6. Article

- **lifecycle type:** Persistent entity
- **should it be stored in the database now?:** Yes
- **why / why not:** Articles are already a real product data layer. They are needed for traceability, ranking, debugging, and digest generation.
- **what creates it:** Source collection and article persistence stage
- **what updates it:** Cleaning, normalization, metadata enrichment, future reprocessing
- **what depends on it:** Ranking, RunQualityReport, Digest generation, UI diagnostics
- **MVP relevance:** High

---

## 7. Run

- **lifecycle type:** Persistent entity
- **should it be stored in the database now?:** Yes
- **why / why not:** Runs are a stable execution and audit object. They anchor status, metrics, errors, and output relationships.
- **what creates it:** User starts a run or a future scheduled trigger starts it
- **what updates it:** Pipeline progression, status transitions, metrics, failures
- **what depends on it:** Digest, run detail UI, retry logic, diagnostics
- **MVP relevance:** High

---

## 8. RunStage

- **lifecycle type:** Ephemeral runtime entity, possibly persistent later
- **should it be stored in the database now?:** No
- **why / why not:** The concept is useful, but stage data can still live in structured metrics while the flow remains relatively simple. A model becomes more useful when stage-level retry, auditing, or timing becomes product-facing.
- **what creates it:** Pipeline execution
- **what updates it:** Stage progression
- **what depends on it:** Detailed progress UI, stage-specific diagnostics, future retry tools
- **MVP relevance:** Low to medium

---

## 9. RunQualityReport

- **lifecycle type:** Derived/generated entity
- **should it be stored in the database now?:** Not as a separate model yet
- **why / why not:** The information is already being derived from ranking and diagnostics. For now, structured data in run metrics or attached payload is enough. A dedicated model may come later if editorial reporting becomes a first-class product feature.
- **what creates it:** Ranking and quality filtering logic
- **what updates it:** Threshold changes, reruns, future diagnostic enrichment
- **what depends on it:** Insufficient-quality UX, editorial explanation layer, future reporting
- **MVP relevance:** Medium

---

## 10. Digest

- **lifecycle type:** Persistent entity
- **should it be stored in the database now?:** Yes
- **why / why not:** Digest is canonical product output and should remain persistent.
- **what creates it:** Digest generation stage
- **what updates it:** Usually replaced or regenerated per run, not edited frequently in place
- **what depends on it:** ContentPackage, digest view, future export or sharing features
- **MVP relevance:** High

---

## 11. ContentPackage

- **lifecycle type:** Persistent entity
- **should it be stored in the database now?:** Yes
- **why / why not:** Packaged content is a real product output and should remain available after generation.
- **what creates it:** Packaging stage
- **what updates it:** Regeneration, future editing workflows
- **what depends on it:** LinkedIn/post output views, audience-oriented output, future export features
- **MVP relevance:** High

---

## 12. AudienceProfile

- **lifecycle type:** Future/planned entity
- **should it be stored in the database now?:** Not yet
- **why / why not:** Audience matters conceptually, but before audience-aware ranking and packaging become stable, a simple field or enum is safer than a full model.
- **what creates it:** User audience selection or system default
- **what updates it:** User preference changes, future saved audience presets
- **what depends on it:** Ranking behavior, packaging prompts, source relevance interpretation
- **MVP relevance:** Medium now, high later

---

## 13. RefinementSuggestion

- **lifecycle type:** Derived/generated entity
- **should it be stored in the database now?:** No
- **why / why not:** Suggestions are generated from the current topic context. They should remain lightweight until the refinement UX proves that saved suggestion history matters.
- **what creates it:** Topic interpretation or refinement engine
- **what updates it:** Usually regenerated when the topic changes
- **what depends on it:** Broad-topic refinement flow, stream creation flow
- **MVP relevance:** Medium

---

## Minimal MVP Persistence Model

The minimal persistence model should stay narrow and support what the product already truly needs.

### Recommended persistent core now

- `User`
- `Topic`
- `Article`
- `Run`
- `Digest`
- `ContentPackage`

### Recommended lightweight configuration, not full entity models yet

- `SourceMode` as a field or config value
- source details as topic/run/article metadata
- audience as a simple field or enum

### Recommended derived/diagnostic layer, not full models yet

- `RunStage`
- `RunQualityReport`
- `RefinementSuggestion`

This keeps the persistence model aligned with current product reality while leaving room to grow.

---

## What Not to Store Yet

These are the main things that should probably not become first-class models immediately.

### 1. `RunStage`

Why not yet:

- too early if stages still live comfortably in metrics
- risks duplicating state before state ownership is settled

### 2. `RunQualityReport`

Why not yet:

- editorial diagnostics are still evolving
- report shape may change quickly as ranking logic improves

### 3. `RefinementSuggestion`

Why not yet:

- suggestions are naturally generated from current context
- it is not clear yet that suggestion history is product-critical

### 4. `AudienceProfile` as a full model

Why not yet:

- audience behavior may start as a simpler enum or config object
- a full model makes more sense once reusable saved audience presets become a real feature

### 5. `DigestStream` as a persistent model

Why not yet:

- multi-stream UX is still in planning
- the concept is important, but model-level persistence can wait until stream management is real

### 6. `Source` as a normalized model

Why not yet:

- source ingestion and validation are still changing
- storing sources too early may lock in structure before source mode UX is stable

---

## Risks of Over-Modeling Too Early

### 1. Freezing the wrong abstractions

If we turn every planned concept into a Django model too early, we risk committing to the wrong product boundaries.

### 2. Duplicating responsibility

For example:

- `RunStage` vs metrics
- `RunQualityReport` vs ranking diagnostics
- `Source` vs article/source metadata

This can create multiple “truths” before the system is mature enough to justify them.

### 3. Making iteration slower

Every extra model introduces:

- migrations
- admin concerns
- serialization questions
- ownership questions
- cleanup burden

That is expensive if the UX is still shifting.

### 4. Making the product harder to reason about

Too many early models can make a small product feel more complex than it really is, especially when some entities are still conceptual.

### 5. Storing unstable generated artifacts too aggressively

Things like refinement suggestions and quality reports may change shape often as heuristics improve. Over-persisting them too early can create migration churn without product payoff.

---

## Implementation Notes

- This document does not require immediate implementation.
- It should guide future model selection and persistence decisions.
- A good next step would be a “first real model changes” proposal that asks:
  - which future entity delivers immediate UX value
  - which entity reduces backend ambiguity
  - which entity is still better left as config or derived output

### Recommended near-term bias

Prefer:

- a smaller persistence model
- richer payloads and structured diagnostics
- clear ownership boundaries

Avoid:

- introducing every conceptual entity as a model immediately
- storing unstable generated structures without clear product need

### Practical rule of thumb

An entity is a good candidate for persistence when at least one of these is true:

- the user expects it to exist later
- multiple flows depend on it
- audit/history matters
- reruns or edits depend on it
- it reduces ambiguity in current business logic

If none of those are true yet, it is often safer to keep the entity derived or ephemeral.

