# DigestFlow First Model Changes v1

## Purpose

This document defines the first minimal Django model changes needed to support the evolving DigestFlow architecture without over-engineering.

It is based on:

- [docs/entities-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/entities-v1.md)
- [docs/entity-lifecycle-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/entity-lifecycle-v1.md)
- [docs/run-states-v1.md](C:/Users\Елена\Documents\DigestFlow/docs/run-states-v1.md)
- [docs/state-transitions-v1.md](C:/Users\Елена\Documents\DigestFlow/docs/state-transitions-v1.md)

This is still a planning document.

It does not change code, create migrations, or redesign UI.

The goal is to identify the smallest useful model changes that unlock the next stage of product development while keeping the current system understandable.

---

## Guiding Principle

The first implementation pass should favor:

- extending models we already have
- introducing only a very small number of new models
- keeping unstable concepts in structured fields or payloads for now

The product is still discovering its true boundaries. The schema should support that discovery rather than freeze it too early.

---

## Recommendation Summary

### New models to introduce first

Only one new first-class model is strongly justified in the first pass:

- `Source`

### Existing models to extend first

- `Topic`
- `DigestRun`

### Models or concepts that should NOT become first-class models yet

- `DigestStream`
- `RunStage`
- `RunQualityReport`
- `AudienceProfile`
- `RefinementSuggestion`
- `SourceMode` as a standalone model

This is the minimum change set that gives us better source control and better run-state clarity without dragging the whole architecture into premature normalization.

---

## 1. Which Entities Should Become Real Django Models First

## A. `Source`

### Why this one first

Source strategy is already becoming a real product concern:

- user-curated sources
- automatic vs custom vs hybrid source modes
- source validation
- source mismatch warnings
- source reuse across runs

Right now, source information is spread across:

- `Topic.source_url`
- run input snapshots
- article metadata

That works for the early prototype, but it becomes fragile once users need multiple sources and multiple source modes.

### Minimal role of the first `Source` model

The first version should be narrow:

- represent a user-provided or system-attached source
- store normalization and validation basics
- attach to a topic

It should **not** try to solve every future source problem yet.

### Minimal required fields

- `id`
- `topic` (ForeignKey to `Topic`)
- `original_url`
- `normalized_url`
- `source_type`
- `platform`
- `is_active`
- `validation_status`
- `last_validation_error` (nullable text)
- `created_at`
- `updated_at`

### Why this unlocks future UX/state improvements

It enables:

- user-curated sources flow
- multiple source entries per topic
- source validation UI
- source mismatch warnings
- clearer source failure diagnostics

### Migration risk level

Medium

Why:

- this is a new model, so risk is localized
- existing behavior can keep working while `Topic.source_url` remains as a compatibility field during rollout

### Backward compatibility concerns

- `Topic.source_url` should not disappear immediately
- first rollout can keep it as:
  - legacy single-source shortcut
  - migration bridge to first `Source`

---

## 2. Which Existing Models Should Be Extended Instead

## A. `Topic`

### Why extend instead of replace

`Topic` is already the stable product anchor.

It should remain the user-owned entry point, but it needs a little more structure to support future refinement and source strategy.

### Minimal recommended extensions

- `source_mode`
- `default_audience` or `audience_key` (simple string/enum for now)
- `quality_threshold` or `default_quality_threshold`
- `needs_refinement` (boolean or derived flag)

### Why these fields first

They let the product represent the most important user-visible configuration without introducing full new models for every concept.

### Why this unlocks future UX/state improvements

It enables:

- clearer topic setup
- source mode selection
- audience-aware defaults
- strictness control
- pre-run refinement behavior

### Migration risk level

Low to medium

Why:

- these are additive fields on an existing model
- defaults can be safe

### Backward compatibility concerns

- defaults must preserve current behavior
- existing topics should continue working without forced migration-time user input

---

## B. `DigestRun`

### Why extend instead of introducing more run models

`DigestRun` already exists and already owns lifecycle, metrics, and errors.

The first pass should make run meaning clearer before introducing more run-related models.

### Minimal recommended extensions

- normalize and expand `status` choices toward the documented state model
- add `result_message` or equivalent user-facing summary field
- add `audience_key` (optional snapshot)
- add `source_mode` (snapshot)
- add `quality_threshold_used`

If desired, one small JSON field may also help:

- `diagnostics` or `outcome_summary`

But even this should stay minimal.

### Why these fields first

They move important product meaning out of implicit metrics-only interpretation.

### Why this unlocks future UX/state improvements

It enables:

- better user-facing run results
- clearer insufficient-quality outcomes
- easier retry behavior
- cleaner mapping to documented run states

### Migration risk level

Low to medium

Why:

- additive fields and status expansion are usually manageable
- current metrics can continue to exist during transition

### Backward compatibility concerns

- old run records must remain readable
- existing code that expects current statuses will need a staged rollout
- status expansion should be aligned with UI and pipeline updates

---

## 3. Which Entities Should Explicitly NOT Become Models Yet

## A. `DigestStream`

### Why not yet

This is a real future concept, but multi-stream UX is not implemented yet.

If we add a model too early, we risk freezing the wrong stream semantics before we know:

- whether streams are long-lived
- whether they are per-topic variants
- whether they are parent/child topic objects

### Better first step

Keep stream logic conceptual and document-driven until multi-direction UX becomes real.

---

## B. `RunStage`

### Why not yet

Stage-level information still fits in structured metrics.

A model becomes worth it later only if we need:

- stage-level retry
- strong auditing
- live progress views
- separate stage ownership

### Better first step

Continue using metrics plus clearer run states.

---

## C. `RunQualityReport`

### Why not yet

Ranking diagnostics are still changing quickly.

Turning them into a dedicated model too early would create unnecessary schema churn.

### Better first step

Keep quality diagnostics in structured metrics or a run-level JSON summary.

---

## D. `AudienceProfile`

### Why not yet

Audience should probably start as a simple stored key or enum, not as a full reusable profile model.

### Better first step

Add `audience_key` to `Topic` and/or `DigestRun`.

---

## E. `RefinementSuggestion`

### Why not yet

Suggestions are derived from the current topic context and may change frequently.

### Better first step

Generate them on demand when refinement UI is introduced.

---

## F. `SourceMode` as a standalone model

### Why not yet

It is configuration, not a rich domain object yet.

### Better first step

Use a field on `Topic` and snapshot it on `DigestRun`.

---

## 4. Minimal Required Fields

This section summarizes the minimal first-pass field additions.

## New model: `Source`

- `topic`
- `original_url`
- `normalized_url`
- `source_type`
- `platform`
- `validation_status`
- `last_validation_error`
- `is_active`
- timestamps

## Extend `Topic`

- `source_mode`
- `audience_key`
- `default_quality_threshold`
- optional `needs_refinement` flag or equivalent lightweight marker

## Extend `DigestRun`

- expanded `status`
- `result_message`
- `source_mode`
- `audience_key`
- `quality_threshold_used`

That is enough for the first real model pass.

---

## 5. Why These Changes Unlock Future UX/State Improvements

### `Source`

Unlocks:

- multiple sources per topic
- source validation UX
- source mismatch warnings
- source failure clarity

### `Topic` extensions

Unlock:

- better topic setup
- audience defaults
- quality threshold defaults
- source mode UX

### `DigestRun` extensions

Unlock:

- better mapping from backend status to user-facing state
- better retry handling
- better result messages
- less dependence on raw metrics in the UI

Together, these changes support most near-term UX improvements without introducing advanced orchestration or workflow engines.

---

## 6. Migration Risk Level

## Overall recommendation

### Low-risk changes

- additive fields on `Topic`
- additive fields on `DigestRun`

### Medium-risk changes

- introducing `Source`
- moving from `Topic.source_url` to multi-source behavior

### Higher-risk changes that should wait

- introducing `DigestStream`
- introducing `RunStage`
- replacing metric-driven diagnostics with normalized reporting models

---

## 7. Backward Compatibility Concerns

### 1. `Topic.source_url`

This field should not be removed immediately.

In the first pass it can remain as:

- legacy compatibility
- default source shortcut
- migration bridge to a proper `Source` list

### 2. Existing runs

Old runs should continue to render even if they do not have:

- `audience_key`
- `source_mode`
- `quality_threshold_used`
- newer status values

### 3. Existing pipeline behavior

The first schema pass should not force a full rewrite of the pipeline.

It should allow:

- old flow to continue
- new fields to be adopted gradually

### 4. Metrics compatibility

Current metrics should remain usable during transition. We should not require immediate migration of all diagnostics into model fields.

---

## 8. Suggested Implementation Order

### Step 1. Extend `Topic`

Add:

- `source_mode`
- `audience_key`
- `default_quality_threshold`

Why first:

- lowest conceptual risk
- immediately useful for upcoming UX flows

### Step 2. Extend `DigestRun`

Add:

- clearer status set
- `result_message`
- `source_mode`
- `audience_key`
- `quality_threshold_used`

Why second:

- improves run-level product meaning quickly
- supports better result handling before deeper source architecture lands

### Step 3. Introduce `Source`

Add first minimal version of the model and attach it to `Topic`.

Why third:

- source strategy becomes a first-class product concept
- but it is slightly more invasive than adding fields

### Step 4. Keep everything else out of the schema for now

Do **not** introduce yet:

- `DigestStream`
- `RunStage`
- `RunQualityReport`
- `AudienceProfile`
- `RefinementSuggestion`
- `SourceMode` model

These should wait until product behavior proves that they are truly needed as first-class persisted objects.

---

## Implementation Notes

- This document is intentionally minimalist.
- It proposes only the first model pass, not the final architecture.
- The main goal is to unlock:
  - better source handling
  - clearer run meaning
  - cleaner future UX work

The safest path is:

1. extend the models we already trust
2. add only one carefully scoped new model
3. keep unstable concepts in fields, payloads, or diagnostics until the UX stabilizes

That keeps DigestFlow moving forward without turning planning documents into premature schema complexity.

