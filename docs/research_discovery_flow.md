# Research Discovery Flow

## Purpose

This document describes the Research / Source Discovery flow end to end: how one Find click turns into provider-backed source discovery, diagnostics, repair planning, bounded rounds, and Research History audit output.

Use this document as a workflow and responsibility map.

Use related docs for adjacent detail:
- [docs/research_discovery_glossary.md](C:\Users\Елена\Documents\DigestFlow\docs\research_discovery_glossary.md) for terminology
- [docs/source_discovery_history.md](C:\Users\Елена\Documents\DigestFlow\docs\source_discovery_history.md) for source-history behavior
- [docs/research_layer_followups.md](C:\Users\Елена\Documents\DigestFlow\docs\research_layer_followups.md) for non-blocking maintainability follow-ups

This document does not replace those.

## Related docs

- [docs/research_discovery_glossary.md](C:\Users\Елена\Documents\DigestFlow\docs\research_discovery_glossary.md) - terminology and diagnostics field definitions
- [docs/source_discovery_history.md](C:\Users\Елена\Documents\DigestFlow\docs\source_discovery_history.md) - source history model and source-state behavior
- [docs/research_layer_followups.md](C:\Users\Елена\Documents\DigestFlow\docs\research_layer_followups.md) - non-blocking maintainability follow-ups for this layer
- [docs/layer_closeout_checklist.md](C:\Users\Елена\Documents\DigestFlow\docs\layer_closeout_checklist.md) - general checklist used to close product layers

## 1. Find click lifecycle

When the user clicks `Find` or `Find New Sources` for a topic, the system starts a discovery cycle.

One discovery cycle can contain multiple underlying provider-backed discovery runs, referred to as rounds.

Current cycle rules:
- target visible usable suggestions: `6`
- max immediate rounds: `3`

The cycle stops when one of these happens:
- the target is reached
- the provider is unavailable
- the round cap is reached
- the system has no useful repair/surface path left for another safe round

The important product boundary is that one user click can span multiple rounds, but it is still one bounded discovery attempt.

## 2. Round 1 planning

Round 1 uses the normal history-aware query planning path.

That planning uses:
- topic context
- recent query history
- source quality feedback
- recent search-surface outcomes

Round 1 should not blindly restart from recently exhausted directions. When recent history is available, it should prefer useful or underexplored surfaces where possible.

Provider-error-only surfaces should not be treated as exhausted just because a prior provider request failed.

Relevant modules:
- [services/sources/content_research_planner.py](C:\Users\Елена\Documents\DigestFlow\services\sources\content_research_planner.py)
- [services/sources/query_history_summary.py](C:\Users\Елена\Documents\DigestFlow\services\sources\query_history_summary.py)

## 3. Query history summary

Query history summary collects recent query performance and quality patterns and feeds them back into planning.

It helps the planner:
- avoid repeating weak directions
- avoid reopening exhausted directions too early
- preserve useful directions when they still have room
- incorporate source quality guidance into later query planning

It also includes search surface memory.

Important boundary:
- `query_history_summary["search_surface_memory"]` is derived from recent history
- it is not a new DB model

For exact term meanings, use the glossary rather than this document.

## 4. Search surface memory

Search surfaces are recurring search directions or evidence layers.

Examples include:
- ETF flows
- institutional flows
- funding rates
- open interest
- market structure
- analyst report
- on-chain analysis

Recent outcomes classify surfaces into compact planning buckets such as:
- avoided
- preferred
- underexplored
- exhausted
- weak
- useful
- unknown

This lets repeated Find clicks avoid starting over from recently exhausted directions when better adjacent surfaces exist.

## 5. Provider search and source filtering

Each round sends queries to the provider and receives raw URLs or items back.

From there, the system:
- normalizes URLs
- filters candidates
- detects duplicates and already-known results
- rejects low-quality or stale material
- preserves visible suggestions that are usable

Provider errors are tracked separately from source-quality failure.

This matters because partial provider failure can still preserve useful suggestions from successful queries. A technical failure in one query should not automatically be interpreted as weak topic evidence.

For deeper source-history and repeated-source behavior, see [docs/source_discovery_history.md](C:\Users\Елена\Documents\DigestFlow\docs\source_discovery_history.md).

## 6. Diagnosis

After each round, the system diagnoses why that round did or did not produce enough visible suggestions.

The diagnosis distinguishes:
- technical provider issues
- duplicate-heavy results
- quality-heavy results
- stale-heavy results
- narrow/broad query problems
- mixed low-yield cases

Important rule:
- `provider_unavailable` is a technical outcome
- duplicate-heavy, quality-heavy, and stale-heavy are search/result-quality signals

Relevant module:
- [services/sources/discovery_diagnostics.py](C:\Users\Елена\Documents\DigestFlow\services\sources\discovery_diagnostics.py)

## 7. Repair planning

If a round underperforms, the system builds a deterministic repair plan for the next possible round.

A repair plan proposes compact, search-grade replacement queries. It should not just retry failed queries verbatim.

Depending on diagnosis, repair can shift:
- to an adjacent search surface
- to a stronger material type
- to a different evidence layer
- to a different actor/entity angle
- to a different timeframe

Repair planning also deduplicates repaired queries and avoids repeating the same search surface inside one cycle when better alternatives exist.

Relevant module:
- [services/sources/discovery_repair.py](C:\Users\Елена\Documents\DigestFlow\services\sources\discovery_repair.py)

## 8. Repair application

A repair plan can exist without being used.

`Repair plan used` means a later round actually executed repaired queries.

Current flow:
- Round 2 can consume the repair plan from Round 1
- Round 3 can consume the repair plan from Round 2 if the cycle is still below target and safety rules still allow another round

This separation matters for debugging:
- repair generation explains what the system planned
- repair usage explains what the provider actually received in a later round

## 9. Third repaired round

The system can run up to 3 immediate rounds in one Find click.

The third round is not automatic.

It runs only if:
- accumulated visible suggestions are still below `6`
- the provider is not unavailable
- a usable repair plan exists
- selected repaired queries are still fresh within the current click
- the cycle has not hit the round cap

Empty or no-evidence cases should not burn a third round unnecessarily.

The goal is bounded target-seeking, not an open-ended retry loop.

## 10. Stop conditions

Current cycle stop decisions include:
- `target_reached`
- `provider_unavailable`
- `max_rounds_reached`
- `partial_target_not_reached`
- `partial_target_not_reached_no_unused_surfaces`
- `partial_target_not_reached_no_usable_repair_queries`

These decisions are stored in discovery cycle diagnostics so partial outcomes are understandable instead of silent.

Use the glossary for exact definitions.

## 11. Research History / Copy full history

Research History is the audit surface for this layer.

It should show:
- Current research state
- Query performance
- Source quality feedback
- Search surface memory
- Seen sources
- Discovery runs
- Discovery cycle
- Search diagnosis
- Strategy repair
- Repair plan used

`Copy full history` exposes a text version of the same audit trail for debugging and review.

Current/workspace feedback should prefer cycle totals over the last underlying run note when cycle diagnostics exist, because one Find click may have required multiple rounds.

Relevant module:
- [services/sources/research_history_presenter.py](C:\Users\Елена\Documents\DigestFlow\services\sources\research_history_presenter.py)

## Module responsibility summary

- [apps/digests/views.py](C:\Users\Елена\Documents\DigestFlow\apps\digests\views.py)
  - HTTP/controller glue, request handling, queryset filtering, template rendering, and the remaining cycle-runner execution path

- [services/sources/content_research_planner.py](C:\Users\Елена\Documents\DigestFlow\services\sources\content_research_planner.py)
  - builds research query plans

- [services/sources/query_history_summary.py](C:\Users\Елена\Documents\DigestFlow\services\sources\query_history_summary.py)
  - builds recent query and search-surface history summary

- [services/sources/discovery_diagnostics.py](C:\Users\Елена\Documents\DigestFlow\services\sources\discovery_diagnostics.py)
  - builds diagnosis and discovery-cycle diagnostic payloads

- [services/sources/discovery_repair.py](C:\Users\Елена\Documents\DigestFlow\services\sources\discovery_repair.py)
  - builds and selects repair queries and repair plans

- [services/sources/research_history_presenter.py](C:\Users\Елена\Documents\DigestFlow\services\sources\research_history_presenter.py)
  - formats Research History and Copy full history presentation data

- [services/sources/discovery_constants.py](C:\Users\Елена\Documents\DigestFlow\services\sources\discovery_constants.py)
  - shared discovery constants and decision strings

## Non-goals / guardrails

- No unlimited search loop
- No more than 3 immediate rounds currently
- Provider unavailable is not treated as weak content
- Source filters and duplicate filters stay separate from query planning and repair
- Search surface memory is derived from recent history, not a new DB-backed memory model
- The cycle runner/provider execution path should be refactored only carefully because it touches provider calls and DB writes

## Maintenance note

When discovery decisions, diagnosis causes, repair strategies, diagnostics payloads, or module responsibilities change, update this document and [docs/research_discovery_glossary.md](C:\Users\Елена\Documents\DigestFlow\docs\research_discovery_glossary.md).
