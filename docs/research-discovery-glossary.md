# Research Discovery Glossary

## Purpose

This document defines common Research / Source Discovery terms so future changes do not confuse similar metrics, states, or diagnostics fields.

It is intended to help with:
- Research History
- Copy full history
- diagnostics payloads
- tests
- source discovery planning and repair logic

Related workflow doc:
- [docs/research-discovery-flow.md](C:\Users\Елена\Documents\DigestFlow\docs\research-discovery-flow.md) for the end-to-end Find-click lifecycle and module responsibility overview

## Core source states

`kept`
: A discovered source the user has actively kept or selected.

`shown`
: A discovered source currently shown as a suggestion.

`shown now`
: The current visible suggestion count in the topic state. This is not necessarily the same as what the latest Find click newly produced.

`seen only`
: A source the system has encountered but is not currently shown, kept, or rejected.

`rejected`
: A discovered source rejected by filters, quality checks, user action, or duplicate/domain logic.

`new suggestion`
: A newly discovered source suggestion the user can review or act on.

`already known`
: A source the topic has already encountered, kept, shown, or otherwise recorded, so it is not counted as newly found.

`removed by user`
: A source that was previously shown or available but was explicitly removed by the user.

`active selected source`
: A source currently active for the topic.

`active selected research source`
: An active source discovered through research/source discovery.

`active selected my source`
: An active source manually provided by the user.

## Discovery run metrics

`returned`
: Raw URLs or items returned by the provider for a query or run.

`accepted`
: Candidates that passed enough checks to be considered usable internally.

`visible`
: New suggestions visible to the user from that query or run.

`visible new suggestions`
: User-actionable new source suggestions produced by a run or cycle.

`passed filtering`
: Candidates that passed source filtering.

`rejected by filters`
: Candidates rejected during filtering or quality checks.

`quality rejected`
: Candidates rejected because quality or substance signals were insufficient.

`stale rejected`
: Candidates rejected because they were too old for the current research need or recency window.

`commercial rejected`
: Candidates rejected because they looked like commercial, product, pricing, or other low-substance service pages rather than usable source material.

`known / duplicate`
: Candidates already known to the topic or normalized as duplicates.

`provider errors`
: Provider or API errors for a query or run.

Important distinctions:
- `returned` does not mean usable.
- `accepted` does not always mean visible.
- `visible` is the closest metric to what the user can act on.
- stage diagnostics may overlap and are not always an additive breakdown.

## Discovery run vs discovery cycle

`Discovery run`
: One underlying provider-backed search execution.

`Discovery cycle`
: The whole Find-click operation, which can include multiple runs.

`round`
: One step inside the discovery cycle.

`Find click`
: The user action that starts a discovery cycle.

Clarifications:
- One Find click can contain multiple underlying runs.
- The final underlying run note may not equal the full cycle result.
- Current/workspace feedback should prefer cycle totals when cycle diagnostics exist.

## Discovery cycle terms

`target visible suggestions`
: Desired visible suggestions per Find cycle. Currently 6.

`accumulated visible suggestions`
: Visible suggestions accumulated across all rounds in one cycle.

`rounds run`
: Number of rounds actually executed.

`max immediate rounds`
: Safety cap for rounds inside one Find click. Currently 3.

`decision`
: Final cycle outcome.

`stop reason`
: Why the cycle stopped if the target was not reached.

Cycle decisions:

`target_reached`
: The cycle accumulated the target number of visible suggestions.

`provider_unavailable`
: The provider could not be used in a meaningful way. This is a technical outcome, not a search-strategy judgment.

`max_rounds_reached`
: The cycle stopped at the safety cap.

`partial_target_not_reached`
: The cycle completed but did not reach the target number of visible suggestions.

`partial_target_not_reached_no_unused_surfaces`
: The cycle stopped because it did not have enough fresh search surfaces left to continue usefully.

`partial_target_not_reached_no_usable_repair_queries`
: The cycle stopped because it did not have enough usable repaired queries to continue usefully.

## Search diagnosis terms

`primary cause`
: Main reason a round or cycle underperformed.

`secondary causes`
: Additional contributing causes.

`severity`
: How severe the diagnosis is.

`explanation`
: Human-readable explanation of what happened.

`recommended next action`
: Suggested strategy response.

Diagnosis labels:

`provider_unavailable`
: No meaningful provider-backed search execution was possible.

`provider_partial_error`
: Some provider-backed work succeeded, but some queries or results failed technically.

`zero_return`
: The provider returned little or nothing usable to inspect.

`duplicate_heavy`
: Results were dominated by already-known or duplicate material.

`quality_heavy`
: Results were dominated by low-quality or low-substance material.

`stale_heavy`
: Results skewed too stale to be useful for the current research need.

`domain_repetition`
: Results repeated the same domains too heavily.

`over_narrow_query`
: Queries were likely too narrow to retrieve enough useful material.

`over_broad_query`
: Queries were likely too broad and produced weak or generic material.

`mixed_low_yield`
: The round or cycle underperformed for several reasons without one clean dominant cause.

`target_reached`
: The run or cycle reached the target, so diagnosis is effectively success/stop.

Important distinction:
- Provider errors are technical availability or processing issues.
- Duplicate-heavy and quality-heavy are search/result quality issues.
- Provider-error-only surfaces must not be treated as exhausted by themselves.

## Query repair terms

`repair plan`
: A deterministic plan for improving the next round's query strategy.

`repair plan used`
: Indicates that a later round actually consumed a repair plan.

`old query`
: The query that underperformed.

`new query`
: The compact repaired query selected for a later round.

`semantic shift`
: The type of movement from old query to new query.

`semantic shift type`
: A label such as query compression, adjacent angle shift, evidence layer shift, actor shift, or timeframe shift.

`material type`
: The kind of source or evidence being targeted, such as report, market data, research paper, or market structure analysis.

`surface key`
: A normalized representation of the search surface or angle.

`diversity reason`
: Why a particular repaired query was selected to avoid repeating the same search surface.

`action`
: The concrete repair action taken for an old query inside a repair plan, such as replacing or reshaping the query for the next round.

Clarifications:
- Repair planning was introduced before repair execution.
- Repaired queries should be compact Google/SerpAPI-friendly search formulas.
- Repaired queries should not be long natural-language intent descriptions.
- Repair plan generation and repair plan execution are separate concepts.

## Quality feedback terms

`main quality issue`
: Short summary of the dominant reason recent candidates were weak, rejected, or low-substance.

`quality guidance`
: Planner-facing guidance derived from recent weak patterns to help the next query plan avoid repeating the same low-yield behavior.

`weak domains`
: Domains that repeatedly appeared in low-quality, weak, or rejected results.

`weak material types`
: Content patterns that repeatedly appeared in weak or rejected results, such as beginner / SEO guide or price prediction / live price.

`preferred material types found`
: Higher-value material types that recent discovery did find, which can guide future planning toward stronger evidence layers.

## Search surface memory terms

`search surface`
: A recurring search direction or evidence layer, such as ETF flows, market structure, open interest, analyst report, or on-chain analysis.

`surface key`
: Stable normalized key for a surface.

`avoided surfaces`
: Recent surfaces the planner should avoid starting with.

`preferred surfaces`
: Recent surfaces that still look useful.

`underexplored surfaces`
: Promising surfaces not heavily used recently.

`exhausted`
: Surface that recently produced mostly duplicates or no useful visible suggestions.

`weak`
: Surface that recently produced low-quality or low-substance candidates.

`useful`
: Surface that recently produced visible suggestions without excessive duplicate pressure.

`unknown`
: Surface with mixed or insufficient evidence.

`provider uncertain`
: A conceptual status for provider-error-only evidence, even if it is not represented as a formal stored status.

Clarifications:
- Search surface memory is derived from recent discovery history.
- It is not a new DB-backed persistent memory model.
- Repeated Find clicks should avoid recently exhausted surfaces when possible.
- Provider-error-only surfaces should not be marked exhausted.

## Common confusion points

`shown now` vs `visible new suggestions`
- `shown now` is current topic state.
- `visible new suggestions` is what a query, run, or cycle produced.

`last discovery run note` vs `cycle total feedback`
- The last run note describes one underlying run.
- Cycle feedback describes the whole Find click.

`accepted` vs `visible`
- Accepted is internal candidate viability.
- Visible means shown to the user as a suggestion.

`duplicate_heavy` vs `provider_unavailable`
- Duplicate-heavy means search returned already-known material.
- Provider unavailable means the provider could not be used.

`surface memory` vs persistent DB memory
- Surface memory is derived from recent history, not stored as a new model.

`round` vs `Find click`
- One Find click can include multiple rounds.

`repair plan` vs `repair plan used`
- A plan can exist without being executed.
- `used` means a later round actually sent repaired queries to the provider.

`already known` vs `new suggestion`
- `already known` means the topic has already seen the source in a way that prevents it from counting as newly found.
- `new suggestion` means the source is newly surfaced and user-actionable in the current discovery result.

## Maintenance note

When new diagnostics fields, cycle decisions, repair strategies, or surface statuses are added, update this glossary so tests, Research History, and Copy full history remain understandable.
