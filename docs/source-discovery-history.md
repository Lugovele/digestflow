# DigestFlow: Source Discovery History and Repeated Source Discovery

## 1. Product model

DigestFlow should treat a Topic as a persistent research stream, not a one-time search.

The first discovery run finds initial sources for a topic. Later discovery runs should look for new sources that the topic has not already seen, rather than resurfacing the same known links again and again.

This means repeated source discovery is not just "run search again." It is "continue research for this topic with memory."

## 2. Difference between TopicSource and SourceDiscoveryHistory

`TopicSource` represents the current source workflow state shown to the user. It is the active working set in the UI:

- manual sources
- New suggestions
- kept/pinned research sources

`SourceDiscoveryHistory` represents topic memory. It should remember every URL the topic has already seen, including:

- shown URLs
- kept URLs
- removed URLs
- quality-rejected URLs
- stale URLs
- commercial URLs
- duplicate URLs
- already-known URLs

`TopicSource` is for the current workflow surface. `SourceDiscoveryHistory` is for long-lived memory.

## 3. Important rule

Discovery history should not be deleted just because the current UI state changes.

- Do not remove a URL from discovery history when the user clicks Keep.
- Do not remove a URL from discovery history when the user clicks Remove.
- Do not remove quality-rejected URLs from discovery history.

Instead, update status and decision fields on the history record.

History is topic memory, not the current UI list.

## 4. Suggested long-lived history statuses

Suggested primary long-lived statuses:

- `seen`
- `shown`
- `kept`
- `removed_by_user`
- `rejected_by_quality`

Meaning of each status:

- `seen`
  - The URL was encountered during discovery for this topic.
  - It may not have been shown to the user.
  - This is the broadest "topic has seen this URL" state.

- `shown`
  - The URL passed enough filtering to appear as a visible research suggestion.
  - A `TopicSource` row may be created from this state.

- `kept`
  - The user explicitly kept the source for future runs.
  - This is stronger than `shown` and should remain sticky across repeated discovery runs.

- `removed_by_user`
  - The URL was shown before, but the user removed it from the working set.
  - The topic still remembers it and should not treat it as brand-new later.

- `rejected_by_quality`
  - The URL was seen, but rejected by quality rules.
  - It remains in history so future runs do not keep reconsidering the same low-value source as if it were new.

## 5. Run-level outcomes

Some labels are better treated as run outcomes rather than long-lived primary status.

Suggested run outcomes:

- `new_shown`
- `already_known`
- `duplicate_url`
- `duplicate_domain`
- `previously_removed`
- `previously_rejected`
- `quality_rejected`
- `stale_rejected`
- `commercial_rejected`

How to think about these:

- `already_known`
  - The provider returned a URL that already exists in topic history.
  - This is a per-run outcome, not a new long-lived state.

- `duplicate_url`
  - The same normalized URL appeared multiple times inside one provider result set.
  - This is a per-run duplicate handling outcome.

- `duplicate_domain`
  - The same domain appears repeatedly and may need review or diversity handling.
  - This is best tracked as a run outcome or review signal, not as a primary long-lived status.

- `previously_removed`
  - The topic has already seen this URL and the user removed it before.

- `previously_rejected`
  - The topic has already seen this URL and it was previously rejected.

- `quality_rejected`
  - The current run rejected the URL for quality reasons.

- `stale_rejected`
  - The current run rejected the URL because it is outside the freshness window.

- `commercial_rejected`
  - The current run rejected the URL because commercial or promotional intent was too strong.

## 6. Lifecycle examples

### New good URL

- Provider returns a new URL for the topic.
- It passes quality and freshness checks.
- History status becomes `shown`.
- A `TopicSource` row is created for the current workflow.

### New bad URL

- Provider returns a new URL for the topic.
- It fails quality checks.
- History status becomes `rejected_by_quality`.
- No `TopicSource` row is created.

### Kept URL appears again later

- URL already exists in history with status `kept`.
- Provider returns it again in a later discovery run.
- Status remains `kept`.
- `seen_count` increments.
- `last_run_outcome` becomes `already_known`.
- It is not shown again as a brand-new suggestion.

### Removed URL appears again later

- URL already exists in history with status `removed_by_user`.
- Provider returns it again later.
- Status remains `removed_by_user`.
- It is not shown again as a new suggestion.

### Quality-rejected URL appears again later

- URL already exists in history with status `rejected_by_quality`.
- Provider returns it again later.
- Status remains `rejected_by_quality`.
- It is not shown again as a new suggestion.

## 7. SourceDiscoveryRun

Future repeated discovery should have an explicit run model.

Suggested `SourceDiscoveryRun` fields:

- `topic`
- `provider`
- `started_at`
- `completed_at`
- `status`
- `search_recency_months`
- `query_count`
- `provider_result_count`
- `known_url_count`
- `already_known_count`
- `accepted_count`
- `rejected_count`
- `new_suggestions_count`
- `excluded_domains`
- `query_angles`

Purpose of `SourceDiscoveryRun`:

- diagnostics
- run review
- future query planning
- understanding why a refresh produced new, repeated, or rejected results

This model should explain what happened during a discovery run without overloading `TopicSource`.

## 8. How history should influence future search

History should shape repeated discovery in two stages.

### Stage 1: after provider response

Do exact URL dedupe using normalized URL against:

- discovery history
- current `TopicSource`

This prevents already-known links from resurfacing as new suggestions.

### Stage 2: before provider request

Later, use history to evolve queries and reduce repetition before search runs.

Guidelines:

- Do not add every known URL to the query.
- Do not add every known domain to the query.
- Do not aggressively exclude authoritative domains.
- Limit `-site:` exclusions to repeatedly bad commercial domains.
- Keep the SerpAPI recency filter active.
- Use query angle rotation later rather than brute-force exclusions everywhere.

The goal is to make repeated discovery smarter without making it brittle or expensive.

## 9. Empty refresh behavior

If `Find new sources` returns no usable new sources:

- do not delete existing New suggestions
- keep the current suggestion set visible
- show:
  - `No new sources found. Existing suggestions were kept.`

An empty refresh should not erase useful current context.

## 10. Implementation plan

### Layer 1

`feat: track source discovery history`

- add `SourceDiscoveryRun`
- add `SourceDiscoveryHistory`
- record provider results
- record shown / kept / removed / rejected
- do exact URL known-history exclusion

### Layer 2

`refine: find new sources on refresh`

- preserve existing suggestions when no new usable sources are found
- show diagnostics for already known / rejected / new

### Layer 3

`refine: use history in source queries`

- query angle rotation
- limited bad-domain exclusions
- diagnostics for query angles

## 11. Main rules summary

- History is not deleted.
- Keep updates history to `kept`.
- Remove updates history to `removed_by_user`.
- Quality-rejected URLs remain in history.
- Already-known URLs are not shown as New suggestions.
- Duplicate checking uses history.
- Repeated `Find new sources` searches for new URLs.
- If no new usable sources are found, existing suggestions are preserved.
- Query planning should use history gradually, not by excluding every domain.
- A Topic is a persistent research stream.
