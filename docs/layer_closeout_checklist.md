# DigestFlow Layer Closeout Checklist

## Purpose

This checklist is used after completing one of DigestFlow's top-level product or architecture layers:

- pipeline
- research
- posting
- UI workflow
- account / workspace shell

Its job is simple: do not move to the next top-level layer while the current one still has unresolved architecture, product, testing, state, UX, or scope-boundary problems.

This is meant to help fast solo development stay honest. It is not approval theater.

## When to use this checklist

Use this checklist:

- [ ] after a major layer is functionally complete
- [ ] before starting the next top-level layer
- [ ] before declaring a layer MVP-ready
- [ ] after a large feature series that changed core behavior
- [ ] before pulling any post-MVP shell/account functionality into MVP scope

## Top-level layer definition

DigestFlow has five top-level layers.

### 1. Overall pipeline structure

Responsible for orchestration, flow between stages, state transitions, and pipeline-level contracts.

### 2. Research / source discovery

Responsible for search planning, providers, candidates, source history, filtering, quality evaluation, discovery diagnostics, and reviewed usable sources.

### 3. Posting / LinkedIn generation

Responsible for turning selected or reviewed sources into LinkedIn-ready content packages.

### 4. UI / user workflow

Responsible for making the core product workflow understandable and usable without developer knowledge:

`topic -> research -> source review -> LinkedIn generation -> copy/save result`

This layer focuses on the working user path, not on full SaaS account, dashboard, billing, or workspace functionality.

### 5. Account / Workspace Shell

Responsible for the product shell around the core workflow.

This may include:

- user account area
- dashboard
- topic/project library
- saved generated packages
- profile/settings
- workspace/project settings
- brand voice settings
- onboarding
- limits
- subscription/billing
- integrations
- team/workspace access

Important:

The Account / Workspace Shell is a separate layer from the MVP UI workflow. It should normally come after the core LinkedIn MVP is usable end-to-end.

Do not quietly blend full account/dashboard/billing/workspace logic into the core MVP topic workflow.

## Universal closeout sequence

### 1. Integration test

- [ ] Confirm the main happy path works end-to-end for the layer.
- [ ] Confirm the layer can receive real upstream input and produce valid downstream output.
- [ ] Confirm the layer behaves correctly when invoked through the normal product path, not only through isolated helper calls.

What "end-to-end" means by layer:

- Pipeline:
  the full orchestration path can move cleanly between stages and persist state correctly.
- Research:
  a topic can run discovery, produce reviewed usable sources, and persist discovery history/diagnostics.
- Posting:
  selected sources can produce a LinkedIn package that is grounded in source material.
- UI workflow:
  a non-developer can move through the core product path without guessing what to do next.
- Account / Workspace Shell:
  the user can find their topics, saved outputs, settings, and workspace-level actions without needing core workflow knowledge.

### 2. Targeted tests

- [ ] Run the narrowest relevant test modules first.
- [ ] Add or update tests for new behavior.
- [ ] Avoid relying only on manual checks.
- [ ] Escalate test scope only after narrow tests pass.
- [ ] Run the full suite before a major merge or release, not after every tiny edit.

### 3. Manual scenario walkthrough

- [ ] Walk through the real user or system scenario manually.
- [ ] Note where the flow feels confusing, fragile, or incomplete.
- [ ] Check whether retry, refresh, and repeated runs still make sense.
- [ ] For UI-related layers, ask: would a non-developer know what to do next?
- [ ] For account/workspace shell, ask: does the user understand where topics, generated packages, settings, and workspace actions live?

### 4. Refactor pass

- [ ] Remove duplicated logic.
- [ ] Move business/domain logic out of views where appropriate.
- [ ] Keep orchestration, domain services, templates, and tests clearly separated.
- [ ] Rename unclear variables, functions, and classes.
- [ ] Remove temporary hacks introduced during development.
- [ ] Keep account/workspace shell concerns separate from core research/posting domain services.

### 5. Code review

#### Technical review

- [ ] Responsibilities are clear.
- [ ] Functions are not doing too much.
- [ ] Errors are handled explicitly.
- [ ] There are no silent failures.
- [ ] Coupling is justified and limited.
- [ ] The layer can be tested in isolation where appropriate.
- [ ] Account/workspace shell is not becoming a dumping ground for unrelated product logic.

#### Product review

- [ ] The layer solves the intended MVP or post-MVP problem.
- [ ] No post-MVP scope leaked into MVP layers.
- [ ] User value actually increased.
- [ ] The next layer was not accidentally implemented inside the current one.
- [ ] Account/workspace functionality is not blocking the core LinkedIn MVP unless explicitly required.

### 6. Boundary check

Explicitly answer:

- [ ] What are this layer's inputs?
- [ ] What are this layer's outputs?
- [ ] What should this layer not know?
- [ ] Which responsibilities belong to the next layer?
- [ ] Did this layer accidentally absorb work from another layer?

Examples:

- Research should not generate LinkedIn posts.
- Posting should not decide search strategy.
- UI workflow should not contain core business logic.
- Pipeline should orchestrate, not own all domain decisions.
- Account/workspace shell should not redefine the core `topic -> research -> posting` workflow.
- Core MVP UI should not become a full personal account/dashboard layer unless explicitly scoped.

### 7. Data and state review

- [ ] Models and fields are still necessary.
- [ ] State transitions are clear.
- [ ] Duplicated state is not spread across multiple places without reason.
- [ ] Migrations are appropriate.
- [ ] Stored history is sufficient to debug behavior.
- [ ] Important user decisions survive refresh and retry.
- [ ] Repeated runs are safe.
- [ ] Account/workspace-level state is clearly separated from topic/research/posting state.

For account/workspace shell:

- [ ] Topic ownership is clear.
- [ ] Saved package ownership is clear.
- [ ] Settings scope is explicit: user-level, workspace-level, topic-level, or package-level.
- [ ] Settings inheritance is understandable.
- [ ] Future billing and limits can be added without rewriting the core workflow.

### 8. Naming and vocabulary cleanup

- [ ] Review product and code terminology.
- [ ] Check that terms are used consistently across models, services, views, templates, and UI copy.
- [ ] If a term is ambiguous, document the chosen meaning.

DigestFlow vocabulary to review:

- source
- article
- candidate
- suggestion
- discovered source
- reviewed source
- pinned source
- checked source
- used article
- known source
- duplicate
- stale
- low-quality
- discovery outcome
- query repair
- pivot
- topic
- project
- workspace
- account
- dashboard
- saved package
- brand voice
- integration

Important distinctions to keep clear:

- Topic workspace:
  the working area for one content topic.
- UI workflow:
  the screens needed to complete the core LinkedIn MVP flow.
- Account / Workspace Shell:
  the broader product area for managing topics, saved outputs, settings, integrations, limits, and future SaaS features.

### 9. Error and empty state review

- [ ] No results are found.
- [ ] External provider fails.
- [ ] Provider returns partial results.
- [ ] All results are duplicates.
- [ ] Results are stale or low-quality.
- [ ] User selects nothing.
- [ ] Generation fails.
- [ ] API key/config is missing.
- [ ] The user refreshes or retries.
- [ ] A repeated run happens after prior history exists.
- [ ] User has no topics yet.
- [ ] User has no saved generated packages yet.
- [ ] Account/workspace settings are incomplete or missing.
- [ ] Integration settings are not configured.

### 10. Diagnostics and observability review

- [ ] The system records enough information to understand what happened.
- [ ] Diagnostics are useful but not overwhelming.
- [ ] Developer-level diagnostics are not confused with user-facing UX.
- [ ] The user sees clear messages when action is needed.
- [ ] Internal details are not exposed unnecessarily.
- [ ] Account/workspace shell shows status clearly without leaking implementation internals.

### 11. Performance and cost check

- [ ] External provider/API calls per user action are understood.
- [ ] Repeated or unnecessary calls are removed.
- [ ] Query and prompt length are sane.
- [ ] Database queries for large histories are acceptable.
- [ ] Behavior with many topics or many saved sources remains usable.
- [ ] Behavior with many saved generated packages remains usable.
- [ ] Dashboard/account pages remain usable with many records.
- [ ] Pages remain usable with hundreds of source records.

### 12. Security and privacy sanity check

- [ ] API keys are not logged or displayed.
- [ ] User input is safely rendered.
- [ ] External URLs, titles, and snippets do not create unsafe HTML.
- [ ] Error messages do not expose secrets or internal implementation details.
- [ ] Stored content is appropriate for the product.
- [ ] Account/workspace pages do not expose another user's topics, sources, packages, or settings.
- [ ] Future integrations and credentials have a clear safe storage boundary.

### 13. Documentation closeout note

For each completed layer, create or update a short note with:

- [ ] layer name
- [ ] status
- [ ] what was completed
- [ ] what is intentionally out of scope
- [ ] main files/services/models involved
- [ ] main tests
- [ ] known limitations
- [ ] post-MVP parking lot items
- [ ] next recommended layer

### 14. Parking lot update

- [ ] Move good but out-of-scope ideas into a parking lot instead of implementing them immediately.

Examples:

- YouTube as source input
- YouTube as separate output channel
- automatic multi-round cycle runner until at least 6 usable sources
- deep personalization
- advanced analytics
- advanced UI design system
- full personal account area
- billing/subscriptions
- team workspaces
- external publishing integrations
- advanced onboarding

### 15. Next layer entry criteria

Before starting the next top-level layer, define:

- [ ] what input it receives from the completed layer
- [ ] what assumptions it can rely on
- [ ] what it should not re-solve
- [ ] which known limitations it must tolerate

For account/workspace shell, explicitly decide whether it is:

- [ ] included in MVP scope, or
- [ ] deferred until after the core LinkedIn MVP flow is usable end-to-end

## Layer-specific closeout notes

### Pipeline layer closeout

- [ ] Orchestration boundaries are clear.
- [ ] Step contracts are explicit.
- [ ] State transitions are understandable.
- [ ] Partial failure handling is deliberate.
- [ ] Retry behavior is sane.
- [ ] Refresh behavior does not corrupt state.
- [ ] Handoff to research, posting, and UI is clear.
- [ ] The layer does not depend unnecessarily on account/workspace shell concerns.

### Research layer closeout

- [ ] Search query planning is stable enough for MVP.
- [ ] Provider boundary is clear.
- [ ] Source candidate conversion is understandable.
- [ ] Source history is persisted and useful.
- [ ] Known, duplicate, stale, and low-quality filtering behave correctly.
- [ ] Quality-aware discovery works.
- [ ] Discovery outcomes are explainable.
- [ ] Query performance history is useful.
- [ ] Query repair/pivot behavior is diagnosed and understandable.
- [ ] SerpAPI query length remains sane.
- [ ] Manual checks were performed on several real topics.
- [ ] There is a clear freeze line before moving to posting/UI.

Important:

After the MVP-ready research layer is closed, do not keep adding discovery intelligence unless it directly blocks the LinkedIn MVP user flow.

### Posting / LinkedIn generation layer closeout

- [ ] Selected/reviewed sources are used as input.
- [ ] Generated output does not invent sources.
- [ ] Generated output can be traced back to source material.
- [ ] Output includes the required MVP LinkedIn package fields.
- [ ] Empty selection behavior is clear.
- [ ] Failed generation behavior is clear.
- [ ] Regenerate/edit/copy/save behavior works, if included.
- [ ] Output quality was reviewed on several real topics.
- [ ] Generated package can later connect to account/workspace saved history without changing generation logic.

Minimum MVP LinkedIn package:

- [ ] main post
- [ ] hook options
- [ ] CTA
- [ ] hashtags
- [ ] source notes or article summary

### UI workflow layer closeout

- [ ] The full core user path is understandable.
- [ ] There are no debug-wall pages pretending to be product UI.
- [ ] Source review is visually scannable.
- [ ] Selected state is preserved.
- [ ] Buttons and actions are clear.
- [ ] Back/next navigation works.
- [ ] Empty/loading/error states are clear.
- [ ] Generated LinkedIn package is easy to copy and use.
- [ ] Terminology is consistent.
- [ ] Layout remains usable on narrower screens.
- [ ] The UI workflow is not overloaded with full account/dashboard responsibilities.

Important:

The MVP UI workflow is the working path inside a topic. It is not the full personal account or SaaS workspace shell.

### Account / Workspace Shell layer closeout

- [ ] Dashboard structure is clear.
- [ ] Topic/project library is usable.
- [ ] Saved generated packages are accessible.
- [ ] Account/profile settings are understandable.
- [ ] Workspace/project settings are understandable.
- [ ] Brand voice settings are scoped correctly, if included.
- [ ] Integration settings are scoped correctly, if included.
- [ ] Limits/subscription/billing placeholders are coherent, if included.
- [ ] Onboarding is coherent, if included.
- [ ] Core pipeline/research/posting logic remains separate.
- [ ] Ownership and access boundaries are explicit.
- [ ] Empty states for new users are clear.
- [ ] The shell scales reasonably with many topics and saved packages.
- [ ] Future extensibility for YouTube, integrations, billing, and team workspaces remains possible.

Minimum account/workspace shell may include:

- [ ] topic dashboard
- [ ] saved LinkedIn packages
- [ ] basic settings
- [ ] basic brand voice/preferences

Full account/workspace shell may include:

- [ ] auth
- [ ] profile
- [ ] workspace
- [ ] billing
- [ ] limits
- [ ] integrations
- [ ] content history
- [ ] brand settings
- [ ] onboarding
- [ ] team access

Important:

This layer should normally be implemented after the core LinkedIn MVP is usable end-to-end, unless it is explicitly moved into MVP scope.

## Final layer closeout decision

Use this template when closing a layer:

```text
Layer:
Date:
Status:
Closed for MVP: yes/no
Included in MVP scope: yes/no
Remaining blockers:
Known limitations accepted for MVP:
Post-MVP items moved to parking lot:
Next layer:
Reviewer notes:
```

