# Layer 8D: Structured LinkedIn Post Brief

## Architectural Decision: Post Generation Quality Pipeline

PostFlow is moving from a DigestFlow-style pipeline toward a LinkedIn post generation product.

`GeneratePostService` is the orchestration layer. It coordinates source sufficiency, research when needed, top-source selection, packaging generation, saving the result, and marking selected sources as used.

The quality of the final LinkedIn post must not be handled by a single "articles -> post" prompt.

Direct generation from selected research articles into a final LinkedIn post has a structural failure mode:

* the model summarizes the articles instead of forming a point of view;
* the output sounds like a digest or generic AI post;
* the post lacks one clear thesis;
* there is no explicit author take;
* research becomes the main subject instead of supporting evidence;
* the post often tries to cover too much;
* the output may mention "articles", "research", or trends mechanically.

Therefore, the generation layer must not do:

```text
3 selected articles
-> final post
```

Instead, PostFlow will use a dedicated synthesis pipeline inside the post generation/packaging layer:

```text
selected_articles
-> EvidencePack
-> PostBrief
-> FinalPost
-> ContentPackage.post_text
```

This is inserted exactly where selected articles are currently converted into `post_text`. It does not replace the research pipeline or `GeneratePostService`.

### Responsibility Split

```text
GeneratePostService
  - checks source sufficiency;
  - triggers research if needed;
  - selects top 3 usable sources;
  - creates/updates ContentPackage;
  - calls PostSynthesisPipeline;
  - marks used sources.

PostSynthesisPipeline
  - extracts evidence from selected articles;
  - creates an editorial post brief;
  - writes the final LinkedIn post;
  - optionally runs quality checks / rewrite pass;
  - returns artifacts for storage/debugging.
```

`GeneratePostService` remains responsible for orchestration.
`PostSynthesisPipeline` is responsible for post quality.

### Intermediate Artifacts

#### EvidencePack

Purpose: extract useful argumentative material from selected articles.

EvidencePack is not a summary. It should contain:

* key claim per article;
* specific supporting data;
* implication for the target professional reader;
* tension with common assumptions;
* cross-article patterns;
* contradictions, if real;
* strongest evidence.

#### PostBrief

Purpose: make the editorial decision before writing.

This is the most important quality artifact. It should contain:

* one chosen angle;
* one core thesis;
* tension;
* author take;
* target reader;
* evidence to use;
* evidence to ignore;
* hook direction;
* intended post structure.

The PostBrief must force focus and explicitly decide what not to include.

#### FinalPost

Purpose: produce the final LinkedIn post from the PostBrief.

The final post should:

* have one clear thesis;
* sound like a human expert, not a summary;
* use research as support, not as the subject;
* avoid generic AI openings;
* avoid "the article says" framing;
* avoid generic CTA endings;
* preserve the chosen angle and author take.

### MVP Implementation

For MVP, use three required steps:

```text
1. EvidencePack
2. PostBrief
3. FinalPost
```

A later optional step may be added:

```text
4. Critique / rewrite
```

The rewrite step should not be mandatory until the 3-step pipeline has been verified against real examples.

### Storage And Debugging

The final user-visible result remains:

```text
ContentPackage.post_text
```

Generation artifacts should be available for debugging and quality iteration. Recommended future fields on `ContentPackage`:

```python
evidence_pack = models.JSONField(null=True, blank=True)
post_brief = models.JSONField(null=True, blank=True)
draft_post = models.TextField(null=True, blank=True)  # optional, if rewrite step exists
generation_meta = models.JSONField(null=True, blank=True)
```

The user should only see the final post and hashtags.

The developer should be able to inspect:

```text
EvidencePack -> PostBrief -> FinalPost
```

This makes weak-output diagnosis clearer:

* bad EvidencePack = extraction problem;
* bad PostBrief = angle/thesis problem;
* good PostBrief but bad post = writing prompt problem;
* good draft but bad final = rewrite problem.

### Integration Point

Insert the synthesis pipeline where the current packaging logic turns selected articles into `post_text`.

Do not rewrite the research pipeline for this.
Do not make `GeneratePostService` responsible for writing quality.

Expected integration:

```python
selected_sources = select_top_3_sources(topic)

result = PostSynthesisPipeline.run(
    topic=topic,
    articles=[source.article for source in selected_sources],
)

package.evidence_pack = result.evidence_pack
package.post_brief = result.post_brief
package.post_text = result.final_post
package.generation_meta = result.meta
package.status = "done"
package.save()
```

### Naming

Use these names in documentation and future code unless the existing codebase strongly suggests another convention:

* `GeneratePostService` for orchestration;
* `PostSynthesisPipeline` for the quality layer;
* `EvidencePack` for extracted research evidence;
* `PostBrief` for editorial angle/thesis/author take;
* `FinalPost` for the final LinkedIn post.

Existing prompt files may be adapted to this structure:

* `generate_author_take_from_evidence.txt`
* `generate_post_brief_from_articles.txt`
* `generate_post_from_articles.txt`

The conceptual flow should become:

```text
generate_evidence_pack_from_articles
generate_post_brief_from_evidence
generate_post_from_brief
```

Do not rename prompt files yet unless it is a small, safe change.

### What Not To Do Now

For MVP:

* do not build a full personalization/voice profile yet;
* do not add user-facing angle selection yet;
* do not rewrite the research pipeline;
* do not create a complex multi-agent system;
* do not make 4+ LLM calls mandatory from day one;
* do not treat prompt wording as the only quality fix;
* do not let final post generation read raw articles and decide everything in one step.

### Tests And Quality Checks

Validate quality with golden examples rather than exact string tests.

Basic checks:

* final post must not start with "In today's...";
* final post must not mention "the article says" or "research suggests" mechanically;
* final post must have one clear thesis;
* final post must preserve the PostBrief angle;
* final post must include interpretation, not just facts;
* final post should use evidence invisibly as support.

## Staged Editorial-Context Hypothesis

Prompt-only tightening of the existing flow improved some constraints, but manual checks showed that generic LinkedIn output can still happen.

The system no longer generates directly from article summaries, but too much editorial reasoning is still concentrated in a few prompts. `author_take` can identify a useful human perspective, but `post_brief` may dilute it into broad personal-branding language. Final generation can also become generic when the writing context is incomplete, too abstract, or still dominated by compressed article summaries.

Adding more constraints to the final prompt or brief prompt increases prompt complexity, but it does not guarantee context preservation. A single large `EditorialContext` would likely move the overload earlier rather than solve it.

The next architecture hypothesis is to split editorial reasoning into smaller staged artifacts. Instead of:

```text
articles
-> big EditorialContext
-> final_post
```

Prefer:

```text
source_evidence_pack
-> author_take
-> angle_decision
-> reader_problem
-> writing_plan
-> final_post
```

This staged pipeline is a proposed direction, not a claim that the full flow exists in production today.

### Staged Artifact Responsibilities

`source_evidence_pack`

Extracts source facts, mechanisms, contrasts, examples, useful terms, and risky generic source language. It should not choose the final post angle.

`author_take`

Creates the human/editorial perspective. It should not produce the full writing plan.

`angle_decision`

Chooses one controlling angle from `author_take` and evidence. It should explicitly say which source terms or adjacent topics must not become the main angle. For example, `Brand Lag` may support an angle, but should not dominate unless it is central to `author_take`.

`reader_problem`

Defines one concrete reader situation, wrong optimization, visible cost, and practical diagnostic/check. It must not change the selected angle.

`writing_plan`

Creates a compact post structure: opening claim, body sequence, evidence to use, terms to avoid, and ending reframe. It must not write final prose.

`final_post`

Writes the LinkedIn post from the `writing_plan`. It should not choose a new angle, summarize articles, or solve upstream reasoning again.

### Staging Principle

Each stage should do one simple job:

* evidence = what exists in sources;
* angle_decision = what the post is really about;
* reader_problem = why the reader should care;
* writing_plan = how the post should unfold;
* final_post = execution only.

Later stages should receive only the context needed for their task. Full article summaries should not continue to dominate late-stage prompts unless needed for factual verification.

### Debugging Benefit

This staged design makes quality failures easier to diagnose:

* bad angle = inspect `angle_decision`;
* generic reader problem = inspect `reader_problem`;
* weak structure = inspect `writing_plan`;
* weak prose = inspect `final_post`;
* drift during repair = inspect whether repair received the correct controlling artifacts.

### MVP Boundary

This is a proposed next architecture direction. Do not claim the full staged pipeline exists in production until it is implemented.

Within the MVP boundary:

* do not change UI, routes, templates, ranking, source lifecycle, used-article marking, LinkedIn API, mock/real switching, models, migrations, or output schema;
* keep the current artifacts internal/debug-only unless explicitly decided otherwise;
* evaluate the smallest safe implementation step before adding more stages.

The next review should evaluate whether the smallest safe implementation step is an `angle_decision` artifact before `post_brief`, because the observed failure mode is angle drift and source-term dominance.

## A. Goal

Layer 8D adds one internal editorial brief step before final LinkedIn post generation so the system chooses the reader, angle, claim, evidence, takeaway, and ending before writing the final post.

The quality gate remains after final post generation.

`quality_gate: pass` is not enough to accept 8D.

8D succeeds only when the generated `POST BRIEF` and final `post_text` are visibly aligned:

* first line reflects `sharp_claim` or `tension`;
* body uses `evidence_points`;
* ending reflects `ending_reframe`;
* post does not invent a different angle;
* post does not collapse into generic advice.

## B. Corrected Flow

```text
digest.get_articles()
-> generate structured post brief
-> validate post brief
-> generate final post using digest articles + post brief
-> normalize payload
-> validate ContentPackage schema
-> quality gate
-> one repair retry if needed
-> save ContentPackage
```

There must be no silent fallback to the old direct "articles only -> final post" path.

## C. Brief Schema And Validation

```json
{
  "target_reader": "string",
  "reader_pain_or_mistake": "string",
  "sharp_claim": "string",
  "tension": "string",
  "evidence_points": ["string", "string"],
  "practical_takeaway": "string",
  "ending_reframe": "string",
  "suggested_hook_direction": "string",
  "avoid_angle": "string"
}
```

Validation requirements:

* all fields required;
* all string fields must be non-empty;
* `evidence_points` must be a list of at least 2 and preferably 2-4 non-empty concise strings;
* evidence points must be grounded in article summaries/key points;
* `avoid_angle` must name the generic angle to avoid;
* the brief should be an editorial decision, not an article summary.

## D. Failure Behavior

If brief generation fails because of provider error, invalid JSON, missing fields, invalid shape, or too few evidence points:

* use the existing safe fallback path;
* do not continue with old direct final-post generation;
* do not save a fake successful package;
* make the fallback reason visible in `debug_info`.

## E. PackagingGenerationResult Changes

Optional fields:

```python
post_brief: dict[str, Any] | None = None
post_brief_prompt: str = ""
```

All construction sites must be checked:

* real OpenAI path;
* mock path;
* fallback path;
* test patched return values;
* any direct `PackagingGenerationResult(...)` construction in tests.

## F. Final Prompt Requirements

The final prompt must obey the brief, not merely include it.

Required constraints:

* do not choose a new angle;
* do not broaden the post beyond the brief;
* if the articles contain more material, ignore what does not serve the brief;
* first line should reflect `sharp_claim` or `tension`;
* body should use `evidence_points`;
* ending should reflect `ending_reframe`;
* source articles are grounding material, not permission to expand into a broad essay;
* keep existing output JSON shape and length rules;
* keep existing no-extra-keys and no-`carousel_outline` instruction.

## G. Debug / Manual Verification Requirements

`run_packaging_stage.py` should print:

```text
=== POST BRIEF ===
...
```

Optionally:

```text
=== POST BRIEF PROMPT ===
...
```

Manual verification must inspect two levels.

POST BRIEF:

* strong `sharp_claim`;
* concrete `reader_pain_or_mistake`;
* useful `practical_takeaway`;
* useful `ending_reframe`;
* specific `avoid_angle`;
* reads like an editorial decision, not a summary.

FINAL POST:

* follows the brief;
* first line reflects `sharp_claim` or `tension`;
* body uses `evidence_points` without article recap;
* ending reflects `ending_reframe`;
* does not drift into a new generic angle;
* does not collapse into generic advice.

## H. Manual Success Criteria

```powershell
.\.venv\Scripts\python.exe manage.py run_packaging_stage --digest-id 130
```

Accept manual verification only if:

* `provider: openai`;
* `is_mock: False`;
* output shows `POST BRIEF`;
* `quality_gate` or `repair_quality_gate` passes;
* `carousel_outline_count` remains `0`;
* the final post visibly follows the generated brief.

If `quality_gate` passes but the post ignores the brief, manual verification fails.

## I. Files Likely Involved

* `services/packaging/generator.py`
* `prompts/linkedin/generate_post_brief_from_articles.txt`
* `prompts/linkedin/generate_post_from_articles.txt`
* `tests/test_packaging_articles_only.py`
* `tests/test_prompt_usage.py`
* `apps/packaging/management/commands/run_packaging_stage.py`

## J. Tests To Add / Update

Prompt tests:

* brief prompt file exists and includes required fields;
* rendered brief prompt includes author profile values and article evidence;
* final post prompt includes brief values;
* final post prompt explicitly says not to choose a new angle or broaden beyond the brief.

Packaging tests:

* valid brief is generated before final post generation;
* final post prompt receives/includes the validated brief;
* invalid brief falls back safely;
* brief failure does not call old direct final-post generation;
* fallback reason is visible in `debug_info`;
* `debug_info` includes `post_brief` on success;
* `PackagingGenerationResult` defaults keep existing patched tests stable;
* final post still strips unknown model keys;
* existing quality gate and one repair retry still work.

Focused test command:

```powershell
.\.venv\Scripts\python.exe manage.py test tests.test_packaging_articles_only tests.test_prompt_usage --verbosity 2
```

## K. Risks / What Not To Change Yet

* No UI changes.
* No route changes.
* No model/schema migration.
* No `ContentPackage` schema changes.
* No source lifecycle changes.
* No LinkedIn API changes.
* No source discovery/search.
* No extra repair attempts.
* Do not remove the existing quality gate.
* Keep the brief internal/debug-only for MVP.
* Do not add durable persistence unless explicitly decided later.
* Do not accept 8D based on `quality_gate: pass` alone; acceptance requires visible brief-to-post alignment.
