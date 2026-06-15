# LinkedIn Post Generation Architecture

## Architectural Decision: Post Generation Quality Pipeline

PostFlow now uses a staged internal synthesis pipeline for LinkedIn post generation.

The goal is not to make every generated post perfect in this step. The goal is to stop treating final post writing as a single direct "articles -> post" prompt and instead create inspectable editorial context before the final post is written and repaired.

The current implemented flow is:

```text
source_evidence_pack
-> author_take
-> author_take repair
-> angle_decision
-> reader_problem
-> post_brief
-> writing_plan
-> final_post
-> editorial_review
-> final_post repair
-> ContentPackage.post_text
```

This flow is internal to the packaging layer. It does not change the user-facing post result UI, routes, models, migrations, source lifecycle, LinkedIn API integration, mock/real provider switching, or the final `ContentPackage` output schema.

## Current Service Boundary

`services/packaging/generator.py` remains the public packaging entry point through `generate_content_package_for_digest(...)`.

`services/packaging/post_synthesis.py` contains the staged synthesis orchestration through `run_post_synthesis_pipeline(...)`.

The split is:

```text
generator.py
  - loads digest articles;
  - builds prompt strings;
  - calls provider helpers;
  - validates and normalizes payloads;
  - creates ContentPackage debug_info;
  - saves ContentPackage.

post_synthesis.py
  - runs the staged editorial artifacts;
  - applies best-effort stage failure behavior;
  - triggers final generation;
  - runs deterministic quality checks;
  - runs editorial review;
  - triggers the existing single final repair path;
  - returns a structured synthesis result.
```

The final saved user-visible result remains `ContentPackage.post_text`.

## Implemented Stages

### source_evidence_pack

`source_evidence_pack` extracts source phrases, specific claims, mechanisms, contrasts, examples, usable terms, avoid terms, and tensions from available source material.

It is grounding material, not the post angle.

If source evidence extraction fails, the error is recorded in debug info and generation continues where safe.

### author_take

`author_take` captures the human/editorial position before angle selection.

It is intended to answer: "What is the useful point of view here?"

It must not invent first-person professional experience, metrics, examples, client claims, cases, or unsupported authority.

### author_take repair

Author take repair is deliberately narrow:

* it runs only when an `author_take` is structurally valid but rejected by deterministic quality checks;
* it runs at most once;
* it uses the same author take JSON schema;
* strict validation and quality rejection still apply after repair;
* if repair fails or the repaired take is still rejected, `author_take` is discarded.

When `author_take` is unavailable after generation/repair, downstream staged artifacts that depend on it can be skipped. In the current flow, `angle_decision` and `reader_problem` are skipped when `author_take` is unavailable so the system does not create a generic source-evidence-led editorial chain.

### angle_decision

`angle_decision` chooses the controlling angle from the accepted `author_take` and source evidence.

It also records:

* `allowed_supporting_terms`;
* `do_not_make_main_angle`;
* `angle_to_avoid`.

Validation normalizes hijack-prone supporting terms into `do_not_make_main_angle` when they were not explicitly selected by `author_take`. If a blocked term appears in `controlling_angle`, the angle decision is rejected and the failure is recorded in debug info.

### reader_problem

`reader_problem` translates the selected angle into a specific reader situation:

* target reader;
* visible behavior;
* wrong optimization;
* visible cost;
* diagnostic check;
* practical reason the reader should care.

It should make the post concrete before the brief is generated.

### post_brief

`post_brief` is the validated editorial brief.

It receives the staged context and should preserve:

* the accepted author take;
* the controlling angle;
* the reader problem;
* grounded evidence and concrete details;
* the avoid angle.

It is still a brief, not final prose.

If post brief generation or validation fails, the system uses the existing safe fallback behavior. It does not silently continue with an old direct articles-only final generation path.

### writing_plan

`writing_plan` is generated after `post_brief`.

It converts the brief into execution structure:

* opening claim;
* first three lines;
* body sequence;
* evidence to use;
* diagnostic check;
* terms to avoid;
* ending reframe.

`writing_plan` is best-effort. If it fails, the error is recorded and final generation can continue from the existing staged context.

Both initial final post generation and final post repair receive `writing_plan` when it is available.

### final_post

The final post prompt treats:

1. `writing_plan` as the primary execution plan when present;
2. `post_brief` as the validated editorial direction;
3. `author_take` as the main human perspective;
4. `source_evidence_pack` as preferred factual grounding;
5. digest article summaries as fallback factual reference.

The final post should not choose a new angle, recap articles, or let article summaries dominate the narrative.

### editorial_review

AI editorial review runs after final post generation.

It can trigger the existing single final repair attempt when deterministic checks pass but editorial review fails or scores below the configured threshold.

Editorial review does not directly cause fallback.

### final_post repair

The final repair path now receives `writing_plan` when available.

Repair uses:

* validated `post_brief`;
* `writing_plan`;
* weak payload;
* editorial review feedback;
* source facts;
* deterministic repair reasons.

When `writing_plan` is present, the repair prompt treats it as the primary execution plan for the repaired `post_text`. Repair should fix the listed quality issues without choosing a new structure, preserve the same angle and reader problem, follow the writing plan's opening/body/diagnostic/ending, and avoid `writing_plan.terms_to_avoid`.

The repair path still has exactly one attempt.

## Failure And Fallback Behavior

The architecture is best-effort where safe.

Stage failures are recorded in `debug_info`, including errors/tokens for the relevant artifact when available.

Examples:

* `source_evidence_error`;
* `author_take_error`;
* `author_take_quality_issues`;
* `author_take_repair_error`;
* `angle_decision_error`;
* `reader_problem_error`;
* `writing_plan_error`;
* `fallback_reason`.

Safe continuation is allowed for optional staged artifacts such as source evidence, author take, angle decision, reader problem, and writing plan.

Post brief failure is different: because the brief is the required editorial decision point for this architecture, brief generation/validation failure uses the existing safe fallback path rather than silently using the old direct generation flow.

Fallback behavior must not create a fake successful real package. Mock/fallback provenance remains visible internally through debug fields and existing provider/is_mock signals.

## Debugging And Manual Inspection

`run_packaging_stage` prints the internal artifacts needed to inspect quality:

```text
=== AUTHOR TAKE ===
=== AUTHOR TAKE ERROR ===
=== AUTHOR TAKE QUALITY ISSUES ===
=== AUTHOR TAKE REPAIR ATTEMPTED ===
=== AUTHOR TAKE REPAIR SUCCEEDED ===
=== AUTHOR TAKE REPAIR QUALITY ISSUES ===
=== AUTHOR TAKE REPAIR ERROR ===
=== ANGLE DECISION ===
=== ANGLE DECISION ERROR ===
=== READER PROBLEM ===
=== READER PROBLEM ERROR ===
=== POST BRIEF ===
=== WRITING PLAN ===
=== WRITING PLAN ERROR ===
=== POST BRIEF PROMPT ===
=== REPAIR PROMPT ===
```

This makes quality failures easier to locate:

* weak evidence = inspect `source_evidence_pack`;
* missing point of view = inspect `author_take`;
* source-term drift = inspect `angle_decision`;
* abstract reader value = inspect `reader_problem`;
* generic editorial direction = inspect `post_brief`;
* weak post structure = inspect `writing_plan`;
* repair drift = inspect `REPAIR PROMPT` and whether it includes the writing plan.

## Current Known Limitation

The architecture is complete enough for evaluation, but quality is not solved.

Manual digest 130 currently reaches the staged artifacts and the real provider path, and final post repair receives `writing_plan`. The repaired post can still contain abstract or corporate-sounding language. Recent digest 130 editorial review output still scored around 6 and included issues such as:

* `too_generic`;
* `weak_hook`;
* `not_enough_point_of_view`;
* `low_reader_value`.

This is acceptable for the architecture step. It means the next work should tune the quality of existing artifacts, not add another stage.

The most likely next quality target is `writing_plan` specificity:

* sharper first three lines;
* more concrete body sequence;
* stronger diagnostic check;
* less generic ending;
* stricter use of `terms_to_avoid`.

Do not claim that `writing_plan` fully prevents corporate language yet.

## MVP Boundary

Keep this quality pipeline internal/debug-only for now.

Do not change:

* UI;
* routes;
* templates;
* models;
* migrations;
* `ContentPackage` schema;
* final output JSON schema;
* ranking;
* source lifecycle;
* used-article marking;
* LinkedIn API integration;
* mock/real provider switching.

Do not add more stages until the current architecture has been evaluated against real examples.

## Runtime Prompt Files

The staged LinkedIn generation path uses these prompt files:

* `prompts/linkedin/extract_source_evidence_for_post.txt`;
* `prompts/linkedin/generate_author_take_from_evidence.txt`;
* `prompts/linkedin/repair_author_take_quality.txt`;
* `prompts/linkedin/decide_post_angle_from_evidence.txt`;
* `prompts/linkedin/define_reader_problem_from_angle.txt`;
* `prompts/linkedin/generate_post_brief_from_articles.txt`;
* `prompts/linkedin/create_writing_plan_from_context.txt`;
* `prompts/linkedin/generate_post_from_articles.txt`;
* `prompts/linkedin/review_post_editorial_quality.txt`;
* `prompts/linkedin/repair_post_quality.txt`.

Keep prompt names stable unless there is a separate, explicit cleanup decision.

## Tests

Focused tests live primarily in:

* `tests/test_packaging_articles_only.py`;
* `tests/test_prompt_usage.py`.

The normal focused verification command is:

```powershell
.\.venv\Scripts\python.exe manage.py test tests.test_packaging_articles_only tests.test_prompt_usage --verbosity 2
```

Manual inspection command:

```powershell
.\.venv\Scripts\python.exe manage.py run_packaging_stage --digest-id 130 --verbosity 2
```

Manual acceptance for this architecture step should check:

* provider reached a real path when credentials/network are available;
* author take is generated or clearly rejected with debug reason;
* angle decision and reader problem run only when author take is accepted;
* post brief is generated and validated;
* writing plan is generated when available;
* final post prompt receives writing plan;
* repair prompt receives writing plan when repair runs;
* final saved package keeps the existing output schema.

Do not accept this architecture based only on `quality_gate: pass`. The useful manual question is whether the artifacts explain the final post behavior and give a stable place for the next quality improvement.
