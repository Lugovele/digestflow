# Layer 8D: Structured LinkedIn Post Brief

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
