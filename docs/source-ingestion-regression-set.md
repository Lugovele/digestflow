# Source Ingestion Regression Set

## Goal

The current milestone is to stabilize curated source ingestion before moving on to the research and source discovery layer.

At this stage, the priority is:

- making sure manually added sources are classified correctly
- making sure usable article and feed URLs are accepted reliably
- making sure failed validations are understandable and easy to correct
- making sure the curated source flow is stable end to end

This milestone is not about building autonomous research or discovery behavior yet.

## Regression URLs

### 1. Lullaby Trust baby sleep patterns

- URL: `https://lullabytrust.org.uk/baby-safety/being-a-parent-or-caregiver/baby-sleep-patterns`
- Expected behavior: article page should be accepted as usable content, not rejected because of boilerplate/menu class names.

### 2. Johns Hopkins infant safe sleep

- URL: `https://www.hopkinsmedicine.org/health/wellness-and-prevention/infant-safe-sleep`
- Expected behavior: article page should be accepted or rejected with a clear extraction reason; the user-entered URL must remain in the input field if validation fails.

### 3. DEV.to AI tag page

- URL: `https://dev.to/t/ai`
- Expected behavior: source should be classified as a tag/listing source, not as a single article URL.

## Acceptance Criteria

- final normalized URL is visible in diagnostics
- detected source type is visible
- HTTP status or fetch failure reason is visible
- extraction strategy is visible
- content length or usable text length is visible
- rejection reason is visible when source is not accepted
- failed URL remains in the input field
- error message clears when the invalid URL is removed from the field
- validation does not use red input border for rejected article content

## Non-goals

- no autonomous research layer yet
- no crawler
- no vector database
- no AI source discovery

## Next Steps

- convert selected URLs into extraction tests or HTML fixtures
- harden boilerplate cleaning
- expose source extraction diagnostics
- verify curated source flow end-to-end
