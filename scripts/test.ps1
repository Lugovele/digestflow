param(
    [Parameter(Position = 0)]
    [string]$Level,

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$python = ".\.venv\Scripts\python.exe"
$manage = "manage.py"

$commandInfo = [ordered]@{
    "help" = @{
        Purpose = "Show available test commands."
        Scope = "No tests; usage and workflow guidance."
        When = "Use when choosing the narrowest useful test scope."
    }
    "single" = @{
        Purpose = "Run one exact regression test."
        Scope = "One Django test path."
        When = "Use for the exact behavior currently being changed."
    }
    "saved-sources" = @{
        Purpose = "Run the saved-source form and persistence loop."
        Scope = "Saved-source add, duplicate, ordering, and validation behavior."
        When = "Use for saved-source form/view work before expanding to broader source coverage."
    }
    "source-ingestion" = @{
        Purpose = "Run source detection, extraction, and saved-source acceptance checks."
        Scope = "RSS adapter ingestion plus saved-source article acceptance and rejection paths."
        When = "Use for fetch, redirect, detection, extraction, and acceptance logic."
    }
    "sources" = @{
        Purpose = "Run the broader source feature area."
        Scope = "All current source ingestion and source workflow tests."
        When = "Use when a source change spans ingestion, saved sources, and review flow together."
    }
    "ranking" = @{
        Purpose = "Run ranking behavior tests."
        Scope = "Ranker tests only."
        When = "Use when touching scoring, prioritization, or rank ordering logic."
    }
    "pipeline" = @{
        Purpose = "Run pipeline execution tests."
        Scope = "Pipeline happy-path and failure-path coverage."
        When = "Use when touching run orchestration, digest generation flow, or pipeline error handling."
    }
    "ui" = @{
        Purpose = "Run validation and rendering behavior checks."
        Scope = "Workspace rendering, validation feedback, hidden diagnostics, and source UI behavior."
        When = "Use when form state, template output, or user-visible validation feedback changes."
    }
    "live-diagnostics" = @{
        Purpose = "Guide manual live URL investigation."
        Scope = "No stable regression suite unless an explicit diagnostic entry point exists."
        When = "Use during manual investigation of real external URLs."
    }
    "full" = @{
        Purpose = "Run the complete Django suite."
        Scope = "All tests."
        When = "Use only at task completion, before commit/push, or after cross-system changes."
    }
}

function Show-Usage {
    Write-Host "Usage:"
    Write-Host "  .\scripts\test.ps1 help"
    Write-Host "  .\scripts\test.ps1 single <test-path>"
    Write-Host "  .\scripts\test.ps1 saved-sources"
    Write-Host "  .\scripts\test.ps1 source-ingestion"
    Write-Host "  .\scripts\test.ps1 sources"
    Write-Host "  .\scripts\test.ps1 ranking"
    Write-Host "  .\scripts\test.ps1 pipeline"
    Write-Host "  .\scripts\test.ps1 ui"
    Write-Host "  .\scripts\test.ps1 live-diagnostics [url]"
    Write-Host "  .\scripts\test.ps1 full"
    Write-Host ""
    Write-Host "Commands:"
    foreach ($name in $commandInfo.Keys) {
        $info = $commandInfo[$name]
        Write-Host ("  {0}" -f $name)
        Write-Host ("    purpose: {0}" -f $info.Purpose)
        Write-Host ("    scope:   {0}" -f $info.Scope)
        Write-Host ("    when:    {0}" -f $info.When)
    }
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\test.ps1 single tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_invalid_url"
    Write-Host "  .\scripts\test.ps1 saved-sources"
    Write-Host "  .\scripts\test.ps1 source-ingestion"
    Write-Host "  .\scripts\test.ps1 ui"
    Write-Host "  .\scripts\test.ps1 ranking"
    Write-Host "  .\scripts\test.ps1 live-diagnostics https://example.com/article"
    Write-Host "  .\scripts\test.ps1 full"
}

if (-not (Test-Path $python)) {
    Write-Error "Python interpreter not found at $python"
    exit 1
}

if (-not $Level) {
    Show-Usage
    exit 1
}

$testTargets = @()
$normalizedLevel = $Level.ToLowerInvariant()

switch ($normalizedLevel) {
    "help" {
        Show-Usage
        exit 0
    }
    "single" {
        if (-not $Args -or $Args.Count -eq 0) {
            Write-Error "The 'single' command requires a Django test path."
            Show-Usage
            exit 1
        }
        $testTargets = $Args
    }
    "saved-sources" {
        $testTargets = @(
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_persists_source_in_inventory",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_saved_sources_render_newest_source_first",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_prevents_duplicate_normalized_url",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_readable_web_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_stanford_style_article_url_with_meaningful_id_query_param",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_hopkins_article_via_saved_source_path_when_primary_fetch_is_blocked",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_lullaby_trust_style_parenting_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_saves_reachable_web_article_even_when_extraction_is_unverified",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_prevents_duplicate_normalized_web_article_url",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_invalid_url",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_unreachable_generic_web_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_generic_web_article_that_returns_404",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_unreadable_rss_feed",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_missing_devto_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_valid_devto_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_devto_author_without_articles",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_can_disable_and_remove_topic_sources_from_review_ui"
        )
    }
    "source-ingestion" {
        $testTargets = @(
            "tests.test_rss_adapter",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_readable_web_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_stanford_style_article_url_with_meaningful_id_query_param",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_hopkins_article_via_saved_source_path_when_primary_fetch_is_blocked",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_lullaby_trust_style_parenting_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_saves_reachable_web_article_even_when_extraction_is_unverified",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_unreachable_generic_web_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_generic_web_article_that_returns_404",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_unreadable_rss_feed",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_missing_devto_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_accepts_valid_devto_article",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_topic_source_rejects_devto_author_without_articles"
        )
    }
    "sources" {
        $testTargets = @(
            "tests.test_rss_adapter",
            "tests.test_topic_rss_source"
        )
    }
    "ranking" {
        $testTargets = @(
            "tests.test_ranker"
        )
    }
    "pipeline" {
        $testTargets = @(
            "tests.test_pipeline_happy_path",
            "tests.test_pipeline_failures"
        )
    }
    "ui" {
        $testTargets = @(
            "tests.test_topic_rss_source.TopicRssSourceTests.test_topic_workspace_renders_focus_chip_editor",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_topic_list_form_disables_browser_native_validation",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_dashboard_sections_render_section_level_collapse_controls",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_newly_created_topic_appears_first_on_dashboard",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_workspace_topic_configuration_autosaves_without_explicit_save_button",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_curated_only_workspace_hides_discovery_section",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_discovery_only_workspace_hides_saved_sources_section",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_saved_source_does_not_render_inside_new_sources_section",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_new_sources_are_checked_by_default_after_discovery_and_persist_when_toggled",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_unchecked_discovered_source_post_matches_browser_shape_and_stays_unchecked_after_refresh",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_rediscovery_keeps_existing_discovered_source_unchecked_instead_of_resetting_default_selection",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_add_source_feedback_renders_below_form_and_without_technical_strings",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_reachable_but_extraction_unverified_source_saves_without_visible_warning",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_source_add_error_clears_when_input_is_edited",
            "tests.test_topic_rss_source.TopicRssSourceTests.test_saved_source_cards_hide_internal_metadata_labels"
        )
    }
    "live-diagnostics" {
        if (-not $Args -or $Args.Count -eq 0) {
            Write-Host "Live URL diagnostics are manual and should not be treated as stable regression tests."
            Write-Host "Pass a URL only when investigating a real fetch, redirect, detection, extraction, acceptance, or UI/form issue."
            Write-Host "There is no automated live-diagnostics entry point yet."
            Write-Host "TODO: document or add a dedicated diagnostic command before automating live URL checks."
            exit 0
        }
        Write-Host ("Live diagnostics are documented but not automated yet for URL: {0}" -f ($Args -join " "))
        Write-Host "TODO: add a dedicated diagnostic entry point before wiring this command to live external fetch behavior."
        exit 0
    }
    "full" {
        $testTargets = @()
    }
    default {
        Write-Error "Unknown test command: $Level"
        Show-Usage
        exit 1
    }
}

$command = @($manage, "test") + $testTargets
Write-Host ("Running: {0} {1}" -f $python, ($command -join " "))

& $python @command
exit $LASTEXITCODE
