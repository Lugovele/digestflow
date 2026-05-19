from django.urls import path

from .views import (
    add_topic_source_view,
    create_topic_and_run_view,
    delete_topic_view,
    discover_sources_view,
    pin_topic_source_view,
    reorder_topics_view,
    run_detail_view,
    run_pipeline_view,
    run_with_selected_sources_view,
    remove_topic_source_view,
    toggle_topic_source_view,
    topic_list_view,
    topic_workspace_view,
    unpin_topic_source_view,
    update_topic_focus_view,
)


urlpatterns = [
    path("", topic_list_view, name="topic-list"),
    path("topics/<int:topic_id>/", topic_workspace_view, name="topic-workspace"),
    path("discover-sources/", discover_sources_view, name="discover-sources"),
    path("topics/reorder/", reorder_topics_view, name="reorder-topics"),
    path("topics/<int:topic_id>/delete/", delete_topic_view, name="delete-topic"),
    path("topics/<int:topic_id>/focus/", update_topic_focus_view, name="update-topic-focus"),
    path("topics/<int:topic_id>/sources/add/", add_topic_source_view, name="add-topic-source"),
    path("quick-start/", create_topic_and_run_view, name="create-topic-and-run"),
    path("runs/<int:run_id>/", run_detail_view, name="run-detail"),
    path("topics/<int:topic_id>/run/", run_pipeline_view, name="run-pipeline"),
    path(
        "topics/<int:topic_id>/sources/<int:source_id>/toggle/",
        toggle_topic_source_view,
        name="toggle-topic-source",
    ),
    path(
        "topics/<int:topic_id>/sources/<int:source_id>/remove/",
        remove_topic_source_view,
        name="remove-topic-source",
    ),
    path(
        "topics/<int:topic_id>/sources/<int:source_id>/pin/",
        pin_topic_source_view,
        name="pin-topic-source",
    ),
    path(
        "topics/<int:topic_id>/sources/<int:source_id>/unpin/",
        unpin_topic_source_view,
        name="unpin-topic-source",
    ),
    path(
        "topics/<int:topic_id>/run-selected/",
        run_with_selected_sources_view,
        name="run-with-selected-sources",
    ),
]
