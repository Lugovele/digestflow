from django.urls import path

from .views import (
    create_topic_and_run_view,
    run_detail_view,
    run_pipeline_view,
    topic_list_view,
)


urlpatterns = [
    path("", topic_list_view, name="topic-list"),
    path("quick-start/", create_topic_and_run_view, name="create-topic-and-run"),
    path("runs/<int:run_id>/", run_detail_view, name="run-detail"),
    path("topics/<int:topic_id>/run/", run_pipeline_view, name="run-pipeline"),
]
