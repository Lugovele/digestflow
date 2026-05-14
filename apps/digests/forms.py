from django import forms

from apps.topics.models import TopicSourceMode


TOPIC_NAME_REQUIRED_MESSAGE = "Enter a topic name before starting the pipeline."


class TopicInputForm(forms.Form):
    topic_name = forms.CharField(
        label="Topic",
        max_length=160,
        strip=True,
        error_messages={
            "required": TOPIC_NAME_REQUIRED_MESSAGE,
        },
    )
    source_url = forms.URLField(
        label="Enter a URL",
        required=False,
        help_text="Optional. Add one source URL to save it on the topic and include it in source review.",
        widget=forms.URLInput(attrs={"placeholder": "Enter a URL"}),
    )
    source_mode = forms.ChoiceField(
        label="Where to look",
        choices=TopicSourceMode.choices,
        initial=TopicSourceMode.HYBRID,
        help_text="",
    )
