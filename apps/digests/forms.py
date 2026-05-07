from django import forms


TOPIC_NAME_REQUIRED_MESSAGE = "Enter a topic name before starting the pipeline."


class TopicInputForm(forms.Form):
    topic_name = forms.CharField(
        label="Topic name",
        max_length=160,
        strip=True,
        error_messages={
            "required": TOPIC_NAME_REQUIRED_MESSAGE,
        },
    )
    source_url = forms.URLField(
        label="Source RSS URL",
        required=False,
        help_text="Optional. Paste an RSS feed URL to use real source items.",
    )
