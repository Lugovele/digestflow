from django import forms


class TopicInputForm(forms.Form):
    topic_name = forms.CharField(
        label="Topic name",
        max_length=160,
        strip=True,
        error_messages={
            "required": "Введите тему, чтобы запустить digest.",
        },
    )
