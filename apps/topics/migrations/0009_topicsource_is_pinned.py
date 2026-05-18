from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("topics", "0008_topic_focus_initialized"),
    ]

    operations = [
        migrations.AddField(
            model_name="topicsource",
            name="is_pinned",
            field=models.BooleanField(default=False),
        ),
    ]
