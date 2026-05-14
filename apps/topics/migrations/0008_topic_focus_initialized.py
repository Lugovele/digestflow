from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("topics", "0007_source_alter_topic_options_alter_topic_source_mode"),
    ]

    operations = [
        migrations.AddField(
            model_name="topic",
            name="focus_initialized",
            field=models.BooleanField(default=False),
        ),
    ]
