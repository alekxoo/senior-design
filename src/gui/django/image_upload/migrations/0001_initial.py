# Generated by Django 5.1.1 on 2024-10-04 16:47

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='image_uploads/')),
                ('upload_time', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]