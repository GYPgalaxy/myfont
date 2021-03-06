# Generated by Django 3.1.4 on 2021-04-18 08:10

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Character',
            fields=[
                ('word', models.CharField(max_length=6, primary_key=True, serialize=False)),
                ('page', models.PositiveSmallIntegerField(default=0)),
                ('row', models.PositiveSmallIntegerField(default=0)),
                ('col', models.PositiveSmallIntegerField(default=0)),
                ('index', models.IntegerField(default=0)),
                ('level', models.CharField(default='一级', max_length=10)),
            ],
            options={
                'db_table': 'character',
            },
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('password', models.CharField(max_length=255)),
                ('name', models.CharField(max_length=20)),
                ('regtime', models.DateField(auto_now=True)),
                ('token', models.CharField(default='', max_length=32)),
                ('actcode', models.CharField(max_length=64)),
                ('actstatus', models.BooleanField(default=False)),
            ],
            options={
                'db_table': 'user',
            },
        ),
        migrations.CreateModel(
            name='User_Char',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('index', models.PositiveSmallIntegerField()),
                ('posttime', models.DateField(auto_now=True)),
                ('turnpoint', models.TextField(default='')),
                ('ctrlpoint', models.TextField(default='')),
                ('character', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='myfont.character')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='myfont.user')),
            ],
            options={
                'db_table': 'user_char',
                'unique_together': {('user', 'character', 'index')},
            },
        ),
    ]
