from django.db import models


# 用户
class User(models.Model):
    email = models.EmailField(max_length=254, unique=True)
    password = models.CharField(max_length=255)
    name = models.CharField(max_length=20)
    regtime = models.DateField(auto_now=True)
    token = models.CharField(max_length=32, default='')
    actcode = models.CharField(max_length=64)
    actstatus = models.BooleanField(default=False)

    def __str__(self):
        return self.email

    class Meta:
        db_table = 'user'   #更改表名


# 字符
class Character(models.Model):
    word = models.CharField(max_length=6, primary_key=True)
    page = models.PositiveSmallIntegerField(default=0)
    row = models.PositiveSmallIntegerField(default=0)
    col = models.PositiveSmallIntegerField(default=0)
    index = models.IntegerField(default=0)
    level = models.CharField(max_length=10, default='一级')

    def __str__(self):
        return self.word

    class Meta:
        db_table = 'character'   #更改表名


# 字符
class User_Char(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    character = models.ForeignKey(Character, on_delete=models.CASCADE)
    index = models.PositiveSmallIntegerField()
    posttime = models.DateField(auto_now=True)
    turnpoint = models.TextField(default='')
    ctrlpoint = models.TextField(default='')

    class Meta:
        db_table = 'user_char'   #更改表名
        unique_together = ('user', 'character', 'index')