from django.db import models

# Create your models here.
class User(models.Model):
    username = models.CharField('User name',max_length=50)
    email = models.CharField('User email', max_length=50)
    password = models.CharField('User Password', max_length=20)

class Contact(models.Model):
    first_name = models.CharField(max_length=10)
    last_name = models.CharField(max_length=10)
    email = models.CharField(max_length=20)
    comment = models.CharField(max_length=100)