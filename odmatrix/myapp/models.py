# models.py
from django.db import models

class ODMatrix(models.Model):
    from_stage = models.CharField(max_length=255)
    to_stage = models.CharField(max_length=255)
    passenger_count = models.IntegerField()
