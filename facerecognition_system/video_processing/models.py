from django.db import models

from django.db import models

class FaceRecognition(models.Model):
    mssv = models.CharField(max_length=50, primary_key=True, default="unknown")
    name = models.CharField(max_length=100, default="unknown")
    encoding = models.BinaryField()

    def __str__(self):
        return self.name