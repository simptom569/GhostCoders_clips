# videoapp/models.py
from django.db import models

class VideoFile(models.Model):
    title = models.CharField(max_length=200)  # Это поле должно быть здесь
    video = models.FileField(upload_to='')  # Убедитесь, что здесь указан путь для сохранения видео
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
