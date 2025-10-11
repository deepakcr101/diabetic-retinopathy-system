import uuid
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Upload(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploads')
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    original_filename = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Upload {self.id} by {self.user.username}"

class Prediction(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    upload = models.OneToOneField(Upload, on_delete=models.CASCADE, related_name='prediction')
    predicted_stage = models.CharField(max_length=64)
    confidence = models.FloatField()
    heatmap = models.ImageField(upload_to='heatmaps/%Y/%m/%d/', blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.predicted_stage} ({self.confidence:.2f})"
