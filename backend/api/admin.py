from django.contrib import admin
from .models import Upload, Prediction

@admin.register(Upload)
class UploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'original_filename', 'created_at')
    readonly_fields = ('id', 'created_at')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'predicted_stage', 'confidence', 'timestamp')
    readonly_fields = ('id', 'timestamp')
