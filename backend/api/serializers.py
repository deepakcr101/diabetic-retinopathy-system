from rest_framework import serializers
from .models import Upload, Prediction

class UploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Upload
        fields = ['id', 'user', 'image', 'original_filename', 'created_at']
        read_only_fields = ['id', 'user', 'created_at']

class PredictionSerializer(serializers.ModelSerializer):
    upload = UploadSerializer()

    class Meta:
        model = Prediction
        fields = ['id', 'upload', 'predicted_stage', 'confidence', 'heatmap', 'timestamp']
        read_only_fields = fields
