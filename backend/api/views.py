import os
import io
import uuid
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser
from django.views.decorators.csrf import csrf_exempt
from .models import Upload, Prediction
from django.contrib.auth import get_user_model
from datetime import datetime
from PIL import Image, ImageDraw

User = get_user_model()

@api_view(['GET'])
@permission_classes([AllowAny])
def ping(request):
    return JsonResponse({'status': 'ok', 'time': datetime.utcnow().isoformat() + 'Z'})

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def predict_view(request):
    """Accepts image upload, stores it, runs mock inference, returns prediction JSON."""
    parser = MultiPartParser()
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image provided'}, status=400)

    image_file = request.FILES['image']
    original_name = image_file.name

    # For now, assign to anonymous user or create a demo user
    user, _ = User.objects.get_or_create(username='demo_user')

    upload = Upload.objects.create(user=user, image=image_file, original_filename=original_name)

    # Mock inference: initial defaults
    stages = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    import random
    predicted_stage = random.choice(stages)
    confidence = round(random.uniform(0.6, 0.99), 2)

    # Run inference (mock or real) - extracted to helper for easier testing/mocking
    try:
        infer_result = run_inference(upload.image.path)
        predicted_stage = infer_result.get('predicted_stage', predicted_stage)
        confidence = infer_result.get('confidence', confidence)
        heatmap_path = infer_result.get('heatmap_path')

        # Save prediction
        pred = Prediction.objects.create(upload=upload, predicted_stage=predicted_stage, confidence=confidence, heatmap=heatmap_path)

        response = {
            'predicted_stage': predicted_stage,
            'confidence': confidence,
            'heatmap_url': request.build_absolute_uri(settings.MEDIA_URL + heatmap_path) if heatmap_path else None,
            'timestamp': pred.timestamp.isoformat(),
        }
        return JsonResponse(response)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def run_inference(image_path: str) -> dict:
        """Mock inference helper.

        Returns a dict with keys: predicted_stage, confidence, heatmap_path (relative to MEDIA_ROOT).
        This can be replaced with real PyTorch code or a FastAPI call.
        """
        stages = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        import random
        predicted_stage = random.choice(stages)
        confidence = round(random.uniform(0.6, 0.99), 2)

        # Generate a dummy heatmap image overlaying red transparent blob
        img = Image.open(image_path).convert('RGB')
        heatmap = Image.new('RGBA', img.size, (255, 0, 0, 0))
        draw = ImageDraw.Draw(heatmap)
        w, h = img.size
        draw.ellipse((w * 0.2, h * 0.2, w * 0.8, h * 0.8), fill=(255, 0, 0, 120))
        heatmap_path = f"heatmaps/{uuid.uuid4()}.png"
        full_heatmap_path = os.path.join(settings.MEDIA_ROOT, heatmap_path)
        os.makedirs(os.path.dirname(full_heatmap_path), exist_ok=True)
        heatmap.convert('RGB').save(full_heatmap_path)

        return {
            'predicted_stage': predicted_stage,
            'confidence': confidence,
            'heatmap_path': heatmap_path,
        }


@api_view(['GET'])
def results_view(request):
    """List previous uploads + predictions."""
    preds = Prediction.objects.select_related('upload').order_by('-timestamp')[:100]
    data = []
    for p in preds:
        data.append({
            'id': str(p.id),
            'predicted_stage': p.predicted_stage,
            'confidence': p.confidence,
            'heatmap_url': request.build_absolute_uri(p.heatmap.url) if p.heatmap else None,
            'image_url': request.build_absolute_uri(p.upload.image.url) if p.upload.image else None,
            'timestamp': p.timestamp.isoformat(),
        })
    return JsonResponse({'results': data})
