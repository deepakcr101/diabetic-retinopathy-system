from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch
from io import BytesIO
from PIL import Image


class PredictEndpointTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('predict')

    def _create_test_image(self):
        img = Image.new('RGB', (256, 256), color=(73, 109, 137))
        buf = BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        return buf

    @patch('api.views.run_inference')
    def test_predict_happy_path(self, mock_infer):
        mock_infer.return_value = {
            'predicted_stage': 'Moderate DR',
            'confidence': 0.86,
            'heatmap_path': 'heatmaps/test-heatmap.png'
        }
        img_buf = self._create_test_image()
        response = self.client.post(self.url, {'image': img_buf}, format='multipart')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('predicted_stage', data)
        self.assertEqual(data['predicted_stage'], 'Moderate DR')
        self.assertIn('confidence', data)
        self.assertAlmostEqual(data['confidence'], 0.86, places=2)
        self.assertIn('heatmap_url', data)