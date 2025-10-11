# Diabetic Retinopathy Detection Web System

This repository contains a Django REST backend and a React frontend (planned) for uploading fundus images and getting diabetic retinopathy predictions with Grad-CAM heatmaps.

Backend (current): Django + DRF

Quick start (backend development):

1. Create a Python virtualenv and install dependencies:

   python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt

2. Create `.env` from `.env.example` and adjust settings if needed.

3. Run migrations and start dev server:

   cd backend; python manage.py migrate; python manage.py createsuperuser; python manage.py runserver

4. API endpoints:
   - GET /api/ping/ — health check
   - POST /api/predict/ — upload image (multipart form-data, field name `image`)
   - GET /api/results/ — list recent predictions

Notes:
- The current implementation uses a mock inference that generates a dummy heatmap and random predictions.
- For production, configure PostgreSQL via the `DATABASE_URL` env var and set DEBUG=0 and a secure SECRET_KEY.

Next steps:
- Implement real PyTorch model integration or a FastAPI model microservice.
- Scaffold React frontend with Tailwind/Material UI.
- Add Docker Compose for full stack local development.
