# Frontend (Vite + React + Tailwind)

This folder contains a Vite + React scaffold with Tailwind CSS.

Quick start:

1. cd frontend
2. npm install
3. npm run dev

The dev server runs at http://localhost:3000 (configured in vite.config.js). The React app calls Django endpoints at `/api/...` — use a proxy or run both apps on the same host (CORS enabled on backend).

Pages:
- Login (POST /api/token/)
- Upload (POST /api/predict/)
- Results (display latest)
- History (GET /api/results/)
