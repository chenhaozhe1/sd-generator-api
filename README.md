# SD Generator API

A lightweight **FastAPI** service for local Stable Diffusion image generation (SD 1.5 / SDXL, optional local “leosam” model).  
It exposes simple HTTP endpoints for generating images and browsing recent outputs.

## Features

- **Models**: SD 1.5, SDXL; optional local model via `HF_ID_LEOSAM` (file or folder path).
- **Endpoints**:
  - `POST /generate` → returns JSON with a file URL under `/outputs/...`
  - `POST /generate_bytes` → returns the image file as the HTTP response body
  - `GET /gallery` → simple HTML gallery for generated images
  - `GET /healthz` → health check
  - `GET /metrics` → Prometheus metrics
  - Swagger UI: `http://127.0.0.1:8000/docs`
- **Safety**: prompt moderation + optional Diffusers safety checker
- **Telemetry**: Prometheus counters & latency histograms

## Quickstart

```bash
# (recommended) create and activate a virtual env first
pip install -r requirements.txt

# start the API
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
