"""FastAPI inference service.

Endpoints:
- GET /health
- POST /predict  (multipart form-data with an image file)
- GET /metrics   (Prometheus metrics)

Monitoring:
- request count (Counter)
- request latency (Histogram)

Logging:
- JSON structured logs (request id, endpoint, latency_ms, outcome)
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

from src.api.logging import configure_logging
from src.api.model_loader import load_label_map, load_model, predict_image

from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests"
)

REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency"
)


configure_logging()
log = structlog.get_logger()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.keras")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "models/label_map.json")

REQUESTS = Counter("inference_requests_total", "Total number of inference requests", ["endpoint", "status"])
LATENCY = Histogram("inference_request_latency_seconds", "Inference latency in seconds", ["endpoint"])

app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0.0")

# Load once at startup
try:
    model = load_model(MODEL_PATH)
    label_map = load_label_map(LABEL_MAP_PATH)
    log.info("model_loaded", model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH)
except Exception as e:
    model = None
    label_map = {0: "cat", 1: "dog"}
    log.error("model_load_failed", error=str(e), model_path=MODEL_PATH)


@app.get("/health")
def health():
    status = "ok" if model is not None else "model_not_loaded"
    return {"status": status}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    req_id = str(uuid.uuid4())
    REQUEST_COUNT.inc()
    start = time.time()
    endpoint = "/predict"

    if model is None:
        REQUESTS.labels(endpoint=endpoint, status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        content = file.file.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Unable to decode image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pred_int, prob_dog = predict_image(model, img_rgb)
        label = label_map.get(pred_int, str(pred_int))
        prob_cat = 1.0 - prob_dog

        latency = time.time() - start
        LATENCY.labels(endpoint=endpoint).observe(latency)
        REQUESTS.labels(endpoint=endpoint, status="ok").inc()

        log.info(
            "prediction",
            request_id=req_id,
            filename=file.filename,
            latency_ms=int(latency * 1000),
            pred_label=label,
            prob_dog=round(prob_dog, 6),
        )
        REQUEST_LATENCY.observe(time.time() - start)
        return {
            "request_id": req_id,
            "label": label,
            "probabilities": {"cat": prob_cat, "dog": prob_dog},
        }

    except Exception as e:
        latency = time.time() - start
        LATENCY.labels(endpoint=endpoint).observe(latency)
        REQUESTS.labels(endpoint=endpoint, status="error").inc()
        log.error("prediction_failed", request_id=req_id, error=str(e), latency_ms=int(latency * 1000))
        raise HTTPException(status_code=400, detail=str(e))
