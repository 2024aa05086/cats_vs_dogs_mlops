"""
FastAPI inference service.

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

import cv2
import numpy as np
import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from src.api.logging import configure_logging
from src.api.model_loader import load_label_map, load_model, predict_image

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
configure_logging()
log = structlog.get_logger()

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.keras")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "models/label_map.json")

# --------------------------------------------------------------------
# Prometheus metrics (custom registry to avoid duplicate timeseries)
# --------------------------------------------------------------------
REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["endpoint", "status"],
    registry=REGISTRY,
)

LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
    registry=REGISTRY,
)

# --------------------------------------------------------------------
# App
# --------------------------------------------------------------------
app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0.0")

# Load once at startup
try:
    model = load_model(MODEL_PATH)
    label_map = load_label_map(LABEL_MAP_PATH)
    log.info("model_loaded", model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH)
except Exception as e:
    model = None
    label_map = {0: "cat", 1: "dog"}  # safe fallback
    log.error("model_load_failed", error=str(e), model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH)


@app.get("/health")
def health():
    status = "ok" if model is not None else "model_not_loaded"
    return {"status": status}


@app.get("/metrics")
def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    endpoint = "/predict"
    req_id = str(uuid.uuid4())
    start = time.time()

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
        prob_cat = 1.0 - float(prob_dog)

        latency_s = time.time() - start
        LATENCY.labels(endpoint=endpoint).observe(latency_s)
        REQUESTS.labels(endpoint=endpoint, status="ok").inc()

        log.info(
            "prediction",
            request_id=req_id,
            endpoint=endpoint,
            filename=file.filename,
            latency_ms=int(latency_s * 1000),
            pred_label=label,
            prob_dog=round(float(prob_dog), 6),
            outcome="ok",
        )

        return {
            "request_id": req_id,
            "label": label,
            "probabilities": {"cat": prob_cat, "dog": float(prob_dog)},
        }

    except Exception as e:
        latency_s = time.time() - start
        LATENCY.labels(endpoint=endpoint).observe(latency_s)
        REQUESTS.labels(endpoint=endpoint, status="error").inc()

        log.error(
            "prediction_failed",
            request_id=req_id,
            endpoint=endpoint,
            latency_ms=int(latency_s * 1000),
            outcome="error",
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
