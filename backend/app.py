
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import extract_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stridesense_rf.joblib"

WINDOW_SIZE = 128
SAMPLE_RATE = 64

app = FastAPI(title="StrideSense AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
bundle = None

if MODEL_PATH.exists():
    bundle = joblib.load(MODEL_PATH)
sensor_buffer = deque(maxlen=WINDOW_SIZE)

sample_count = 0
last_received_at = None
latest_sample = None

latest_prediction = {
    "status": "waiting_for_sensor",
    "score": None,
    "message": "Waiting for sensor measurements."
}


class Sample(BaseModel):
    timestamp_us: int
    ax_g: float
    ay_g: float
    az_g: float
    gx_dps: float
    gy_dps: float
    gz_dps: float


class SampleBatch(BaseModel):
    samples: list[Sample] = Field(min_length=1, max_length=256)


@app.get("/api/health")
def health():
    return {
        "backend": "online",
        "model_loaded": bundle is not None,
        "model_path": str(MODEL_PATH),
        "buffer_size": len(sensor_buffer)
    }


@app.post("/api/imu/batch")
def receive_batch(batch: SampleBatch):

    global sample_count
    global last_received_at
    global latest_sample
    global latest_prediction

    for sample in batch.samples:

        row = sample.model_dump()

        sensor_buffer.append(row)
        latest_sample = row
        sample_count += 1

    last_received_at = datetime.now(
        timezone.utc
    ).isoformat()

    if len(sensor_buffer) < WINDOW_SIZE:

        latest_prediction = {
            "status": "warming_up",
            "score": None,
            "message": "Collecting a complete two-second window."
        }

        return {
            "received": len(batch.samples),
            "buffer_size": len(sensor_buffer)
        }

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Run train_model.py first."
        )

    rows = list(sensor_buffer)

    acceleration = np.array([
        [r["ax_g"], r["ay_g"], r["az_g"]]
        for r in rows
    ], dtype=float)

    magnitude_mg = np.linalg.norm(
        acceleration,
        axis=1
    ) * 1000.0

    feature_dict = extract_features(
        magnitude_mg,
        sample_rate=SAMPLE_RATE
    )

    feature_names = bundle["feature_names"]

    X = pd.DataFrame(
        [[feature_dict[name] for name in feature_names]],
        columns=feature_names
    )

    model = bundle["model"]

    probabilities = model.predict_proba(X)[0]

    positive_index = list(model.classes_).index(1)

    model_score = float(probabilities[positive_index])

    latest_prediction = {
        "status": "experimental",
        "research_model_score": model_score,
        "message": (
            "Experimental model output for a window "
            "similar to research windows labelled as freezing. "
            "Not a medical assessment."
        )
    }

    return {
        "received": len(batch.samples),
        "buffer_size": len(sensor_buffer),
        "research_model_score": model_score
    }


@app.get("/api/imu/latest")
def get_latest():

    return {
        "sensor": latest_sample,
        "prediction": latest_prediction,
        "samples_received": sample_count,
        "last_received_at": last_received_at,
        "sensor_connected": last_received_at is not None
    }


@app.get("/api/imu/history")
def get_history():

    return list(sensor_buffer)


@app.get("/api/model/metrics")
def get_model_metrics():

    if bundle is None:
        raise HTTPException(
            status_code=404,
            detail="Model not loaded."
        )

    return bundle.get("evaluation", {})