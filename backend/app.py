"""StrideSense API: live sensor ingestion, session recording and review.

This is a research prototype. Model outputs are not medical assessments.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
import csv
import io
import json
import sqlite3
import threading
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from features import extract_features

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from xml.sax.saxutils import escape

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stridesense_rf.joblib"
DB_PATH = BASE_DIR / "recordings" / "stridesense.sqlite3"
WINDOW_SIZE = 128                 # 2 seconds at 64 Hz
INFERENCE_STEP = 32                # infer every 0.5 second
SAMPLE_RATE = 64
# This is a review threshold for showing candidate windows, not a clinical cutoff.
CANDIDATE_SCORE_THRESHOLD = 0.50

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="StrideSense AI",
    version="1.0.0",
    description="Experimental mobility research prototype. Not for clinical use.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_lock = threading.RLock()
sensor_buffer: deque[dict] = deque(maxlen=WINDOW_SIZE)
recent_samples: deque[dict] = deque(maxlen=WINDOW_SIZE)
sample_count = 0
last_received_at: str | None = None
latest_sample: dict | None = None
latest_prediction: dict = {
    "status": "waiting_for_sensor",
    "research_model_score": None,
    "message": "Waiting for sensor measurements.",
}
active_session_id: str | None = None
samples_since_inference = 0
bundle = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat(timespec="milliseconds")


@contextmanager
def db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialise_database() -> None:
    with db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                prediction_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                received_at TEXT NOT NULL,
                timestamp_us INTEGER NOT NULL,
                ax_g REAL NOT NULL,
                ay_g REAL NOT NULL,
                az_g REAL NOT NULL,
                gx_dps REAL NOT NULL,
                gy_dps REAL NOT NULL,
                gz_dps REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id, id);
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                created_at TEXT NOT NULL,
                window_start_at TEXT,
                window_end_at TEXT NOT NULL,
                score REAL NOT NULL,
                threshold REAL NOT NULL,
                candidate INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id, id);
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                review_status TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(session_id, event_id),
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(event_id) REFERENCES predictions(id)
            );
            """
        )
        # A process restart must not leave a recording falsely marked active.
        conn.execute(
            "UPDATE sessions SET status='interrupted', ended_at=? WHERE status='recording'",
            (utc_iso(),),
        )


@app.on_event("startup")
def startup_event():
    global bundle
    initialise_database()
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
        except Exception as exc:
            print(f"WARNING: model could not be loaded: {exc}")
            bundle = None
    else:
        bundle = None


class Sample(BaseModel):
    timestamp_us: int
    ax_g: float = Field(ge=-20, le=20)
    ay_g: float = Field(ge=-20, le=20)
    az_g: float = Field(ge=-20, le=20)
    gx_dps: float = Field(ge=-3000, le=3000)
    gy_dps: float = Field(ge=-3000, le=3000)
    gz_dps: float = Field(ge=-3000, le=3000)


class SampleBatch(BaseModel):
    samples: list[Sample] = Field(min_length=1, max_length=256)


class SessionStart(BaseModel):
    label: str = Field(default="Mobility recording", min_length=1, max_length=80)


class ReviewInput(BaseModel):
    review_status: Literal["confirmed", "not_observed", "uncertain"]
    note: str = Field(default="", max_length=500)


def run_model_inference(window_rows: list[dict]) -> float:
    if bundle is None:
        raise RuntimeError("Trained model is not loaded.")
    acceleration = np.asarray(
        [[r["ax_g"], r["ay_g"], r["az_g"]] for r in window_rows],
        dtype=np.float64,
    )
    # ESP32 firmware emits g; the Daphnet training files store milli-g.
    magnitude_mg = np.linalg.norm(acceleration, axis=1) * 1000.0
    features = extract_features(magnitude_mg, sample_rate=SAMPLE_RATE)
    names = bundle["feature_names"]
    frame = pd.DataFrame([[features[name] for name in names]], columns=names)
    model = bundle["model"]
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError("Model bundle does not contain the freezing research class.")
    return float(probabilities[classes.index(1)])


def get_candidate_events(session_id: str) -> list[dict]:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE session_id=? AND candidate=1 ORDER BY id",
            (session_id,),
        ).fetchall()
        reviews = conn.execute(
            "SELECT event_id, review_status, note FROM reviews WHERE session_id=?",
            (session_id,),
        ).fetchall()
    review_map = {r["event_id"]: dict(r) for r in reviews}

    # Merge overlapping 2-second windows into reviewable segments.
    events: list[dict] = []
    for row in rows:
        created = datetime.fromisoformat(row["created_at"])
        if events:
            previous = events[-1]
            prev_created = datetime.fromisoformat(previous["last_prediction_at"])
            gap = (created - prev_created).total_seconds()
            if gap <= 1.25:
                previous["last_prediction_at"] = row["created_at"]
                previous["end_at"] = row["window_end_at"]
                previous["window_count"] += 1
                previous["max_score"] = max(previous["max_score"], row["score"])
                previous["mean_score_sum"] += row["score"]
                previous["mean_score"] = previous["mean_score_sum"] / previous["window_count"]
                continue
        event = {
            "event_id": row["id"],
            "start_at": row["window_start_at"] or row["created_at"],
            "end_at": row["window_end_at"],
            "last_prediction_at": row["created_at"],
            "window_count": 1,
            "max_score": row["score"],
            "mean_score_sum": row["score"],
            "mean_score": row["score"],
        }
        events.append(event)

    for event in events:
        event.pop("last_prediction_at", None)
        event.pop("mean_score_sum", None)
        review = review_map.get(event["event_id"])
        event["review_status"] = review["review_status"] if review else "awaiting_review"
        event["review_note"] = review["note"] if review else ""
        try:
            start = datetime.fromisoformat(event["start_at"])
            end = datetime.fromisoformat(event["end_at"])
            event["duration_seconds"] = max(0.0, (end - start).total_seconds())
        except ValueError:
            event["duration_seconds"] = None
    return events


@app.get("/api/health")
def health():
    return {
        "backend": "online",
        "model_loaded": bundle is not None,
        "model_path": str(MODEL_PATH),
        "recording_active": active_session_id is not None,
    }


@app.post("/api/imu/batch")
def receive_batch(batch: SampleBatch):
    global sample_count, last_received_at, latest_sample
    global latest_prediction, samples_since_inference

    now = utc_now()
    received_at = now.isoformat(timespec="milliseconds")
    rows = [sample.model_dump() for sample in batch.samples]

    with state_lock:
        for row in rows:
            sensor_buffer.append(row)
            recent_samples.append(row)
            latest_sample = row
            sample_count += 1
            samples_since_inference += 1
        last_received_at = received_at
        current_session = active_session_id
        buffer_rows = list(sensor_buffer)
        should_infer = (
            len(buffer_rows) >= WINDOW_SIZE and
            samples_since_inference >= INFERENCE_STEP
        )
        if should_infer:
            samples_since_inference = 0

    if current_session:
        with db_connection() as conn:
            conn.executemany(
                """INSERT INTO samples
                (session_id, received_at, timestamp_us, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(
                    current_session, received_at, r["timestamp_us"], r["ax_g"], r["ay_g"],
                    r["az_g"], r["gx_dps"], r["gy_dps"], r["gz_dps"]
                ) for r in rows],
            )
            conn.execute(
                "UPDATE sessions SET sample_count=sample_count+? WHERE id=? AND status='recording'",
                (len(rows), current_session),
            )

    inference_result = None
    if should_infer:
        if bundle is None:
            latest_prediction = {
                "status": "model_missing",
                "research_model_score": None,
                "message": "Copy the trained model to backend/models/stridesense_rf.joblib.",
            }
        else:
            try:
                score = run_model_inference(buffer_rows[-WINDOW_SIZE:])
                candidate = score >= CANDIDATE_SCORE_THRESHOLD
                created = utc_iso()
                window_end = created
                window_start = (datetime.fromisoformat(created) - timedelta(seconds=2)).isoformat(timespec="milliseconds")
                latest_prediction = {
                    "status": "experimental",
                    "research_model_score": score,
                    "candidate_for_review": candidate,
                    "review_threshold": CANDIDATE_SCORE_THRESHOLD,
                    "message": (
                        "This uncalibrated research-model score is only a prompt for optional review. "
                        "It is not a probability of freezing and is not a medical assessment."
                    ),
                }
                inference_result = score
                if current_session:
                    with db_connection() as conn:
                        cur = conn.execute(
                            """INSERT INTO predictions
                            (session_id, created_at, window_start_at, window_end_at, score, threshold, candidate)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (current_session, created, window_start, window_end,
                             score, CANDIDATE_SCORE_THRESHOLD, int(candidate)),
                        )
                        conn.execute(
                            "UPDATE sessions SET prediction_count=prediction_count+1 WHERE id=?",
                            (current_session,),
                        )
                    latest_prediction["prediction_id"] = cur.lastrowid
            except Exception as exc:
                latest_prediction = {
                    "status": "inference_error",
                    "research_model_score": None,
                    "message": f"Inference failed: {type(exc).__name__}: {exc}",
                }

    response = {"received": len(rows), "buffer_size": len(sensor_buffer)}
    if inference_result is not None:
        response["research_model_score"] = inference_result
    return response


@app.get("/api/imu/latest")
def get_latest():
    with state_lock:
        age_seconds = None
        if last_received_at:
            age_seconds = max(0.0, (utc_now() - datetime.fromisoformat(last_received_at)).total_seconds())
        return {
            "sensor": latest_sample,
            "prediction": latest_prediction,
            "samples_received": sample_count,
            "last_received_at": last_received_at,
            "sensor_connected": age_seconds is not None and age_seconds < 3.0,
            "sensor_data_age_seconds": age_seconds,
            "recording_session_id": active_session_id,
        }


@app.get("/api/imu/history")
def get_history():
    with state_lock:
        return list(recent_samples)


@app.get("/api/model/metrics")
def get_model_metrics():
    if bundle is None:
        raise HTTPException(status_code=404, detail="Model is not loaded. Train it or copy the model bundle into backend/models.")
    return bundle.get("metrics", bundle.get("evaluation", {}))


@app.post("/api/sessions/start")
def start_session(payload: SessionStart):
    global active_session_id, samples_since_inference
    with state_lock:
        if active_session_id is not None:
            raise HTTPException(status_code=409, detail="A recording session is already active.")
        session_id = str(uuid.uuid4())
        started = utc_iso()
        with db_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, label, started_at, status) VALUES (?, ?, ?, 'recording')",
                (session_id, payload.label.strip(), started),
            )
        active_session_id = session_id
        # Start a clean session-aligned window so the first prediction cannot
        # accidentally include sensor samples recorded before the session.
        sensor_buffer.clear()
        samples_since_inference = 0
    return {"id": session_id, "label": payload.label.strip(), "started_at": started, "status": "recording"}


@app.post("/api/sessions/stop")
def stop_session():
    global active_session_id
    with state_lock:
        if active_session_id is None:
            raise HTTPException(status_code=409, detail="No recording session is active.")
        session_id = active_session_id
        ended = utc_iso()
        with db_connection() as conn:
            conn.execute(
                "UPDATE sessions SET status='completed', ended_at=? WHERE id=?",
                (ended, session_id),
            )
        active_session_id = None
    return {"id": session_id, "ended_at": ended, "status": "completed"}


@app.get("/api/sessions/current")
def current_session():
    if active_session_id is None:
        return {"active": False, "session": None}
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (active_session_id,)).fetchone()
    return {"active": row is not None, "session": dict(row) if row else None}


@app.get("/api/sessions")
def list_sessions():
    with db_connection() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 100").fetchall()
    return [dict(row) for row in rows]


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return dict(row)


@app.get("/api/sessions/{session_id}/events")
def session_events(session_id: str):
    with db_connection() as conn:
        exists = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
    if exists is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return get_candidate_events(session_id)


@app.post("/api/sessions/{session_id}/events/{event_id}/review")
def review_event(session_id: str, event_id: int, payload: ReviewInput):
    events = get_candidate_events(session_id)
    event = next((e for e in events if e["event_id"] == event_id), None)
    if event is None:
        raise HTTPException(status_code=404, detail="Candidate segment not found.")
    with db_connection() as conn:
        conn.execute(
            """INSERT INTO reviews (session_id, event_id, review_status, note, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, event_id) DO UPDATE SET
            review_status=excluded.review_status, note=excluded.note, created_at=excluded.created_at""",
            (session_id, event_id, payload.review_status, payload.note, utc_iso()),
        )
    return {"saved": True, "event_id": event_id, "review_status": payload.review_status}



@app.get("/api/sessions/{session_id}/report")
def session_report(session_id: str):
    detail = session_detail(session_id)
    events = get_candidate_events(session_id)

    review_counts = {
        "confirmed": sum(e["review_status"] == "confirmed" for e in events),
        "not_observed": sum(e["review_status"] == "not_observed" for e in events),
        "uncertain": sum(e["review_status"] == "uncertain" for e in events),
        "awaiting_review": sum(e["review_status"] == "awaiting_review" for e in events),
    }

    with db_connection() as conn:
        sample_rows = conn.execute(
            """
            SELECT timestamp_us, ax_g, ay_g, az_g,
                   gx_dps, gy_dps, gz_dps
            FROM samples
            WHERE session_id=?
            ORDER BY rowid
            """,
            (session_id,),
        ).fetchall()

        prediction_rows = conn.execute(
            """
            SELECT score
            FROM predictions
            WHERE session_id=?
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()

    if not sample_rows:
        findings = [
            "RECORDING STATUS\nNo sensor samples were recorded during this session.",
            "MODEL FINDINGS\nThere is insufficient sensor data to evaluate movement patterns.",
            "OVERALL INTERPRETATION\nNo meaningful movement interpretation can be generated from this recording.",
            "SUGGESTED NEXT STEP\nRecord a new session with the sensor connected and verify that measurements are being received.",
        ]

        return {
            "session": detail,
            "sample_count": 0,
            "prediction_windows": 0,
            "candidate_segments": len(events),
            "review_counts": review_counts,
            "mean_research_model_score": None,
            "max_research_model_score": None,
            "recorded_span_seconds": 0,
            "movement_metrics": None,
            "movement_timeline": [],
            "session_summary": "No sensor samples were recorded during this session.",
            "findings_summary": "\n\n".join(findings),
            "movement_observations": [],
            "candidate_events": events,
            "disclaimer": "Experimental sensor observations only. Not clinically validated.",
        }

    ax = np.array([float(r["ax_g"]) for r in sample_rows])
    ay = np.array([float(r["ay_g"]) for r in sample_rows])
    az = np.array([float(r["az_g"]) for r in sample_rows])

    gx = np.array([float(r["gx_dps"]) for r in sample_rows])
    gy = np.array([float(r["gy_dps"]) for r in sample_rows])
    gz = np.array([float(r["gz_dps"]) for r in sample_rows])

    acceleration_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    dynamic_acceleration = np.abs(acceleration_magnitude - 1.0)
    rotation_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)

    sample_count = len(sample_rows)
    duration_seconds = sample_count / SAMPLE_RATE
    duration_minutes = round(duration_seconds / 60, 1)

    samples_per_bin = max(1, int(SAMPLE_RATE * 5))
    movement_timeline = []

    for start in range(0, sample_count, samples_per_bin):
        end = min(start + samples_per_bin, sample_count)

        acc_segment = dynamic_acceleration[start:end]
        gyro_segment = rotation_magnitude[start:end]

        movement_timeline.append({
            "start_seconds": round(start / SAMPLE_RATE, 2),
            "end_seconds": round(end / SAMPLE_RATE, 2),
            "mean_dynamic_acceleration_g": round(float(np.mean(acc_segment)), 4),
            "rms_dynamic_acceleration_g": round(
                float(np.sqrt(np.mean(acc_segment**2))), 4
            ),
            "mean_rotation_dps": round(float(np.mean(gyro_segment)), 3),
        })

    low_motion_threshold = 0.08
    low_motion_samples = int(
        np.sum(dynamic_acceleration < low_motion_threshold)
    )

    scores = [float(r["score"]) for r in prediction_rows]

    movement_metrics = {
        "mean_dynamic_acceleration_g": round(float(np.mean(dynamic_acceleration)), 4),
        "rms_dynamic_acceleration_g": round(
            float(np.sqrt(np.mean(dynamic_acceleration**2))), 4
        ),
        "dynamic_acceleration_std_g": round(float(np.std(dynamic_acceleration)), 4),
        "mean_rotation_dps": round(float(np.mean(rotation_magnitude)), 3),
        "rms_rotation_dps": round(
            float(np.sqrt(np.mean(rotation_magnitude**2))), 3
        ),
        "low_motion_sample_percentage": round(
            100 * low_motion_samples / sample_count, 2
        ),
        "low_motion_threshold_g": low_motion_threshold,
    }

    session_summary = (
        f"This session recorded {duration_minutes} minutes of movement data "
        f"across {sample_count:,} sensor samples. "
    )

    if events:
        session_summary += (
            f"The research model flagged {len(events)} candidate movement "
            "segments for human review. "
        )
    else:
        session_summary += (
            "The research model did not flag any candidate movement segments. "
            "This does not rule out movement difficulties. "
        )

    session_summary += (
        f"{review_counts['confirmed']} segments were reported as experienced, "
        f"{review_counts['not_observed']} were not noticed, "
        f"{review_counts['uncertain']} were marked uncertain, and "
        f"{review_counts['awaiting_review']} still require review."
    )

    findings = []

    findings.append(
        "MODEL FINDINGS\n"
        + (
            f"The research model identified {len(events)} candidate movement "
            "segments for review. These are experimental model detections, "
            "not confirmed freezing-of-gait episodes."
            if events else
            "The research model did not identify candidate segments in this "
            "recording. This does not establish that movement difficulties "
            "were absent."
        )
    )

    findings.append(
        "USER FEEDBACK\n"
        f"The user reported experiencing {review_counts['confirmed']} "
        f"segments, did not notice {review_counts['not_observed']} segments, "
        f"marked {review_counts['uncertain']} as uncertain, and has "
        f"{review_counts['awaiting_review']} segments awaiting review."
    )

    findings.append(
        "SENSOR OBSERVATIONS\n"
        f"{movement_metrics['low_motion_sample_percentage']}% of recorded "
        "samples fell below the prototype's experimental low-motion "
        "signal threshold. This describes the sensor measurements only. "
        "It does not independently establish freezing or inactivity."
    )

    if review_counts["confirmed"] > 0:
        interpretation = (
            "Some model-flagged periods were also reported by the user as "
            "movement difficulties. These periods may be useful to discuss "
            "with a physiotherapist. The prototype cannot determine whether "
            "they were clinical freezing-of-gait episodes."
        )
    elif events:
        interpretation = (
            "The model flagged candidate periods, but the available user "
            "feedback does not confirm that these represented movement "
            "difficulties. They should be interpreted cautiously."
        )
    else:
        interpretation = (
            "No candidate periods were flagged by the model. This is not "
            "evidence that the user had no movement difficulties, because "
            "the prototype may miss relevant events."
        )

    findings.append("OVERALL INTERPRETATION\n" + interpretation)

    if review_counts["awaiting_review"] > 0:
        next_step = (
            "Review the remaining candidate segments and record feedback "
            "where possible."
        )
    elif review_counts["confirmed"] > 0:
        next_step = (
            "Consider discussing the user-reported periods and corresponding "
            "sensor observations with a qualified physiotherapist."
        )
    else:
        next_step = (
            "If the user experienced difficulties during the recording, "
            "document them and discuss them with a qualified healthcare professional."
        )

    findings.append("SUGGESTED NEXT STEP\n" + next_step)

    movement_observations = [
        (
            f"{movement_metrics['low_motion_sample_percentage']}% of samples "
            "fell below the experimental low-motion threshold. "
            "This is a sensor observation, not a clinical finding."
        ),
        f"The recording contained {len(movement_timeline)} five-second analysis intervals.",
        f"Mean dynamic acceleration: {movement_metrics['mean_dynamic_acceleration_g']} g.",
        f"Mean rotational activity: {movement_metrics['mean_rotation_dps']} degrees per second.",
    ]

    if events:
        movement_observations.append(
            f"{len(events)} candidate segments are available for review."
        )

    return {
        "session": detail,
        "sample_count": sample_count,
        "prediction_windows": len(scores),
        "candidate_segments": len(events),
        "review_counts": review_counts,
        "mean_research_model_score": float(np.mean(scores)) if scores else None,
        "max_research_model_score": float(np.max(scores)) if scores else None,
        "recorded_span_seconds": duration_seconds,
        "movement_metrics": movement_metrics,
        "movement_timeline": movement_timeline,
        "session_summary": session_summary,
        "findings_summary": "\n\n".join(findings),
        "movement_observations": movement_observations,
        "candidate_events": events,
        "disclaimer": (
            "Experimental sensor observations only. Low sensor activity does "
            "not establish inactivity or freezing. Candidate segments require "
            "human review. Not clinically validated."
        ),
    }



@app.get("/api/sessions/{session_id}/report.pdf")
def export_session_pdf(session_id: str):
    report = session_report(session_id)
    session = report["session"]
    metrics = report.get("movement_metrics")

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=2.4 * cm,
        bottomMargin=1.8 * cm,
        title="StrideSense AI - Session Report",
        author="StrideSense AI Research Prototype",
    )

    navy = colors.HexColor("#101B35")
    teal = colors.HexColor("#14B8A6")
    light_bg = colors.HexColor("#F1F5F9")
    muted = colors.HexColor("#64748B")
    border = colors.HexColor("#DCE4EE")

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=navy,
        alignment=TA_LEFT,
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=muted,
        spaceAfter=15,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=navy,
        spaceBefore=13,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="BodyTextCustom",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="FindingHeading",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=navy,
        spaceBefore=5,
        spaceAfter=3,
    ))

    styles.add(ParagraphStyle(
        name="SmallText",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=muted,
    ))

    styles.add(ParagraphStyle(
        name="TableText",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=navy,
    ))

    styles.add(ParagraphStyle(
        name="WhiteHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    ))

    def P(value, style="BodyTextCustom"):
        return Paragraph(escape(str(value if value is not None else "—")), styles[style])

    def section(title):
        return [
            Spacer(1, 5),
            Paragraph(title, styles["SectionHeading"]),
            HRFlowable(
                width="100%",
                thickness=1,
                color=teal,
                spaceAfter=9,
            ),
        ]

    def make_table(data, widths=None):
        table = Table(
            data,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, border),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, light_bg]),
        ]))
        return table

    def page_layout(canvas, document):
        canvas.saveState()
        page_width, page_height = landscape(A4)

        canvas.setFillColor(navy)
        canvas.rect(
            0, page_height - 1.15 * cm,
            page_width, 1.15 * cm,
            fill=1, stroke=0,
        )

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(
            1.6 * cm, page_height - 0.72 * cm,
            "STRIDESENSE AI",
        )

        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            page_width - 1.6 * cm,
            page_height - 0.72 * cm,
            "EXPERIMENTAL MOBILITY RESEARCH REPORT",
        )

        canvas.setStrokeColor(border)
        canvas.line(
            1.6 * cm, 1.25 * cm,
            page_width - 1.6 * cm, 1.25 * cm,
        )

        canvas.setFillColor(muted)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            1.6 * cm, 0.85 * cm,
            "StrideSense AI | Experimental sensor observations only",
        )
        canvas.drawRightString(
            page_width - 1.6 * cm, 0.85 * cm,
            f"Page {document.page}",
        )

        canvas.restoreState()

    story = []

    story.append(Spacer(1, 12))
    story.append(P("Mobility Session Report", "ReportTitle"))
    story.append(P(
        "An experimental analysis of recorded movement sensor data",
        "Subtitle",
    ))
    story.append(HRFlowable(
        width="100%",
        thickness=2,
        color=teal,
        spaceAfter=14,
    ))

    # 01 - Session overview
    story.extend(section("01  |  Session Overview"))

    overview = [
        ["SESSION", "RECORDING", "ANALYSIS"],
        [
            P(session.get("label", "Mobility recording"), "TableText"),
            P(session.get("status", "Unknown"), "TableText"),
            P(f"{report['prediction_windows']} prediction windows", "TableText"),
        ],
        [
            P("Session ID: " + str(session.get("id", "—")), "SmallText"),
            P(f"{report['sample_count']:,} sensor samples", "TableText"),
            P(f"{report['candidate_segments']} candidate segments", "TableText"),
        ],
        [
            P("Started: " + str(session.get("started_at", "—")), "SmallText"),
            P(f"{report.get('recorded_span_seconds', 0):.1f} seconds recorded", "TableText"),
            P("Human review required", "TableText"),
        ],
    ]

    overview_table = Table(
        overview,
        colWidths=[8.5 * cm, 8.5 * cm, 8.5 * cm],
    )
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), light_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(overview_table)

    # 02 - Session summary
    story.extend(section("02  |  Session Summary"))

    summary = report.get("session_summary", "No session summary is available.")
    summary_box = Table(
        [[P(summary)]],
        colWidths=[25.5 * cm],
    )
    summary_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_bg),
        ("BOX", (0, 0), (-1, -1), 0.7, border),
        ("LINEBEFORE", (0, 0), (0, -1), 3, teal),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(summary_box)

    # 03 - New written findings and interpretation
    story.extend(section("03  |  Findings and Interpretation"))

    findings_text = report.get(
        "findings_summary",
        "No written interpretation is available.",
    )

    finding_parts = [
        part.strip()
        for part in findings_text.split("\n\n")
        if part.strip()
    ]

    finding_flowables = []

    for part in finding_parts:
        lines = part.split("\n", 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        finding_flowables.append(P(heading, "FindingHeading"))

        if body:
            finding_flowables.append(P(body))

    findings_box = Table(
        [[finding_flowables]],
        colWidths=[25.5 * cm],
    )
    findings_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_bg),
        ("BOX", (0, 0), (-1, -1), 0.7, border),
        ("LINEBEFORE", (0, 0), (0, -1), 3, teal),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(findings_box)

    # 04 - Movement observations
    story.extend(section("04  |  Movement Observations"))

    observations = report.get("movement_observations", [])
    if observations:
        for observation in observations:
            story.append(P("•  " + str(observation)))
    else:
        story.append(P("No movement observations are available."))

    # 05 - Technical measurements
    story.extend(section("05  |  Technical Measurements"))

    if metrics:
        metric_rows = [[
            P("MEASUREMENT", "WhiteHeader"),
            P("RECORDED VALUE", "WhiteHeader"),
        ]]

        metric_labels = {
            "mean_dynamic_acceleration_g": "Mean dynamic acceleration (g)",
            "rms_dynamic_acceleration_g": "RMS dynamic acceleration (g)",
            "dynamic_acceleration_std_g": "Dynamic acceleration standard deviation (g)",
            "mean_rotation_dps": "Mean rotational activity (degrees/second)",
            "rms_rotation_dps": "RMS rotational activity (degrees/second)",
            "low_motion_sample_percentage": "Samples below experimental low-motion threshold (%)",
            "low_motion_threshold_g": "Experimental low-motion threshold (g)",
        }

        for key, value in metrics.items():
            metric_rows.append([
                P(metric_labels.get(key, key), "TableText"),
                P(value, "TableText"),
            ])

        story.append(make_table(metric_rows, widths=[17 * cm, 8.5 * cm]))
    else:
        story.append(P("No technical measurements are available."))

    # 06 - Candidate segments
    story.extend(section("06  |  Candidate Segments for Review"))

    candidate_events = report.get("candidate_events", [])

    if candidate_events:
        event_rows = [[
            P("ID", "WhiteHeader"),
            P("START", "WhiteHeader"),
            P("END", "WhiteHeader"),
            P("DURATION", "WhiteHeader"),
            P("WINDOWS", "WhiteHeader"),
            P("MAX SCORE", "WhiteHeader"),
            P("MEAN SCORE", "WhiteHeader"),
            P("USER REVIEW", "WhiteHeader"),
        ]]

        for event in candidate_events:
            event_rows.append([
                P(event.get("event_id"), "TableText"),
                P(event.get("start_at"), "SmallText"),
                P(event.get("end_at"), "SmallText"),
                P(
                    f"{event.get('duration_seconds', 0):.2f}s"
                    if event.get("duration_seconds") is not None else "—",
                    "TableText",
                ),
                P(event.get("window_count"), "TableText"),
                P(f"{event.get('max_score', 0):.3f}", "TableText"),
                P(f"{event.get('mean_score', 0):.3f}", "TableText"),
                P(event.get("review_status", "awaiting_review"), "TableText"),
            ])

        story.append(make_table(
            event_rows,
            widths=[
                1.2 * cm, 5.0 * cm, 5.0 * cm, 2.0 * cm,
                1.5 * cm, 2.0 * cm, 2.0 * cm, 3.8 * cm,
            ],
        ))
    else:
        story.append(P(
            "No candidate segments were generated. This does not establish "
            "that no movement difficulties occurred."
        ))

    # 07 - User feedback
    story.extend(section("07  |  User Feedback"))

    reviews = report.get("review_counts", {})
    review_rows = [
        [P("REVIEW RESPONSE", "WhiteHeader"), P("NUMBER OF SEGMENTS", "WhiteHeader")],
        [P("Reported as experienced", "TableText"), P(reviews.get("confirmed", 0), "TableText")],
        [P("Not noticed by the user", "TableText"), P(reviews.get("not_observed", 0), "TableText")],
        [P("Uncertain", "TableText"), P(reviews.get("uncertain", 0), "TableText")],
        [P("Awaiting review", "TableText"), P(reviews.get("awaiting_review", 0), "TableText")],
    ]
    story.append(make_table(review_rows, widths=[17 * cm, 8.5 * cm]))

    # 08 - Movement timeline
    story.extend(section("08  |  Five-Second Movement Timeline"))

    timeline = report.get("movement_timeline", [])
    if timeline:
        timeline_rows = [[
            P("START (s)", "WhiteHeader"),
            P("END (s)", "WhiteHeader"),
            P("MEAN ACCELERATION (g)", "WhiteHeader"),
            P("RMS ACCELERATION (g)", "WhiteHeader"),
            P("MEAN ROTATION (degrees/s)", "WhiteHeader"),
        ]]

        for interval in timeline:
            timeline_rows.append([
                P(interval["start_seconds"], "TableText"),
                P(interval["end_seconds"], "TableText"),
                P(interval["mean_dynamic_acceleration_g"], "TableText"),
                P(interval["rms_dynamic_acceleration_g"], "TableText"),
                P(interval["mean_rotation_dps"], "TableText"),
            ])

        story.append(make_table(
            timeline_rows,
            widths=[3 * cm, 3 * cm, 6 * cm, 6 * cm, 7 * cm],
        ))
    else:
        story.append(P("No timeline data is available."))

    # 09 - Research model
    story.extend(section("09  |  Research Model Information"))

    story.append(P(
        f"Mean research model score: {report.get('mean_research_model_score')}"
    ))
    story.append(P(
        f"Maximum research model score: {report.get('max_research_model_score')}"
    ))

    story.append(Spacer(1, 15))
    story.append(P(
        "Generated by StrideSense AI | Experimental research prototype",
        "SmallText",
    ))

    doc.build(story, onFirstPage=page_layout, onLaterPages=page_layout)

    buffer.seek(0)
    filename = f"StrideSense_Report_{session_id[:8]}.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )