import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AeroTwin AI Digital Twin API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "python_ml", "artifacts")
RUL_PATH = os.path.join(ARTIFACTS_DIR, "rul_rf_model.joblib")
ANOMALY_PATH = os.path.join(ARTIFACTS_DIR, "anomaly_iforest.joblib")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

rul_model = joblib.load(RUL_PATH) if os.path.exists(RUL_PATH) else None
anomaly_model = joblib.load(ANOMALY_PATH) if os.path.exists(ANOMALY_PATH) else None

class TelemetryInput(BaseModel):
    egt_celsius: float = Field(..., ge=300, le=1000)
    n2_rpm_pct: float = Field(..., ge=50, le=110)
    fuel_flow_pph: float = Field(..., ge=500, le=5000)
    vibration_ips: float = Field(..., ge=0.0, le=5.0)
    oil_pressure_psi: float = Field(..., ge=10.0, le=100.0)

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "online",
        "ml_models_loaded": rul_model is not None and anomaly_model is not None
    }

@app.post("/api/v1/twin/evaluate")
def evaluate_digital_twin(data: TelemetryInput):
    expected_egt = 310.0 + (4.15 * data.n2_rpm_pct) + (0.048 * data.fuel_flow_pph)
    egt_residual = data.egt_celsius - expected_egt

    features = np.array([[
        data.egt_celsius,
        data.n2_rpm_pct,
        data.fuel_flow_pph,
        data.vibration_ips,
        data.oil_pressure_psi,
        egt_residual
    ]])

    if rul_model is not None and anomaly_model is not None:
        predicted_rul = float(rul_model.predict(features)[0])
        anomaly_flag = int(anomaly_model.predict(features)[0]) == -1
    else:
        penalty = max(0.0, egt_residual * 1.4) + max(0.0, (data.vibration_ips - 0.5) * 45.0)
        predicted_rul = max(0.0, 210.0 - penalty)
        anomaly_flag = egt_residual > 28.0 or data.vibration_ips > 1.3

    health_score = max(0.0, min(100.0, (predicted_rul / 210.0) * 100.0))
    if anomaly_flag and health_score > 65.0:
        health_score = 62.0

    if health_score >= 78.0 and not anomaly_flag:
        status = "NOMINAL"
        recommendation = "Continue standard flight profile. Next scheduled borescope at standard interval."
    elif health_score >= 48.0:
        status = "ADVISORY"
        recommendation = "Elevated thermal/vibration residual detected. Schedule trend diagnostics at next hub."
    else:
        status = "CRITICAL_ALERT"
        recommendation = "High surge/degradation risk. Reduce throttle setting and inspect HPT blades immediately."

    return {
        "expected_egt_celsius": round(expected_egt, 2),
        "egt_residual_celsius": round(egt_residual, 2),
        "predicted_rul_cycles": round(predicted_rul, 1),
        "health_index_pct": round(health_score, 1),
        "anomaly_detected": anomaly_flag,
        "status": status,
        "recommendation": recommendation
    }


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")
