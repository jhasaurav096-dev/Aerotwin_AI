import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def generate_cmapss_inspired_dataset(num_engines=60, seed=42):
    rng = np.random.default_rng(seed)
    records = []

    for engine_id in range(1, num_engines + 1):
        max_cycles = rng.integers(160, 260)
        wear_rate = rng.uniform(0.008, 0.015)

        for cycle in range(1, max_cycles + 1):
            rul = max_cycles - cycle
            degradation = np.exp(wear_rate * cycle) - 1.0

            n2_rpm = 90.0 + rng.normal(0, 1.2)
            fuel_flow = 2350.0 + (n2_rpm - 90.0) * 45.0 + degradation * 35.0 + rng.normal(0, 15.0)
            expected_egt = 310.0 + (4.15 * n2_rpm) + (0.048 * fuel_flow)
            egt = expected_egt + degradation * 16.5 + rng.normal(0, 3.0)
            vibration = 0.35 + degradation * 0.42 + abs(rng.normal(0, 0.05))
            oil_pressure = max(32.0, 64.0 - degradation * 3.8 + rng.normal(0, 0.8))

            records.append({
                "engine_id": engine_id,
                "cycle": cycle,
                "egt_celsius": round(egt, 2),
                "n2_rpm_pct": round(n2_rpm, 2),
                "fuel_flow_pph": round(fuel_flow, 2),
                "vibration_ips": round(vibration, 3),
                "oil_pressure_psi": round(oil_pressure, 2),
                "egt_residual": round(egt - expected_egt, 2),
                "rul": rul
            })

    return pd.DataFrame(records)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Generating physics-informed turbofan degradation dataset...")
    df = generate_cmapss_inspired_dataset()
    df.to_csv(os.path.join(artifacts_dir, "synthetic_turbofan_telemetry.csv"), index=False)

    features = [
        "egt_celsius", "n2_rpm_pct", "fuel_flow_pph",
        "vibration_ips", "oil_pressure_psi", "egt_residual"
    ]
    X = df[features]
    y = df["rul"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Remaining Useful Life (RUL) Random Forest Regressor...")
    rul_model = RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)
    rul_model.fit(X_train, y_train)

    preds = rul_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"RUL Validation MAE: {mae:.2f} cycles | R2 Score: {r2:.4f}")

    print("Training Isolation Forest Anomaly Detector on healthy baseline cycles...")
    healthy_X = df[df["rul"] > 90][features]
    anomaly_model = IsolationForest(contamination=0.04, random_state=42)
    anomaly_model.fit(healthy_X)

    joblib.dump(rul_model, os.path.join(artifacts_dir, "rul_rf_model.joblib"))
    joblib.dump(anomaly_model, os.path.join(artifacts_dir, "anomaly_iforest.joblib"))
    print(f"Saved model artifacts to {artifacts_dir}")

if __name__ == "__main__":
    main()
