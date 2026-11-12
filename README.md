# AeroTwin AI — Intelligent Aircraft Engine Digital Twin (macOS VS Code Edition)

A physics-informed AI/ML and C++17 embedded telemetry digital twin for aircraft turbofan health monitoring, Remaining Useful Life (RUL) forecasting, and thermal surge detection.

## Quick Start in VS Code on macOS

1. **Prerequisites:** Ensure Xcode Command Line Tools (`xcode-select --install`) and Python 3.9+ are installed on your Mac.
2. **Open Project:** Unzip this folder and open `AeroTwin_AI_macOS` in **Visual Studio Code**.
3. **Run One-Click Tasks (`Cmd + Shift + P` -> `Tasks: Run Task`):**
   - **Step 1:** Run `Setup macOS Python Virtual Environment` (creates `.venv` and installs FastAPI, Scikit-learn, Pandas, NumPy).
   - **Step 2:** Run `Train AeroTwin ML Models` (generates synthetic turbofan cycle data and saves `.joblib` models).
   - **Step 3:** Run `Build & Run C++ Telemetry Engine (macOS)` (compiles with Apple `clang++ -std=c++17` and benchmarks microsecond telemetry latency).
   - **Step 4:** Run `Start FastAPI Digital Twin Backend` (starts live API on `http://127.0.0.1:8000`).
   - **Step 5:** Run `Launch Avionics Dashboard in Browser` (opens `frontend/index.html` in Safari/Chrome).

## Project Structure
- `.vscode/` — Pre-configured macOS `settings.json`, `tasks.json`, `launch.json` (LLDB & debugpy), and `extensions.json`.
- `cpp_engine/telemetry_parser.cpp` — Low-latency C++17 sensor parser and exponential state filter.
- `python_ml/train_rul_model.py` — Random Forest RUL regressor and Isolation Forest anomaly detector.
- `backend/main.py` — FastAPI REST inference service combining thermodynamic physics with ML predictions.
- `frontend/index.html` — Interactive flight-deck engine health & fault-injection dashboard.


## Deploy on Render

This repository includes a `render.yaml` Blueprint for the FastAPI service and dashboard. The build installs Python requirements and creates the synthetic demo models; the service serves the frontend at `/` and the API at `/api/v1`.

1. Push this project to a GitHub repository.
2. In Render, choose **New + → Blueprint**, connect that repository, and apply the `render.yaml` configuration.
3. After deployment, open the Render URL. The free instance may sleep after inactivity and take a little longer to respond on its first request.

The live RUL/anomaly output is trained from synthetic demonstration data, not real aircraft telemetry. The C++ telemetry executable remains a local command-line companion and is not part of the hosted web service.
