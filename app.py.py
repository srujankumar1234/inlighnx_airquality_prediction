"""
Air Quality Prediction – Flask REST API
=======================================
Organisation : INLIGHN TECH
Project      : Air Quality Prediction
Endpoints    :
  GET  /            → Serve the HTML dashboard
  GET  /health      → API health check
  POST /predict     → Predict PM2.5 from input features
  GET  /models      → List available models and last benchmark results
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
import joblib
import json
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

ARTIFACTS_DIR = "model_artifacts"

# ─────────────────────────────────────────
# Load models & scaler at startup
# ─────────────────────────────────────────
def load_artifacts():
    """Load all serialised model artifacts from disk."""
    models = {}
    scaler = None
    feature_cols = []

    if not os.path.isdir(ARTIFACTS_DIR):
        print(f"[WARNING] '{ARTIFACTS_DIR}' folder not found. "
              "Run the notebook first to generate model artifacts.")
        return models, scaler, feature_cols

    model_files = {
        "Random Forest"     : "random_forest.pkl",
        "Gradient Boosting" : "gradient_boosting.pkl",
        "RF Tuned"          : "rf_tuned.pkl",
        "GB Tuned"          : "gb_tuned.pkl",
    }

    for name, filename in model_files.items():
        path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  ✅ Loaded: {name}")
        else:
            print(f"  ⚠️  Missing: {name} ({filename})")

    scaler_path = os.path.join(ARTIFACTS_DIR, "feature_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("  ✅ Loaded: feature_scaler")

    cols_path = os.path.join(ARTIFACTS_DIR, "feature_cols.json")
    if os.path.exists(cols_path):
        with open(cols_path) as f:
            feature_cols = json.load(f)
        print(f"  ✅ Loaded: feature_cols ({len(feature_cols)} features)")

    return models, scaler, feature_cols


MODELS, SCALER, FEATURE_COLS = load_artifacts()


# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def pm25_to_category(val: float) -> str:
    """Map a numeric PM2.5 value to an CPCB AQI category string."""
    if val <= 30:
        return "Good / Satisfactory"
    elif val <= 60:
        return "Moderate"
    elif val <= 90:
        return "Poor"
    elif val <= 120:
        return "Very Poor"
    else:
        return "Severe"


def validate_features(data: dict) -> tuple[dict | None, str | None]:
    """
    Validate that all required features are present and numeric.
    Returns (cleaned_dict, None) on success or (None, error_message).
    """
    cleaned = {}
    for col in FEATURE_COLS:
        if col not in data:
            return None, f"Missing required feature: '{col}'"
        try:
            cleaned[col] = float(data[col])
        except (ValueError, TypeError):
            return None, f"Feature '{col}' must be a numeric value."
    return cleaned, None


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the HTML frontend from templates/index.html."""
    html_path = os.path.join("templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return "<h2>templates/index.html not found. Please add it.</h2>", 404


@app.route("/health", methods=["GET"])
def health():
    """API health check endpoint."""
    return jsonify({
        "status"       : "ok",
        "models_loaded": list(MODELS.keys()),
        "scaler_loaded": SCALER is not None,
        "feature_count": len(FEATURE_COLS),
    })


@app.route("/models", methods=["GET"])
def list_models():
    """Return available models and benchmark results (if saved)."""
    results_path = os.path.join(ARTIFACTS_DIR, "model_results.csv")
    benchmark = []

    if os.path.exists(results_path):
        import pandas as pd
        df = pd.read_csv(results_path)
        benchmark = df.to_dict(orient="records")

    return jsonify({
        "available_models": list(MODELS.keys()),
        "features"        : FEATURE_COLS,
        "benchmark"       : benchmark,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accept JSON with feature values and return PM2.5 predictions
    from all loaded models.

    Request body example:
    {
        "co": 900, "no": 8.5, "no2": 32, "o3": 25, "so2": 18,
        "pm10": 120, "nh3": 12, "aqi": 3,
        "day_of_week": 2, "month": 11, "quarter": 4, "day_of_year": 320,
        "pm2_5_lag_1": 95, "pm2_5_lag_7": 88, "pm2_5_lag_30": 75,
        "pm2_5_roll7_mean": 91, "pm2_5_roll7_std": 10.2, "pm2_5_roll30_mean": 82
    }
    """
    if not MODELS:
        return jsonify({
            "error": "No models loaded. Please run the notebook first to generate model artifacts."
        }), 503

    if SCALER is None:
        return jsonify({"error": "Feature scaler not loaded."}), 503

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON payload."}), 400

    # Validate & clean input features
    cleaned, err = validate_features(data)
    if err:
        return jsonify({"error": err}), 400

    # Build feature vector in the correct column order
    input_vec = np.array([[cleaned[col] for col in FEATURE_COLS]])
    input_scaled = SCALER.transform(input_vec)

    # Run inference through all available models
    predictions = {}
    for name, model in MODELS.items():
        try:
            pred_val = float(model.predict(input_scaled)[0])
            predictions[name] = {
                "pm2_5"   : round(pred_val, 2),
                "category": pm25_to_category(pred_val),
            }
        except Exception as e:
            predictions[name] = {"error": str(e)}

    # Recommend the best model (GB Tuned if available, else first available)
    recommended_model = "GB Tuned" if "GB Tuned" in predictions else list(predictions.keys())[0]
    recommended_pred  = predictions[recommended_model]

    return jsonify({
        "input_features"  : cleaned,
        "predictions"     : predictions,
        "recommended"     : {
            "model"    : recommended_model,
            "pm2_5"    : recommended_pred.get("pm2_5"),
            "category" : recommended_pred.get("category"),
        },
    })


@app.route("/predict/dummy", methods=["GET"])
def predict_dummy():
    """
    GET /predict/dummy
    Run predictions on the three built-in synthetic test scenarios
    and return results for all loaded models.
    """
    if not MODELS or SCALER is None:
        return jsonify({"error": "Models or scaler not loaded."}), 503

    dummy_inputs = [
        {
            "label"            : "Moderate Pollution Day",
            "co": 900.0, "no": 8.5, "no2": 32.0, "o3": 25.0, "so2": 18.0,
            "pm10": 120.0, "nh3": 12.0, "aqi": 3,
            "day_of_week": 2, "month": 11, "quarter": 4, "day_of_year": 320,
            "pm2_5_lag_1": 95.0, "pm2_5_lag_7": 88.0, "pm2_5_lag_30": 75.0,
            "pm2_5_roll7_mean": 91.0, "pm2_5_roll7_std": 10.2, "pm2_5_roll30_mean": 82.0,
        },
        {
            "label"            : "Low Pollution Day (Clear Weather)",
            "co": 420.0, "no": 2.0, "no2": 14.0, "o3": 40.0, "so2": 6.0,
            "pm10": 35.0, "nh3": 4.0, "aqi": 1,
            "day_of_week": 6, "month": 7, "quarter": 3, "day_of_year": 195,
            "pm2_5_lag_1": 28.0, "pm2_5_lag_7": 31.0, "pm2_5_lag_30": 42.0,
            "pm2_5_roll7_mean": 30.0, "pm2_5_roll7_std": 4.5, "pm2_5_roll30_mean": 38.0,
        },
        {
            "label"            : "High Pollution Spike (Smog/Festival)",
            "co": 2800.0, "no": 25.0, "no2": 80.0, "o3": 10.0, "so2": 55.0,
            "pm10": 380.0, "nh3": 38.0, "aqi": 5,
            "day_of_week": 4, "month": 10, "quarter": 4, "day_of_year": 300,
            "pm2_5_lag_1": 310.0, "pm2_5_lag_7": 180.0, "pm2_5_lag_30": 120.0,
            "pm2_5_roll7_mean": 250.0, "pm2_5_roll7_std": 60.0, "pm2_5_roll30_mean": 160.0,
        },
    ]

    output = []
    for scenario in dummy_inputs:
        label = scenario.pop("label")
        input_vec    = np.array([[scenario[col] for col in FEATURE_COLS]])
        input_scaled = SCALER.transform(input_vec)

        model_preds = {}
        for name, model in MODELS.items():
            pred_val = float(model.predict(input_scaled)[0])
            model_preds[name] = {
                "pm2_5"   : round(pred_val, 2),
                "category": pm25_to_category(pred_val),
            }

        output.append({"scenario": label, "predictions": model_preds})

    return jsonify({"dummy_predictions": output})


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌫️  Air Quality Prediction API")
    print("================================")
    print(f"  Models loaded : {list(MODELS.keys())}")
    print(f"  Features      : {len(FEATURE_COLS)}")
    print(f"  Running on    : http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
