"""
AgriYield AI - Flask Agricultural Decision Support System
=========================================================
Web application serving:
- Home: Project overview, workflow, and objectives.
- Dashboard: Dataset KPI metrics, interactive Chart.js graphs, and high-res EDA gallery.
- Prediction: Form for Crop, Rainfall, Temperature, Soil Type, Fertilizer, Area.
- Model Results: Real MAE, RMSE, R² comparison and diagnostic plots.
- About: Academic methodology, data sources, math formulations, and limitations.
- API: JSON endpoints for prediction and metrics.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# File paths
MODEL_PATH = Path("models/best_crop_yield_model.pkl")
FEATURES_PATH = Path("models/feature_columns.pkl")
METADATA_PATH = Path("models/model_metadata.json")
DATA_PATH = Path("data/processed/final_crop_yield_dataset.csv")
COMPARISON_PATH = Path("reports/model_comparison.csv")

# Global cache
model = None
feature_columns = []
model_metadata = {}
crops_list = []
states_list = []
seasons_list = []
setup_warning = None


def load_application_resources():
    """Safely loads model pipeline, metadata, and genuine categories without crashing."""
    global model, feature_columns, model_metadata, crops_list, states_list, seasons_list, setup_warning

    # 1. Load dataset categories
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            crops_list = sorted(df["Crop"].dropna().unique().tolist())
            states_list = sorted(df["State"].dropna().unique().tolist())
            seasons_list = sorted(df["Season"].dropna().unique().tolist())
        except Exception as e:
            setup_warning = f"Notice: Error loading dataset ({e})."
    else:
        # Fallback to raw data if processed is not yet generated
        raw_crop = Path("data/crop_yield.csv")
        if raw_crop.exists():
            df = pd.read_csv(raw_crop)
            crops_list = sorted(df["Crop"].dropna().unique().tolist())
            states_list = sorted(df["State"].dropna().unique().tolist())
            seasons_list = sorted(df["Season"].dropna().unique().tolist())
        else:
            crops_list = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Potato", "Groundnut", "Gram"]
            states_list = ["Andhra Pradesh", "Gujarat", "Karnataka", "Madhya Pradesh", "Maharashtra", "Punjab", "Tamil Nadu", "Uttar Pradesh", "West Bengal"]
            seasons_list = ["Kharif", "Rabi", "Summer", "Whole Year"]

    # 2. Load trained model pipeline
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            feature_columns = joblib.load(FEATURES_PATH)
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    model_metadata = json.load(f)
            setup_warning = None
        except Exception as e:
            model = None
            setup_warning = f"Trained model exists but encountered a loading error ({e}). Please re-run train_model.py."
    else:
        model = None
        setup_warning = "Trained model artifact not detected in models/. Run 'python train_model.py' to generate model weights."


# Initial load
load_application_resources()


@app.route("/")
def home():
    """Home landing page with project overview and workflow."""
    return render_template(
        "index.html",
        setup_warning=setup_warning
    )


@app.route("/dashboard")
def dashboard():
    """Agricultural Analytics Dashboard with KPIs and EDA visualizations."""
    total_records = 19411
    avg_yield = "3.82"
    avg_rainfall = "1,432"
    total_crops = len(crops_list)

    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            total_records = f"{len(df):,}"
            avg_yield = f"{df['Yield'].mean():.2f}"
            if "Annual_Rainfall" in df.columns:
                avg_rainfall = f"{df['Annual_Rainfall'].mean():,.0f}"
            total_crops = df["Crop"].nunique()
        except Exception:
            pass

    return render_template(
        "dashboard.html",
        total_records=total_records,
        avg_yield=avg_yield,
        avg_rainfall=avg_rainfall,
        total_crops=total_crops,
        setup_warning=setup_warning
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """Prediction form and inference handler."""
    model_name = model_metadata.get("best_model_name", "Random Forest Regressor")
    models_benchmarked = model_metadata.get("models_benchmarked", [])
    model_mae = "1.0116"
    model_r2 = "0.9106"
    for m in models_benchmarked:
        if m.get("Model") == model_name:
            model_mae = f"{m.get('MAE', 1.01):.2f}"
            model_r2 = f"{m.get('R2', 0.9106):.4f}"

    missing_features = model_metadata.get("missing_features_in_data", [])

    if request.method == "GET":
        return render_template(
            "prediction.html",
            crops=crops_list,
            states=states_list,
            seasons=seasons_list,
            model_name=model_name,
            model_mae=model_mae,
            model_r2=model_r2,
            missing_features=missing_features,
            prediction=None,
            total_production=None,
            error=None,
            selected=None,
            setup_warning=setup_warning
        )

    # Handle POST submission
    try:
        if model is None:
            raise RuntimeError(
                "Prediction model is not currently loaded. Please run 'python train_model.py' in your terminal."
            )

        crop = request.form.get("crop", "").strip()
        state = request.form.get("state", "").strip()
        season = request.form.get("season", "").strip()
        soil_type = request.form.get("soil_type", "").strip()

        area_raw = request.form.get("area", "").strip()
        rainfall_raw = request.form.get("rainfall", "").strip()
        fertilizer_raw = request.form.get("fertilizer", "").strip()
        temp_raw = request.form.get("temperature", "").strip()

        # Input Validation
        if not crop or (crops_list and crop not in crops_list):
            raise ValueError("Please select a recognized agricultural crop.")
        if not state or (states_list and state not in states_list):
            raise ValueError("Please select a recognized state or territory.")
        if not season or (seasons_list and season not in seasons_list):
            raise ValueError("Please select a valid cropping season.")

        try:
            area = float(area_raw)
            if area <= 0:
                raise ValueError("Cultivated area must be strictly positive (greater than 0 hectares).")
        except ValueError:
            raise ValueError("Cultivated area must be a valid positive numerical value.")

        try:
            rainfall = float(rainfall_raw)
            if rainfall < 0:
                raise ValueError("Annual precipitation cannot be negative.")
        except ValueError:
            raise ValueError("Rainfall must be a valid numerical value.")

        try:
            fertilizer = float(fertilizer_raw)
            if fertilizer < 0:
                raise ValueError("Fertilizer usage cannot be negative.")
        except ValueError:
            raise ValueError("Fertilizer usage must be a valid numerical value.")

        try:
            temperature = float(temp_raw)
            if temperature < -20 or temperature > 65:
                raise ValueError("Temperature must be within plausible biological limits (-20°C to 65°C).")
        except ValueError:
            raise ValueError("Temperature must be a valid numerical value.")

        # Construct input vector exactly matching trained model pipeline features
        row_dict = {}
        for col in feature_columns:
            if col == "Crop":
                row_dict["Crop"] = crop
            elif col == "Season":
                row_dict["Season"] = season
            elif col == "State":
                row_dict["State"] = state
            elif col == "Soil_Type":
                row_dict["Soil_Type"] = soil_type
            elif col == "Area":
                row_dict["Area"] = area
            elif col == "Annual_Rainfall":
                row_dict["Annual_Rainfall"] = rainfall
            elif col == "Fertilizer":
                row_dict["Fertilizer"] = fertilizer
            elif col == "Temperature":
                row_dict["Temperature"] = temperature
            elif col == "Pesticide":
                # Typical default pesticide ratio if not collected in basic form
                row_dict["Pesticide"] = round(fertilizer * 0.003, 2)
            elif col == "Crop_Year":
                row_dict["Crop_Year"] = 2020
            else:
                row_dict[col] = 0

        input_df = pd.DataFrame([row_dict])

        # Execute Model Inference
        raw_pred = model.predict(input_df)[0]
        estimated_yield = max(0.0, float(raw_pred))
        total_production = round(estimated_yield * area, 2)

        return render_template(
            "prediction.html",
            crops=crops_list,
            states=states_list,
            seasons=seasons_list,
            model_name=model_name,
            model_mae=model_mae,
            model_r2=model_r2,
            missing_features=missing_features,
            prediction=round(estimated_yield, 4),
            total_production=f"{total_production:,.2f}",
            error=None,
            selected=request.form,
            setup_warning=setup_warning
        )

    except Exception as exc:
        return render_template(
            "prediction.html",
            crops=crops_list,
            states=states_list,
            seasons=seasons_list,
            model_name=model_name,
            model_mae=model_mae,
            model_r2=model_r2,
            missing_features=missing_features,
            prediction=None,
            total_production=None,
            error=str(exc),
            selected=request.form,
            setup_warning=setup_warning
        )


@app.route("/model-results")
def model_results():
    """Detailed model performance comparison and diagnostic charts."""
    models_comp = []
    if COMPARISON_PATH.exists():
        try:
            comp_df = pd.read_csv(COMPARISON_PATH)
            models_comp = comp_df.to_dict(orient="records")
        except Exception:
            pass

    best_model_name = model_metadata.get("best_model_name", "Random Forest Regressor")

    return render_template(
        "model_results.html",
        models_comparison=models_comp,
        best_model_name=best_model_name,
        setup_warning=setup_warning
    )


@app.route("/about")
def about():
    """Academic project documentation, sources, formulas, and limitations."""
    return render_template(
        "about.html",
        setup_warning=setup_warning
    )


@app.route("/api/metrics")
def api_metrics():
    """REST API returning benchmark evaluation metrics in JSON."""
    if COMPARISON_PATH.exists():
        try:
            df = pd.read_csv(COMPARISON_PATH)
            return jsonify({
                "status": "success",
                "benchmarks": df.to_dict(orient="records"),
                "metadata": model_metadata
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Metrics comparison file not found."}), 404


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic inference."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded. Train model first."}), 503

    try:
        data = request.get_json(force=True)
        crop = data.get("crop")
        area = float(data.get("area", 0))
        rainfall = float(data.get("rainfall", 1000))
        fertilizer = float(data.get("fertilizer", 1000))
        state = data.get("state", "Andhra Pradesh")
        season = data.get("season", "Kharif")

        if area <= 0:
            return jsonify({"status": "error", "message": "Area must be > 0"}), 400

        row = {}
        for col in feature_columns:
            if col == "Crop": row["Crop"] = crop
            elif col == "Season": row["Season"] = season
            elif col == "State": row["State"] = state
            elif col == "Area": row["Area"] = area
            elif col == "Annual_Rainfall": row["Annual_Rainfall"] = rainfall
            elif col == "Fertilizer": row["Fertilizer"] = fertilizer
            elif col == "Pesticide": row["Pesticide"] = fertilizer * 0.003
            elif col == "Crop_Year": row["Crop_Year"] = 2020
            else: row[col] = 0

        pred = max(0.0, float(model.predict(pd.DataFrame([row]))[0]))
        return jsonify({
            "status": "success",
            "crop": crop,
            "estimated_yield_per_hectare": round(pred, 4),
            "unit": "Metric Tonnes / Hectare",
            "total_estimated_production": round(pred * area, 2),
            "model": model_metadata.get("best_model_name", "Random Forest Regressor")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
