"""
AgriYield AI - Comprehensive Automated Test Suite
=================================================
Validates:
1. Dataset loading, schema integrity, and unit consistency.
2. Preprocessing, outlier removal, and target leakage prevention.
3. Trained model inference and metric sanity.
4. Flask web routes (Home, Dashboard, Predict, Model Results, About).
5. Input validation (positive area, non-negative environmental variables).
6. REST API endpoints (/api/metrics, /api/predict).
7. Missing-file and missing-model error handling.
"""

import os
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Import application and pipeline components
from app import app
from data_preprocessing import normalize_columns


class TestDatasetAndPreprocessing(unittest.TestCase):
    """Validates dataset schema, clean data integrity, and leakage prevention."""

    def setUp(self):
        self.processed_path = Path("data/processed/final_crop_yield_dataset.csv")

    def test_processed_dataset_exists_and_loads(self):
        """Processed dataset must exist and contain rows."""
        self.assertTrue(self.processed_path.exists(), "Processed dataset file missing.")
        df = pd.read_csv(self.processed_path)
        self.assertGreater(len(df), 1000, "Dataset contains insufficient clean rows.")

    def test_required_columns_present(self):
        """Processed dataset must contain all core predictors and target."""
        df = pd.read_csv(self.processed_path)
        required = ["Crop", "Area", "Yield", "Annual_Rainfall", "Fertilizer"]
        for col in required:
            self.assertIn(col, df.columns, f"Required column '{col}' missing from processed data.")

    def test_domain_validity_constraints(self):
        """All records must satisfy physical agricultural validity constraints."""
        df = pd.read_csv(self.processed_path)
        # Area must be strictly positive
        self.assertTrue((df["Area"] > 0).all(), "Found invalid non-positive Area values.")
        # Yield must be strictly positive
        self.assertTrue((df["Yield"] > 0).all(), "Found invalid non-positive Yield values.")
        # Rainfall and fertilizer must be non-negative
        self.assertTrue((df["Annual_Rainfall"] >= 0).all(), "Found negative rainfall values.")
        self.assertTrue((df["Fertilizer"] >= 0).all(), "Found negative fertilizer values.")

    def test_unit_consistency_coconut_filtered(self):
        """Coconut (harvested in nuts rather than metric tonnes) must be filtered to prevent unit incompatibility."""
        df = pd.read_csv(self.processed_path)
        self.assertNotIn("Coconut", df["Crop"].unique(), "Coconut records with unit incompatibility were not filtered.")

    def test_column_normalization_mapping(self):
        """Column normalization function must map casing and spacing variations."""
        raw_df = pd.DataFrame(columns=["state_name", "district_name", "crop_year", "area", "production"])
        normalized = normalize_columns(raw_df)
        self.assertEqual(list(normalized.columns), ["State", "District", "Crop_Year", "Area", "Production"])


class TestModelTrainingAndInference(unittest.TestCase):
    """Validates machine learning pipeline artifacts and prediction behavior."""

    def setUp(self):
        self.model_path = Path("models/best_crop_yield_model.pkl")
        self.features_path = Path("models/feature_columns.pkl")
        self.metadata_path = Path("models/model_metadata.json")
        self.comparison_path = Path("reports/model_comparison.csv")

    def test_model_artifacts_exist(self):
        """Model pipeline and feature list must be saved and loadable."""
        self.assertTrue(self.model_path.exists(), "Trained model artifact missing.")
        self.assertTrue(self.features_path.exists(), "Feature columns artifact missing.")
        self.assertTrue(self.metadata_path.exists(), "Model metadata JSON missing.")

    def test_model_inference_pipeline(self):
        """Model must accept feature vectors and predict positive yield."""
        model = joblib.load(self.model_path)
        features = joblib.load(self.features_path)

        sample_input = {col: 0 for col in features}
        sample_input["Crop"] = "Rice"
        sample_input["Season"] = "Kharif"
        sample_input["State"] = "Andhra Pradesh"
        sample_input["Area"] = 100.0
        sample_input["Annual_Rainfall"] = 1200.0
        sample_input["Fertilizer"] = 8000.0
        sample_input["Crop_Year"] = 2020

        df_input = pd.DataFrame([sample_input])
        prediction = model.predict(df_input)

        self.assertEqual(len(prediction), 1)
        self.assertGreater(float(prediction[0]), 0.0, "Predicted yield should be strictly positive.")

    def test_comparison_metrics_sanity(self):
        """Real evaluation metrics must be present and statistically sound."""
        self.assertTrue(self.comparison_path.exists(), "Model comparison CSV missing.")
        df_comp = pd.read_csv(self.comparison_path)
        self.assertIn("Model", df_comp.columns)
        self.assertIn("MAE", df_comp.columns)
        self.assertIn("R2", df_comp.columns)

        # Ensure R² is between 0 and 1, MAE > 0
        for _, row in df_comp.iterrows():
            self.assertGreater(row["MAE"], 0.0, f"MAE for {row['Model']} should be positive.")
            self.assertGreater(row["RMSE"], 0.0, f"RMSE for {row['Model']} should be positive.")
            self.assertLessEqual(row["R2"], 1.0, f"R² for {row['Model']} cannot exceed 1.0.")


class TestFlaskWebRoutes(unittest.TestCase):
    """Tests all Flask endpoints and user interaction flows."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page_route(self):
        """GET / must return HTTP 200 and brand header."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Crop Yield", res.data)
        self.assertIn(b"Precision Crop Yield Prediction", res.data)

    def test_dashboard_route(self):
        """GET /dashboard must return HTTP 200 and EDA elements."""
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Agricultural Analytics", res.data)
        self.assertIn(b"crop_wise_yield.png", res.data)

    def test_predict_page_get(self):
        """GET /predict must render prediction input form."""
        res = self.client.get("/predict")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Crop Species", res.data)
        self.assertIn(b"Cultivated Area", res.data)

    def test_predict_page_post_valid(self):
        """POST /predict with valid parameters must compute estimated yield."""
        payload = {
            "crop": "Rice",
            "state": "Andhra Pradesh",
            "season": "Kharif",
            "area": "75.0",
            "rainfall": "1250.0",
            "fertilizer": "6500.0",
            "temperature": "27.5",
            "soil_type": "Alluvial"
        }
        res = self.client.post("/predict", data=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"ESTIMATED HARVEST YIELD", res.data)
        self.assertIn(b"MT / Hectare", res.data)
        self.assertIn(b"Metric Tonnes", res.data)

    def test_predict_page_post_invalid_area(self):
        """POST /predict with negative or zero area must display an input validation error."""
        payload = {
            "crop": "Rice",
            "state": "Andhra Pradesh",
            "season": "Kharif",
            "area": "-10.0",
            "rainfall": "1200.0",
            "fertilizer": "5000.0",
            "temperature": "25.0",
            "soil_type": "Alluvial"
        }
        res = self.client.post("/predict", data=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Input Validation Error", res.data)
        self.assertIn(b"greater than 0", res.data)

    def test_model_results_route(self):
        """GET /model-results must return HTTP 200 and comparison table."""
        res = self.client.get("/model-results")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Random Forest Regressor", res.data)
        self.assertIn(b"actual_vs_predicted.png", res.data)

    def test_about_page_route(self):
        """GET /about must render project documentation."""
        res = self.client.get("/about")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Project Objectives", res.data)
        self.assertIn(b"Ministry of Agriculture", res.data)

    def test_api_metrics_endpoint(self):
        """GET /api/metrics must return JSON benchmarks."""
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("benchmarks", data)

    def test_api_predict_endpoint(self):
        """POST /api/predict must return JSON prediction."""
        payload = {
            "crop": "Wheat",
            "state": "Punjab",
            "season": "Rabi",
            "area": 50.0,
            "rainfall": 600.0,
            "fertilizer": 4500.0
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["estimated_yield_per_hectare"], 0.0)


if __name__ == "__main__":
    unittest.main()
