"""
Crop Yield Prediction - Machine Learning Model Training Module
==============================================================
Trains and benchmarks two regression algorithms:
1. Linear Regression (Baseline)
2. Random Forest Regressor (Non-linear Ensemble)

Key Highlights:
- Reproducible train/test split (random_state=42).
- Rigorous ColumnTransformer pipeline with OneHotEncoder and StandardScaler.
- Target leakage prevention: Production is strictly excluded from predictor features.
- Cross-validation on training partition.
- Real held-out test evaluation (MAE, RMSE, R²).
- Exports best model pipeline, feature schema, and reports/model_comparison.csv.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_models():
    data_file = Path("data/processed/final_crop_yield_dataset.csv")
    model_dir = Path("models")
    reports_dir = Path("reports")
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {data_file}. Please run data_preprocessing.py first."
        )

    print("=" * 70)
    print("   CROP YIELD PREDICTION: MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    df = pd.read_csv(data_file)
    print(f"Loaded processed dataset: {len(df):,} records.")

    # Target variable
    target_col = "Yield"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Determine predictor features dynamically based on verified data presence
    # Target leakage strictly prevented: Production is EXCLUDED.
    candidate_categoricals = ["Crop", "Season", "State", "Soil_Type"]
    candidate_numericals = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Temperature", "Crop_Year"]

    categorical_features = [col for col in candidate_categoricals if col in df.columns]
    numerical_features = [col for col in candidate_numericals if col in df.columns]

    feature_columns = categorical_features + numerical_features

    print(f"\nTarget Variable: {target_col} (Metric Tonnes / Hectare)")
    print(f"Selected Predictors ({len(feature_columns)} features):")
    print(f"  - Categorical: {categorical_features}")
    print(f"  - Numerical  : {numerical_features}")

    # Note regarding Soil_Type and Temperature
    missing_desired = []
    if "Soil_Type" not in df.columns:
        missing_desired.append("Soil_Type (requires data/raw/soil.csv)")
    if "Temperature" not in df.columns:
        missing_desired.append("Temperature (requires data/raw/weather.csv)")
    if missing_desired:
        print(f"  - Notice: The following optional features are not in the raw data and were not trained: {missing_desired}")

    X = df[feature_columns].copy()
    y = df[target_col].copy()

    # Reproducible Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    print(f"\nDataset Partitioning:")
    print(f"  - Training Set: {len(X_train):,} samples")
    print(f"  - Test Set    : {len(X_test):,} samples")

    # Preprocessing pipelines
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Model definitions
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_pipelines = {}

    for name, model in algorithms.items():
        print(f"\nFitting {name} pipeline...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        # 5-fold cross-validation on training data for stability verification
        print("  -> Computing 5-fold cross-validation R² on training set...")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        mean_cv_r2 = float(np.mean(cv_scores))
        std_cv_r2 = float(np.std(cv_scores))
        print(f"  -> Cross-Validation R²: {mean_cv_r2:.4f} (± {std_cv_r2:.4f})")

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Evaluate on held-out test set
        y_pred = pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        print(f"  -> Test MAE : {mae:.4f} MT/ha")
        print(f"  -> Test RMSE: {rmse:.4f} MT/ha")
        print(f"  -> Test R²  : {r2:.4f}")

        results.append({
            "Model": name,
            "CV_R2_Mean": round(mean_cv_r2, 4),
            "CV_R2_Std": round(std_cv_r2, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        })

    # Save model comparison table
    results_df = pd.DataFrame(results)
    comparison_csv = reports_dir / "model_comparison.csv"
    results_df.to_csv(comparison_csv, index=False)
    # Also save to data/ for backward compatibility
    results_df.to_csv("data/model_comparison_results.csv", index=False)
    print(f"\nSaved model comparison metrics to: {comparison_csv}")

    # Select best model based on held-out test MAE and R²
    # Sort by MAE ascending
    best_record = results_df.sort_values(by="MAE", ascending=True).iloc[0]
    best_name = best_record["Model"]
    best_pipeline = trained_pipelines[best_name]

    print(f"\n=======================================================")
    print(f"   WINNING MODEL SELECTED: {best_name}")
    print(f"   Test MAE: {best_record['MAE']} | RMSE: {best_record['RMSE']} | R²: {best_record['R2']}")
    print(f"=======================================================")

    # Serialize trained pipelines
    joblib.dump(best_pipeline, model_dir / "best_crop_yield_model.pkl")
    joblib.dump(trained_pipelines["Linear Regression"], model_dir / "linear_regression_pipeline.pkl")
    joblib.dump(trained_pipelines["Random Forest Regressor"], model_dir / "random_forest_pipeline.pkl")
    joblib.dump(feature_columns, model_dir / "feature_columns.pkl")

    # Serialize test partition for independent evaluation
    test_data_path = model_dir / "test_data.pkl"
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_data_path)

    # Save metadata JSON
    metadata = {
        "best_model_name": best_name,
        "target_variable": target_col,
        "target_unit": "Metric Tonnes / Hectare (MT/ha)",
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "missing_features_in_data": missing_desired,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "models_benchmarked": results,
        "random_state": 42
    }

    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved artifacts to:")
    print(f"  - {model_dir / 'best_crop_yield_model.pkl'}")
    print(f"  - {model_dir / 'feature_columns.pkl'}")
    print(f"  - {metadata_path}")

    return results_df


if __name__ == "__main__":
    train_models()
