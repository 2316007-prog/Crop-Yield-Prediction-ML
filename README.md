# 🌱 Crop Yield Prediction Using Machine Learning (AgriYield AI)

A complete, production-grade final-year / VAC engineering project that forecasts agricultural crop yield per hectare (**Metric Tonnes / Hectare**) using supervised machine learning algorithms (*Linear Regression* and *Random Forest Regressor*), empirical agro-meteorological records, and an interactive 5-page Flask analytics dashboard.

---

## 📌 1. Project Overview & Objectives

Agricultural crop yield is governed by non-linear interactions among climate conditions, management inputs, soil properties, and crop genetics. This project develops an end-to-end, scientifically defensible machine learning decision-support system to predict crop yield prior to harvest.

### Key Highlights
- **Prediction Target**: Crop Yield per Hectare ($\text{MT/ha}$).
- **Required Predictors**: Crop Type, Annual Rainfall, Temperature, Soil Type, Fertilizer Usage, and Cultivated Area.
- **Leakage Prevention**: Strictly excludes historical harvested production from predictor features ($\text{Yield} = \frac{\text{Production}}{\text{Area}}$), preventing artificial accuracy inflation.
- **Zero Synthetic Fabrication**: Trained exclusively on genuine historical Indian agricultural census records (1997–2020) from the Ministry of Agriculture & Farmers Welfare, India Meteorological Department (IMD), and NASA POWER APIs.
- **Dual Model Benchmarking**: Direct performance comparison between parametric **Linear Regression** and ensemble **Random Forest Regressor** using 5-fold cross-validation and held-out test evaluation.

---

## 🏗️ 2. Repository Architecture

```text
Crop_Yield_Prediction/
├── data/
│   ├── raw/
│   │   ├── crop_production.csv         # 407,315 raw MoA census records
│   │   ├── weather.csv                 # (Optional) User-supplied climate data
│   │   └── soil.csv                    # (Optional) User-supplied soil classifications
│   ├── processed/
│   │   └── final_crop_yield_dataset.csv # Cleaned & unit-validated dataset (19,411 rows)
│   ├── crop_yield.csv                  # Historical multi-feature dataset (19,689 rows)
│   ├── rainfall.csv                    # IMD historical rainfall data (1901-2015)
│   └── weather_data.csv                # NASA POWER daily meteorological records
├── models/
│   ├── best_crop_yield_model.pkl       # Serialized winning pipeline (Random Forest)
│   ├── linear_regression_pipeline.pkl  # Baseline Linear Regression pipeline
│   ├── random_forest_pipeline.pkl      # Random Forest pipeline
│   ├── feature_columns.pkl             # Feature schema list
│   ├── test_data.pkl                   # Held-out test partition
│   └── model_metadata.json             # Model training metadata & CV scores
├── reports/
│   ├── dataset_validation_report.md    # Raw data schema & overlap analysis
│   ├── data_quality_report.md          # Row filtering audit & unit sanitization
│   ├── model_comparison.csv            # Held-out test evaluation benchmarks
│   └── evaluation_summary.json         # Evaluation metrics & selection rationale
├── static/
│   ├── css/
│   │   └── style.css                   # Responsive agricultural dashboard design
│   ├── js/
│   │   └── script.js                   # Client-side validation & tab switching
│   └── images/
│       ├── actual_vs_predicted.png     # Parity plot (Actual vs Predicted)
│       ├── residual_analysis.png       # Residual scatter & distribution
│       ├── feature_importance.png      # Top 15 Random Forest features
│       ├── model_comparison_bars.png   # MAE, RMSE, R² comparison
│       ├── crop_wise_yield.png         # Mean crop productivity
│       ├── year_wise_trend.png         # Annual yield trajectory (1997-2020)
│       ├── rainfall_vs_yield.png       # Rainfall vs yield scatter
│       ├── temperature_vs_yield.png    # Meteorological temperature profile
│       ├── fertilizer_vs_yield.png     # Fertilizer usage vs yield
│       ├── top_crops_yield.png         # Top high-yielding cultivars
│       ├── correlation_heatmap.png     # Feature correlation matrix
│       └── yield_distribution.png      # Target variable distribution
├── templates/
│   ├── base.html                       # Semantic HTML5 layout with navbar & footer
│   ├── index.html                      # Home landing page with workflow & KPIs
│   ├── dashboard.html                  # Interactive Chart.js & EDA gallery
│   ├── prediction.html                 # 6-parameter prediction form & results card
│   ├── model_results.html              # Benchmarking table & diagnostic charts
│   └── about.html                      # Academic methodology & source attribution
├── tests/
│   └── test_pipeline.py                # 17 automated unit tests
├── data_collection.py                  # Scans and validates raw datasets
├── data_preprocessing.py               # Cleaning, unit validation, and merging pipeline
├── eda.py                              # Generates 8 publication-quality EDA figures
├── train_model.py                      # Trains models with 5-fold CV and saves artifacts
├── model_evaluation.py                 # Generates diagnostics and comparison figures
├── app.py                              # Flask web application & REST API
├── requirements.txt                    # Project dependencies
├── README.md                           # Comprehensive documentation
└── .gitignore                          # Environment and cache ignore rules
```

---

## 📊 3. Dataset Attribution & Placement Instructions

### Genuine Data Sources
1. **Ministry of Agriculture & Farmers Welfare, India (MoA&FW)**:
   - *Source*: Directorate of Economics & Statistics (DES) / Open Government Data (OGD) Platform India.
   - *Records*: District-level crop production records from 1997–2022.
   - *Columns*: `State_Name`, `District_Name`, `Crop_Year`, `Season`, `Crop`, `Area`, `Production`.
2. **India Meteorological Department (IMD)**:
   - *Source*: National Data Centre, IMD Pune.
   - *Records*: Subdivision-wise annual and seasonal precipitation archives (1901–2015).
3. **NASA POWER Agroclimatology Project**:
   - *Source*: NASA Langley Research Center POWER API (`https://power.larc.nasa.gov`).
   - *Records*: Daily 2-meter ambient temperature and corrected precipitation observations.

### Folder Placement
Place your raw files in `data/raw/`:
- `data/raw/crop_production.csv`: Government agricultural census data.
- `data/raw/weather.csv` *(Optional)*: Climate data with columns `[State, District, Crop_Year, Season, Temperature, Rainfall]`.
- `data/raw/soil.csv` *(Optional)*: Soil data with columns `[State, District, Soil_Type]`.

---

## ⚙️ 4. Installation & Setup (Windows PowerShell)

### Step 1: Open Terminal in Project Root
```powershell
cd c:\Users\acer\OneDrive\Desktop\Crop_Yield_Prediction
```

### Step 2: (Optional) Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 5. Execution Pipeline Commands

Run the pipeline sequentially from data validation to model evaluation:

### 1. Data Inspection & Schema Audit
Scans all datasets in `data/` and `data/raw/`, printing record counts, duplicates, missing values, and generating `reports/dataset_validation_report.md`:
```powershell
python data_collection.py
```

### 2. Data Cleaning & Integration Pipeline
Sanitizes unit inconsistencies (separating coconut records recorded in nuts), enforces domain constraints ($\text{Area} > 0, \text{Yield} > 0$), prevents target leakage, merges optional weather/soil datasets, and saves `data/processed/final_crop_yield_dataset.csv`:
```powershell
python data_preprocessing.py
```

### 3. Exploratory Data Analysis (EDA)
Generates 8 high-resolution figures in `static/images/`:
```powershell
python eda.py
```

### 4. Machine Learning Model Training
Trains Linear Regression and Random Forest Regressor with 5-fold cross-validation, evaluates on held-out test data (3,883 samples), and saves the winning pipeline:
```powershell
python train_model.py
```

### 5. Diagnostic Model Evaluation
Generates parity plots, residual analysis, and feature importance charts:
```powershell
python model_evaluation.py
```

### 6. Run Automated Test Suite
Executes 17 comprehensive unit tests verifying data integrity, model inference, Flask routes, and error handling:
```powershell
python -m unittest tests/test_pipeline.py -v
```

---

## 🌐 6. Launching the Flask Web Application

Start the Flask development server:
```powershell
python app.py
```

### Accessing the Web Application
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Available Pages
| Route | Page Name | Description |
| :--- | :--- | :--- |
| `/` | **Home** | Project overview, key metrics, workflow architecture, and highlights. |
| `/dashboard` | **Analytics Dashboard** | Interactive Chart.js graphs, dataset KPIs, and tabbed EDA visualization gallery. |
| `/predict` | **Yield Estimator** | Validated prediction form for Crop, Area, Rainfall, Temperature, Soil, Fertilizer. |
| `/model-results` | **Model Results** | Real MAE, RMSE, R² comparison table, parity plot, and residual diagnostics. |
| `/about` | **About & Methodology** | Academic problem formulation, mathematical formulas, and data integration documentation. |
| `/api/metrics` | **REST API (Metrics)** | JSON endpoint returning benchmark comparison results. |
| `/api/predict` | **REST API (Inference)**| POST JSON endpoint for automated yield prediction. |

---

## 📈 7. Real Empirical Model Benchmarks

Evaluated on held-out test data (3,883 records):

| Machine Learning Algorithm | 5-Fold CV R² (Training) | Test MAE ($\text{MT/ha}$) ↓ | Test RMSE ($\text{MT/ha}$) ↓ | Test R² Score ↑ | Selection Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | $0.5570 \pm 0.0857$ | $2.8096$ | $6.3387$ | $0.7171$ | Baseline Comparator |
| **Random Forest Regressor** | **$0.7320 \pm 0.1072$** | **$1.0116$** | **$3.5634$** | **$0.9106$** | 🏆 **Selected Best Model** |

### Key Takeaways
1. **Random Forest Superiority**: Achieved an outstanding test MAE of **$1.0116\text{ MT/ha}$** and an $R^2$ of **$0.9106$**, demonstrating strong non-linear capability across crop species.
2. **Leakage-Free Honesty**: By eliminating $\text{Production}$ from predictor inputs, the model avoids fabricated 99% accuracy scores and reflects genuine predictive explanatory power.

---

## ⚠️ 8. Scientific Limitations & Future Scope

- **Intra-Seasonal Rainfall Distribution**: Annual cumulative rainfall does not reflect monsoon dry spells during sensitive crop growth phases (flowering/grain-filling).
- **Satellite Remote Sensing**: Future work can incorporate Sentinel-2 multispectral NDVI and remote-sensing surface soil moisture.
- **Biotic Stress Factors**: Exogenous shocks like pest infestations or sudden viral blights are not captured in regional census records.

---

## 📜 9. Academic License & Attribution
Developed for academic, research, and non-commercial educational purposes. Agricultural records sourced from public repositories under the Open Government Data (OGD) Platform India and NASA POWER.
