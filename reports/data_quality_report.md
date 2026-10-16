# Data Preprocessing & Quality Assurance Report

> **Project**: Crop Yield Prediction Using Machine Learning
> **Generated at**: 2026-10-16 09:53:43
> **Primary Dataset Source**: `crop_yield.csv`

---

## 1. Preprocessing Summary & Row Accounting

- **Initial Records Loaded**: 19,689
- **Clean Records Retained**: 19,411 (98.59%)
- **Total Rows Filtered**: 278

### Audit of Removed Records & Explanations

| Removal Reason | Rows Filtered | Rationale & Justification |
| :--- | :--- | :--- |
| **Exact Duplicate Rows** | 0 | Identical entries recorded multiple times across census reporting cycles. |
| **Coconut Records (Unit Incompatibility: Nuts vs Tonnes)** | 172 | Coconut harvest is officially measured in thousands of individual nuts rather than metric tonnes. Computing Yield = Production / Area gives 10,000+ nuts/ha, creating severe unit incompatibility with standard field crops (tonnes/ha). |
| **Unphysical / Zero Yield Outliers (>500 or <=0 tonnes/ha)** | 106 | Filtered records where Yield was zero (complete crop abandonment or non-reporting) or exceeded physical biological limits (>500 MT/ha). |

---

## 2. Final Processed Dataset Schema

- **File Path**: `data/processed/final_crop_yield_dataset.csv`
- **Total Observations**: 19,411
- **Total Columns**: 10

| Column Name | Data Type | Non-Null Count | Unit of Measurement | Role |
| :--- | :--- | :--- | :--- | :--- |
| `Crop` | `str` | 19,411 | Categorical | Predictor Feature |
| `Crop_Year` | `int64` | 19,411 | Calendar Year (1997–2020) | Temporal Feature / Grouping |
| `Season` | `str` | 19,411 | Categorical | Predictor Feature |
| `State` | `str` | 19,411 | Categorical | Predictor Feature |
| `Area` | `float64` | 19,411 | Hectares (ha) | Predictor Feature |
| `Production` | `int64` | 19,411 | Metric Tonnes (MT) | Historical Target Source (Excluded from training features to prevent leakage) |
| `Annual_Rainfall` | `float64` | 19,411 | Millimeters (mm) | Predictor Feature |
| `Fertilizer` | `float64` | 19,411 | Kilograms (kg) | Predictor Feature |
| `Pesticide` | `float64` | 19,411 | Kilograms (kg) | Predictor Feature |
| `Yield` | `float64` | 19,411 | Metric Tonnes / Hectare (MT/ha) | **Prediction Target** |

---

## 3. Statistical Distribution of Processed Variables

| Variable | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Crop_Year` | 2009.12 | 6.50 | 1997.00 | 2004.00 | 2010.00 | 2015.00 | 2020.00 |
| `Area` | 180492.74 | 737428.04 | 0.50 | 1411.50 | 9300.00 | 74947.50 | 50808100.00 |
| `Production` | 765300.98 | 5688660.49 | 0.00 | 1415.00 | 13673.00 | 116547.50 | 179698158.00 |
| `Annual_Rainfall` | 1435.27 | 815.41 | 301.30 | 940.70 | 1247.00 | 1635.90 | 6552.70 |
| `Fertilizer` | 24176259.18 | 95533166.68 | 54.17 | 191119.99 | 1229114.56 | 9960417.85 | 4835406877.00 |
| `Pesticide` | 49003.72 | 214637.29 | 0.09 | 362.63 | 2416.47 | 19918.97 | 15750511.00 |
| `Yield` | 4.38 | 12.80 | 0.00 | 0.60 | 1.03 | 2.34 | 381.42 |

---

## 4. Leakage Prevention & Scientific Compliance

> [!IMPORTANT]
> 1. **Target Leakage Strict Prohibition**: `Production` is strictly excluded from `X` (features). Because `Yield = Production / Area`, training with `Production` would induce trivial mathematical leakage.
> 2. **Unit Consistency**: All retained yields are measured in **Metric Tonnes per Hectare**. The coconut unit anomaly was isolated and removed.
> 3. **Zero Synthetic Generation**: No synthetic rainfall, fertilizer, or soil records were manufactured.
