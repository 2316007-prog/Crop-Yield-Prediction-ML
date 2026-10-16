"""
Crop Yield Prediction - Exploratory Data Analysis (EDA) Module
==============================================================
Generates 8 high-resolution publication-quality visualization charts:
1. crop_wise_yield.png      - Crop-wise average yield (tonnes/ha)
2. year_wise_trend.png       - Year-wise historical crop yield trend (1997-2020)
3. rainfall_vs_yield.png     - Annual Rainfall (mm) vs Yield (tonnes/ha)
4. temperature_vs_yield.png  - Temperature profile & relationship
5. fertilizer_vs_yield.png   - Fertilizer usage vs Crop Yield
6. top_crops_yield.png       - Top crops ranked by average yield
7. correlation_heatmap.png   - Feature correlation matrix
8. yield_distribution.png    - Yield target variable distribution

All charts are saved separately in static/images/ using verified real data.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "bold"


def run_eda():
    data_path = Path("data/processed/final_crop_yield_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run data_preprocessing.py first.")

    df = pd.read_csv(data_path)
    output_dir = Path("static/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded processed dataset: {len(df):,} records.")
    print("Generating EDA charts in static/images/ ...")

    palette = sns.color_palette("viridis")

    # 1. Crop-wise average yield (Top 15 most frequent crops)
    plt.figure(figsize=(12, 6))
    frequent_crops = df["Crop"].value_counts().head(15).index
    crop_yield_avg = (
        df[df["Crop"].isin(frequent_crops)]
        .groupby("Crop")["Yield"]
        .mean()
        .sort_values(ascending=False)
    )
    ax = sns.barplot(x=crop_yield_avg.values, y=crop_yield_avg.index, palette="mako")
    plt.title("Crop-Wise Average Yield (Top 15 Most Common Crops)")
    plt.xlabel("Average Yield (Metric Tonnes / Hectare)")
    plt.ylabel("Crop Species")
    for i, v in enumerate(crop_yield_avg.values):
        ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=9, fontweight="bold", color="#1c3d28")
    plt.tight_layout()
    plt.savefig(output_dir / "crop_wise_yield.png", dpi=300)
    plt.close()
    print("  -> Saved: crop_wise_yield.png")

    # 2. Year-wise yield trend (1997-2020)
    plt.figure(figsize=(11, 5.5))
    yearly_stats = df.groupby("Crop_Year")["Yield"].agg(["mean", "median", "std"]).reset_index()
    plt.plot(yearly_stats["Crop_Year"], yearly_stats["mean"], marker="o", color="#1b7837", linewidth=2.5, label="Mean Yield")
    plt.plot(yearly_stats["Crop_Year"], yearly_stats["median"], marker="s", linestyle="--", color="#762a83", linewidth=2, label="Median Yield")
    plt.fill_between(
        yearly_stats["Crop_Year"],
        np.maximum(0, yearly_stats["mean"] - 0.5 * yearly_stats["std"]),
        yearly_stats["mean"] + 0.5 * yearly_stats["std"],
        color="#a6dba0",
        alpha=0.35,
        label="Yield Variability (±0.5 Std)"
    )
    plt.title("Historical Crop Yield Trend in India (1997 – 2020)")
    plt.xlabel("Harvest Year")
    plt.ylabel("Crop Yield (Metric Tonnes / Hectare)")
    plt.xticks(yearly_stats["Crop_Year"][::2], rotation=45)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / "year_wise_trend.png", dpi=300)
    plt.close()
    print("  -> Saved: year_wise_trend.png")

    # 3. Rainfall versus yield
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="Annual_Rainfall",
        y="Yield",
        alpha=0.45,
        color="#2b83ba",
        edgecolor=None,
        s=35
    )
    # Fit polynomial trendline
    p = np.poly1d(np.polyfit(sample_df["Annual_Rainfall"], sample_df["Yield"], 2))
    x_line = np.linspace(sample_df["Annual_Rainfall"].min(), sample_df["Annual_Rainfall"].max(), 200)
    plt.plot(x_line, p(x_line), color="#d7191c", linewidth=2.5, label="Non-Linear Polynomial Trend")
    plt.title("Annual Rainfall versus Crop Yield")
    plt.xlabel("Annual Rainfall (Millimeters - mm)")
    plt.ylabel("Crop Yield (Metric Tonnes / Hectare)")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "rainfall_vs_yield.png", dpi=300)
    plt.close()
    print("  -> Saved: rainfall_vs_yield.png")

    # 4. Temperature versus yield / Temperature Profile
    plt.figure(figsize=(10, 6))
    if "Temperature" in df.columns:
        sns.scatterplot(data=sample_df, x="Temperature", y="Yield", alpha=0.5, color="#e66101")
        sns.regplot(data=sample_df, x="Temperature", y="Yield", scatter=False, color="#5e3c99")
        plt.title("Ambient Temperature versus Crop Yield")
        plt.xlabel("Average Temperature (°C)")
        plt.ylabel("Crop Yield (Metric Tonnes / Hectare)")
    else:
        # Check verified weather dataset (NASA POWER records)
        weather_path = Path("data/weather_data.csv")
        if weather_path.exists():
            w_df = pd.read_csv(weather_path)
            plt.plot(pd.to_datetime(w_df["Date"]), w_df["Temperature"], color="#d95f02", linewidth=1.8, label="Daily Mean Temperature (°C)")
            plt.axhline(w_df["Temperature"].mean(), color="#7570b3", linestyle="--", label=f"Annual Mean ({w_df['Temperature'].mean():.1f}°C)")
            plt.title("Observed Agricultural Daily Temperature Profile (Verified NASA POWER Data)")
            plt.xlabel("Observation Date")
            plt.ylabel("Temperature (Degrees Celsius - °C)")
            plt.legend(frameon=True)
        else:
            plt.text(0.5, 0.5, "Temperature feature pending genuine weather.csv integration", ha="center", va="center")
            plt.title("Temperature versus Yield Profile")
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_vs_yield.png", dpi=300)
    plt.close()
    print("  -> Saved: temperature_vs_yield.png")

    # 5. Fertilizer usage versus yield
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    # Log transform fertilizer for readable plotting due to scale
    sns.scatterplot(
        data=sample_df,
        x=np.log1p(sample_df["Fertilizer"]),
        y="Yield",
        alpha=0.45,
        color="#31a354",
        edgecolor=None,
        s=35
    )
    sns.regplot(
        x=np.log1p(sample_df["Fertilizer"]),
        y=sample_df["Yield"],
        scatter=False,
        color="#006d2c",
        line_kws={"linewidth": 2.5}
    )
    plt.title("Fertilizer Usage (Log Scale) versus Crop Yield")
    plt.xlabel("Log(1 + Fertilizer Usage) [kg]")
    plt.ylabel("Crop Yield (Metric Tonnes / Hectare)")
    plt.tight_layout()
    plt.savefig(output_dir / "fertilizer_vs_yield.png", dpi=300)
    plt.close()
    print("  -> Saved: fertilizer_vs_yield.png")

    # 6. Top crops by yield (Highest average yields)
    plt.figure(figsize=(11, 6))
    top_crops = df.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10)
    ax = sns.barplot(x=top_crops.values, y=top_crops.index, palette="crest")
    plt.title("Top 10 High-Yielding Agricultural Crops in India")
    plt.xlabel("Mean Yield (Metric Tonnes / Hectare)")
    plt.ylabel("Crop Species")
    for i, v in enumerate(top_crops.values):
        ax.text(v + 0.5, i, f"{v:.1f} MT/ha", va="center", fontsize=9, fontweight="bold", color="#1c3d28")
    plt.tight_layout()
    plt.savefig(output_dir / "top_crops_yield.png", dpi=300)
    plt.close()
    print("  -> Saved: top_crops_yield.png")

    # 7. Correlation heatmap
    plt.figure(figsize=(9, 7))
    num_cols = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
    if "Crop_Year" in df.columns:
        num_cols.insert(0, "Crop_Year")
    if "Temperature" in df.columns:
        num_cols.append("Temperature")

    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        linewidths=1.2,
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )
    plt.title("Agricultural Variables Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close()
    print("  -> Saved: correlation_heatmap.png")

    # 8. Yield distribution (Histogram & KDE + Log scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.histplot(df["Yield"], bins=50, kde=True, color="#238b45", ax=ax1)
    ax1.set_title("Yield Distribution (Original Scale)")
    ax1.set_xlabel("Yield (Metric Tonnes / Hectare)")
    ax1.set_ylabel("Frequency Count")

    sns.histplot(np.log1p(df["Yield"]), bins=50, kde=True, color="#08519c", ax=ax2)
    ax2.set_title("Yield Distribution (Log1p Transformed)")
    ax2.set_xlabel("Log(1 + Yield)")
    ax2.set_ylabel("Frequency Count")

    plt.suptitle("Target Variable (Yield per Hectare) Empirical Distribution", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "yield_distribution.png", dpi=300)
    plt.close()
    print("  -> Saved: yield_distribution.png")

    print("\nAll 8 EDA charts successfully generated in static/images/.")


if __name__ == "__main__":
    print("=" * 70)
    print("   CROP YIELD PREDICTION: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    run_eda()
    print("=" * 70)
