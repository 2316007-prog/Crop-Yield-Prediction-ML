
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Input and output paths
RESULTS_FILE = Path("data/model_comparison_results.csv")
OUTPUT_DIR = Path("static")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not RESULTS_FILE.exists():
    raise FileNotFoundError(
        "Results file not found. Run python train_model.py first."
    )

results = pd.read_csv(RESULTS_FILE)

required_columns = {"Model", "MAE", "RMSE", "R2"}

if not required_columns.issubset(results.columns):
    raise ValueError("The model comparison file has unexpected columns.")

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 55)
print(results.to_string(index=False))

# Plot 1: Mean Absolute Error
plt.figure(figsize=(9, 5))
bars = plt.bar(results["Model"], results["MAE"])

plt.title("Model Comparison - Mean Absolute Error")
plt.ylabel("MAE (lower is better)")
plt.xlabel("Machine Learning Model")
plt.grid(axis="y", linestyle="--", alpha=0.35)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_mae.png", dpi=300)
plt.show()
plt.close()

# Plot 2: Root Mean Squared Error
plt.figure(figsize=(9, 5))
bars = plt.bar(results["Model"], results["RMSE"])

plt.title("Model Comparison - Root Mean Squared Error")
plt.ylabel("RMSE (lower is better)")
plt.xlabel("Machine Learning Model")
plt.grid(axis="y", linestyle="--", alpha=0.35)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_rmse.png", dpi=300)
plt.show()
plt.close()

# Plot 3: R-squared score
plt.figure(figsize=(9, 5))
bars = plt.bar(results["Model"], results["R2"])

plt.title("Model Comparison - R-squared Score")
plt.ylabel("R² score (higher is generally better)")
plt.xlabel("Machine Learning Model")
plt.grid(axis="y", linestyle="--", alpha=0.35)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_r2.png", dpi=300)
plt.show()
plt.close()

print("\nAll comparison graphs saved successfully:")
print(OUTPUT_DIR / "model_mae.png")
print(OUTPUT_DIR / "model_rmse.png")
print(OUTPUT_DIR / "model_r2.png")
