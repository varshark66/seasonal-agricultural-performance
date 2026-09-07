# Seasonal Agriculture Performance Analysis
# Major Project – Data Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = "seasonal_agriculture_performance_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------
# 2. Basic Dataset Information
# -----------------------------
print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Cells:", int(df.isnull().sum().sum()))

# -----------------------------
# 3. Data Cleaning
# -----------------------------
# Fill missing numeric values with the median.
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with the mode.
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

# -----------------------------
# 4. Seasonal Performance Analysis
# -----------------------------
season_summary = (
    df.groupby("Season")
      .agg(
          Average_Yield_Tonnes_Ha=("Yield_Tonnes_Ha", "mean"),
          Total_Production_Tonnes=("Production_Tonnes", "sum"),
          Average_Revenue_INR=("Revenue_INR", "mean"),
          Average_Profit_INR=("Profit_INR", "mean"),
          Average_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean"),
          Farm_Count=("Farm_ID", "count")
      )
      .sort_values("Average_Yield_Tonnes_Ha", ascending=False)
)

print("\nSeasonal Performance Summary:")
print(season_summary.round(2))

# -----------------------------
# 5. Visualization – Average Yield by Season
# -----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(
    data=season_summary.reset_index(),
    x="Season",
    y="Average_Yield_Tonnes_Ha"
)
plt.title("Average Crop Yield by Season")
plt.xlabel("Season")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Visualization – Production by Season
# -----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(
    data=season_summary.reset_index(),
    x="Season",
    y="Total_Production_Tonnes"
)
plt.title("Total Production by Season")
plt.xlabel("Season")
plt.ylabel("Production (Tonnes)")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Visualization – Average Profit by Season
# -----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(
    data=season_summary.reset_index(),
    x="Season",
    y="Average_Profit_INR"
)
plt.title("Average Profit by Season")
plt.xlabel("Season")
plt.ylabel("Average Profit (INR)")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. State-wise Yield Analysis
# -----------------------------
state_yield = (
    df.groupby("State")["Yield_Tonnes_Ha"]
      .mean()
      .sort_values(ascending=False)
)

print("\nTop 10 States by Average Yield:")
print(state_yield.head(10).round(2))

plt.figure(figsize=(10, 6))
state_yield.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 States by Average Crop Yield")
plt.xlabel("Average Yield (Tonnes/Ha)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Crop-wise Yield Analysis
# -----------------------------
crop_yield = (
    df.groupby("Crop")["Yield_Tonnes_Ha"]
      .mean()
      .sort_values(ascending=False)
)

print("\nAverage Yield by Crop:")
print(crop_yield.round(2))

plt.figure(figsize=(10, 6))
crop_yield.sort_values().plot(kind="barh")
plt.title("Average Crop Yield by Crop")
plt.xlabel("Average Yield (Tonnes/Ha)")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Irrigation Method Analysis
# -----------------------------
irrigation_summary = (
    df.groupby("Irrigation_Method")
      .agg(
          Average_Yield=("Yield_Tonnes_Ha", "mean"),
          Average_Water_Used=("Water_Used_m3", "mean"),
          Average_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean")
      )
      .sort_values("Average_Yield", ascending=False)
)

print("\nIrrigation Method Summary:")
print(irrigation_summary.round(2))

plt.figure(figsize=(9, 5))
sns.barplot(
    data=irrigation_summary.reset_index(),
    x="Irrigation_Method",
    y="Average_Yield"
)
plt.title("Average Yield by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# -----------------------------
# 11. Correlation Analysis
# -----------------------------
numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

print("\nCorrelation with Crop Yield:")
yield_corr = correlation["Yield_Tonnes_Ha"].drop("Yield_Tonnes_Ha").sort_values(ascending=False)
print(yield_corr.round(3))

plt.figure(figsize=(13, 9))
sns.heatmap(correlation, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Numeric Variables")
plt.tight_layout()
plt.show()

# -----------------------------
# 12. Strongest Relationships with Yield
# -----------------------------
print("\nTop Positive Relationships with Yield:")
print(yield_corr.head(5).round(3))

# -----------------------------
# 13. Key Findings
# -----------------------------
best_season = season_summary["Average_Yield_Tonnes_Ha"].idxmax()
best_profit_season = season_summary["Average_Profit_INR"].idxmax()
best_state = state_yield.idxmax()
best_crop = crop_yield.idxmax()

print("\n========== KEY FINDINGS ==========")
print(f"1. Highest average yield season: {best_season}")
print(f"2. Highest average profit season: {best_profit_season}")
print(f"3. Highest average-yield state: {best_state}")
print(f"4. Highest average-yield crop: {best_crop}")
print(f"5. Strongest positive numeric relationship with yield: {yield_corr.idxmax()}")
print("==================================")

# -----------------------------
# 14. Save Summary Tables
# -----------------------------
season_summary.to_csv("seasonal_performance_summary.csv")
state_yield.to_csv("state_yield_summary.csv", header=["Average_Yield_Tonnes_Ha"])
crop_yield.to_csv("crop_yield_summary.csv", header=["Average_Yield_Tonnes_Ha"])
yield_corr.to_csv("yield_correlation_summary.csv", header=["Correlation_with_Yield"])

print("\nSummary CSV files created successfully.")
