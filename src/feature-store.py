"""
src/feature_store.py
================================================================
Feature Store:
- computes features ONCE
- stores in Azure Blob
- any model can load them
- no duplicate preprocessing!

Features stored:
- amount_scaled    → scaled transaction amount
- frequency_scaled → scaled transaction frequency
- label            → spam or not spam
================================================================
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import argparse
from sklearn.preprocessing import StandardScaler
from azure.storage.blob import BlobServiceClient

# ================================================================
# ARGUMENT PARSER
# ================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--input_data",    type=str, help="Path to clean data")
parser.add_argument("--output_features", type=str, help="Path to save features")
args = parser.parse_args()

# ================================================================
# STORAGE CONFIG
# ================================================================

STORAGE_ACCOUNT_NAME = "rndbhavishyarg8495"
CONTAINER_NAME       = "spam-detector-data"
FEATURE_STORE_PATH   = "feature-store"

# ================================================================
# STEP 1: LOAD CLEAN DATA
# ================================================================

print("Loading clean data...")
input_path = args.input_data if args.input_data else "data"
df         = pd.read_csv(os.path.join(input_path, "clean_data.csv"))
print(f"Loaded {len(df)} rows")

# ================================================================
# STEP 2: COMPUTE FEATURES
# ================================================================

print("\nComputing features...")

# raw features
X = df[["amount", "frequency"]]
y = df["label"]

# scale features
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# build feature dataframe
features_df = pd.DataFrame(X_scaled, columns=["amount_scaled", "frequency_scaled"])
features_df["amount_raw"]    = df["amount"].values
features_df["frequency_raw"] = df["frequency"].values
features_df["label"]         = y.values

print(f"Features computed: {features_df.shape}")
print(features_df.head())

# ================================================================
# STEP 3: COMPUTE FEATURE STATS
# used later for drift detection!
# ================================================================

print("\nComputing feature statistics...")

feature_stats = {
    "amount": {
        "mean":  float(df["amount"].mean()),
        "std":   float(df["amount"].std()),
        "min":   float(df["amount"].min()),
        "max":   float(df["amount"].max()),
        "p25":   float(df["amount"].quantile(0.25)),
        "p75":   float(df["amount"].quantile(0.75))
    },
    "frequency": {
        "mean":  float(df["frequency"].mean()),
        "std":   float(df["frequency"].std()),
        "min":   float(df["frequency"].min()),
        "max":   float(df["frequency"].max()),
        "p25":   float(df["frequency"].quantile(0.25)),
        "p75":   float(df["frequency"].quantile(0.75))
    },
    "label": {
        "spam_ratio":   float(y.mean()),
        "normal_ratio": float(1 - y.mean()),
        "total_rows":   len(df)
    },
    "scaler": {
        "amount_mean":      float(scaler.mean_[0]),
        "amount_scale":     float(scaler.scale_[0]),
        "frequency_mean":   float(scaler.mean_[1]),
        "frequency_scale":  float(scaler.scale_[1])
    }
}

print("Feature stats computed!")
print(json.dumps(feature_stats, indent=2))

# ================================================================
# STEP 4: SAVE FEATURES LOCALLY
# ================================================================

print("\nSaving features locally...")
output_path = args.output_features if args.output_features else "features"
os.makedirs(output_path, exist_ok=True)

# save feature dataframe
features_df.to_csv(
    os.path.join(output_path, "features.csv"),
    index=False
)

# save feature stats (used for drift detection!)
with open(os.path.join(output_path, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=4)

# save scaler (same scaler for all models!)
joblib.dump(scaler, os.path.join(output_path, "scaler.pkl"))

print(f"✅ features.csv saved!")
print(f"✅ feature_stats.json saved!")
print(f"✅ scaler.pkl saved!")
print("\nFeature Store complete!")