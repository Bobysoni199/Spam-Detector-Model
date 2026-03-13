"""
src/monitor.py
================================================================
Model Monitoring - Data Drift Detection:
- loads baseline feature stats from Feature Store
- compares with new incoming data stats
- detects if data has drifted
- sends alert if drift detected
- triggers retrain if needed

What is drift?
Training data: amount mean = 255
New data:      amount mean = 4000  ← DRIFT! model will fail!
================================================================
"""

import json
import os
import math
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ================================================================
# ARGUMENT PARSER
# ================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--baseline_stats", type=str, help="Path to baseline feature stats")
parser.add_argument("--new_data",       type=str, help="Path to new data to monitor")
parser.add_argument("--output_report",  type=str, help="Path to save drift report")
args = parser.parse_args()

# ================================================================
# DRIFT THRESHOLDS
# how much change is too much?
# ================================================================

MEAN_DRIFT_THRESHOLD = 0.2   # 20% change in mean = drift!
STD_DRIFT_THRESHOLD  = 0.3   # 30% change in std  = drift!
SPAM_RATIO_THRESHOLD = 0.15  # 15% change in spam ratio = drift!

# ================================================================
# STEP 1: LOAD BASELINE STATS
# from feature store → what data looked like during training
# ================================================================

print("Loading baseline stats...")
baseline_path = args.baseline_stats if args.baseline_stats else "features"
with open(os.path.join(baseline_path, "feature_stats.json"), "r") as f:
    baseline = json.load(f)

print("Baseline stats loaded!")
print(f"Training amount mean : {baseline['amount']['mean']:.2f}")
print(f"Training freq mean   : {baseline['frequency']['mean']:.2f}")
print(f"Training spam ratio  : {baseline['label']['spam_ratio']:.2%}")

# ================================================================
# STEP 2: LOAD NEW DATA
# ================================================================

print("\nLoading new data...")
new_data_path = args.new_data if args.new_data else "data"
df_new        = pd.read_csv(os.path.join(new_data_path, "clean_data.csv"))
print(f"New data: {len(df_new)} rows")

# ================================================================
# STEP 3: COMPUTE NEW DATA STATS
# ================================================================

print("\nComputing new data stats...")

new_stats = {
    "amount": {
        "mean": float(df_new["amount"].mean()),
        "std":  float(df_new["amount"].std()),
    },
    "frequency": {
        "mean": float(df_new["frequency"].mean()),
        "std":  float(df_new["frequency"].std()),
    },
    "label": {
        "spam_ratio": float(df_new["label"].mean())
    }
}

print(f"New amount mean : {new_stats['amount']['mean']:.2f}")
print(f"New freq mean   : {new_stats['frequency']['mean']:.2f}")
print(f"New spam ratio  : {new_stats['label']['spam_ratio']:.2%}")

# ================================================================
# STEP 4: DETECT DRIFT
# compare baseline vs new data
# ================================================================

print("\nDetecting drift...")

drift_detected = False
drift_details  = []

def check_drift(feature, metric, baseline_val, new_val, threshold):
    # calculate % change
    change = abs(new_val - baseline_val) / (abs(baseline_val) + 1e-10)
    drifted = change > threshold
    status  = "🚨 DRIFT!" if drifted else "✅ OK"
    print(f"{feature} {metric}: {baseline_val:.3f} → {new_val:.3f} | change: {change:.2%} | {status}")
    return drifted, {
        "feature":      feature,
        "metric":       metric,
        "baseline":     round(baseline_val, 4),
        "new_value":    round(new_val, 4),
        "change_pct":   round(change * 100, 2),
        "threshold_pct": round(threshold * 100, 2),
        "drifted":      drifted
    }

# check amount drift
drifted, detail = check_drift(
    "amount", "mean",
    baseline["amount"]["mean"],
    new_stats["amount"]["mean"],
    MEAN_DRIFT_THRESHOLD
)
if drifted: drift_detected = True
drift_details.append(detail)

drifted, detail = check_drift(
    "amount", "std",
    baseline["amount"]["std"],
    new_stats["amount"]["std"],
    STD_DRIFT_THRESHOLD
)
if drifted: drift_detected = True
drift_details.append(detail)

# check frequency drift
drifted, detail = check_drift(
    "frequency", "mean",
    baseline["frequency"]["mean"],
    new_stats["frequency"]["mean"],
    MEAN_DRIFT_THRESHOLD
)
if drifted: drift_detected = True
drift_details.append(detail)

# check spam ratio drift
drifted, detail = check_drift(
    "spam_ratio", "ratio",
    baseline["label"]["spam_ratio"],
    new_stats["label"]["spam_ratio"],
    SPAM_RATIO_THRESHOLD
)
if drifted: drift_detected = True
drift_details.append(detail)

# ================================================================
# STEP 5: BUILD DRIFT REPORT
# ================================================================

print("\nBuilding drift report...")

report = {
    "timestamp":       datetime.now().isoformat(),
    "drift_detected":  drift_detected,
    "status":          "DRIFT DETECTED - RETRAIN NEEDED!" if drift_detected else "NO DRIFT - MODEL OK",
    "baseline_rows":   baseline["label"]["total_rows"],
    "new_data_rows":   len(df_new),
    "drift_details":   drift_details,
    "recommendation":  "retrain" if drift_detected else "no_action"
}

# ================================================================
# STEP 6: SAVE REPORT
# ================================================================

output_path = args.output_report if args.output_report else "monitoring"
os.makedirs(output_path, exist_ok=True)

report_file = os.path.join(output_path, "drift_report.json")
with open(report_file, "w") as f:
    json.dump(report, f, indent=4)

print(f"\n✅ Drift report saved to {report_file}")

# ================================================================
# STEP 7: PRINT SUMMARY
# ================================================================

print("\n" + "=" * 60)
if drift_detected:
    print("🚨 DRIFT DETECTED!")
    print("=" * 60)
    print("Model may be giving wrong predictions!")
    print("Recommendation: RETRAIN MODEL")
    print("\nDrifted features:")
    for d in drift_details:
        if d["drifted"]:
            print(f"  → {d['feature']} {d['metric']}: changed by {d['change_pct']}%")
else:
    print("✅ NO DRIFT DETECTED!")
    print("=" * 60)
    print("Model is still performing well!")
    print("Recommendation: NO ACTION NEEDED")

print(f"\nFull report: {report_file}")
print("=" * 60)