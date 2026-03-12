"""
src/train.py
================================================================
AZURE ML PIPELINE - STEP 2 (MODEL TRAINING)

This script does the following:
1. Reads clean_data.csv produced by the preprocess step
2. Prepares ML features
3. Scales numeric features
4. Splits data into training and testing sets
5. Trains a Logistic Regression model
6. Evaluates model performance
7. Saves evaluation metrics
8. Saves the trained model and scaler

Outputs are saved to Azure Blob Storage paths provided
by the Azure ML pipeline.
================================================================
"""

# ================================================================
# IMPORT LIBRARIES
# These are tools used for data handling, ML training, and saving outputs
# ================================================================

import pandas as pd              # used for reading and handling datasets
import numpy as np               # used for numerical operations
import json                      # used to save metrics as JSON
import os                        # used to create directories and paths
import argparse                  # used to read input arguments from Azure ML
import joblib                    # used to save trained model files

# Machine Learning tools from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ================================================================
# ARGUMENT PARSER
# Azure ML pipeline automatically passes input/output paths
#
# Example pipeline command:
# python train.py \
#   --input_data /azureml/preprocess_output \
#   --output_model /azureml/model_output \
#   --output_metrics /azureml/metrics_output
# ================================================================

parser = argparse.ArgumentParser()

# Path where clean_data.csv is stored (from preprocess step)
parser.add_argument("--input_data", type=str, help="Path to clean data")

# Path where trained model will be saved
parser.add_argument("--output_model", type=str, help="Path to save models")

# Path where metrics will be saved
parser.add_argument("--output_metrics", type=str, help="Path to save metrics")

args = parser.parse_args()


# ================================================================
# READ CLEAN DATA
# This data was generated in Step 1 (preprocess.py)
# ================================================================

print("Reading clean data...")

# Determine input folder
# If Azure ML provides path → use it
# Otherwise default to local "data" folder
input_path = args.input_data if args.input_data else "data"

# Construct full file path
data_file = os.path.join(input_path, "clean_data.csv")

# Load dataset
df = pd.read_csv(data_file)

print(f"Loaded {len(df)} rows")


# ================================================================
# PREPARE FEATURES
# Machine learning separates:
# X = input features
# y = labels (answers)
# ================================================================

print("\nPreparing features...")

# Features used by the model
X = df[["amount", "frequency"]]

# Target variable (0 = normal transaction, 1 = spam transaction)
y = df["label"]


# ================================================================
# SCALE FEATURES
# ML models perform better when features have similar ranges
#
# Example problem:
# amount = 4500
# frequency = 35
#
# Scaling converts them to comparable values.
# ================================================================

print("\nScaling features...")

# Create scaler object
scaler = StandardScaler()

# Learn scaling parameters and transform data
X_scaled = scaler.fit_transform(X)

print(f"Mean:  {scaler.mean_}")
print(f"Scale: {scaler.scale_}")


# ================================================================
# SPLIT DATA INTO TRAINING AND TEST SETS
#
# Training set → used to train the model
# Test set → used to evaluate the model
# ================================================================

print("\nSplitting 80/20...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,        # 20% data for testing
    random_state=42       # ensures reproducibility
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")


# ================================================================
# TRAIN MACHINE LEARNING MODEL
#
# Logistic Regression learns the relationship between:
# amount + frequency → spam or not
# ================================================================

print("\nTraining model...")

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model using training dataset
model.fit(X_train, y_train)

print("Trained!")

# Display learned weights
print(f"Amount weight:    {round(model.coef_[0][0], 4)}")
print(f"Frequency weight: {round(model.coef_[0][1], 4)}")


# ================================================================
# EVALUATE MODEL PERFORMANCE
#
# Here we test the model using the test dataset
# ================================================================

print("\nEvaluating...")

# Predict labels for test dataset
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round(accuracy * 100, 2)}%")

# Show detailed classification metrics
print(classification_report(
    y_test,
    y_pred,
    target_names=["NOT SPAM", "SPAM"]
))


# ================================================================
# MANUAL TESTS
# Test the model with example transactions
# ================================================================

print("\nManual tests:")

tests = [
    [150,  2,  "Normal"],       # expected normal
    [4500, 35, "Spam"],         # expected spam
    [900,  15, "Suspicious"],   # borderline case
]

for amount, freq, label in tests:

    # scale input features
    scaled = scaler.transform([[amount, freq]])

    # predict class
    pred = model.predict(scaled)[0]

    # get spam probability
    prob = model.predict_proba(scaled)[0][1]

    # convert prediction to readable text
    result = "SPAM" if pred == 1 else "NOT SPAM"

    # print result
    print(f"Amount:{amount:5} Freq:{freq:2} | {label:12} | {result} ({round(prob*100,1)}%)")


# ================================================================
# SAVE METRICS
#
# Metrics help track experiment performance
# These are often tracked by DVC / MLflow / Azure ML
# ================================================================

print("\nSaving metrics...")

# Determine metrics folder
metrics_path = args.output_metrics if args.output_metrics else "metrics"

# Create folder if needed
os.makedirs(metrics_path, exist_ok=True)

# Prepare metrics dictionary
metrics = {
    "accuracy": round(accuracy * 100, 2),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "spam_count": int(y.sum()),
    "normal_count": int((y == 0).sum())
}

# Save metrics as JSON file
with open(os.path.join(metrics_path, "scores.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print(f"✅ Metrics: {metrics}")


# ================================================================
# SAVE TRAINED MODEL AND SCALER
#
# These files will be used later during deployment
# ================================================================

print("\nSaving model + scaler...")

# Determine model folder
model_path = args.output_model if args.output_model else "models"

# Create folder if needed
os.makedirs(model_path, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(model_path, "spam_model.pkl"))

# Save scaler
joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))

print(f"✅ spam_model.pkl saved!")
print(f"✅ scaler.pkl saved!")

print("\nTraining complete!")