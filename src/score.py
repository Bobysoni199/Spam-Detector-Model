"""
src/score.py
================================================================
Azure ML Scoring Script
- Loaded once when endpoint starts (init)
- Called on every API request (run)
================================================================
"""

import json
import joblib
import os
import numpy as np

model  = None
scaler = None

def init():
    global model, scaler

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    print(f"Model dir: {model_dir}")
    print(f"Files: {os.listdir(model_dir)}")

    # model registered from pipeline output
    # files are directly in model_dir (no subfolder!)
    model_file  = os.path.join(model_dir, "spam_model.pkl")
    scaler_file = os.path.join(model_dir, "scaler.pkl")

    # fallback - check one level deeper if not found
    if not os.path.exists(model_file):
        print("Not found directly, checking subfolders...")
        for item in os.listdir(model_dir):
            sub = os.path.join(model_dir, item)
            if os.path.isdir(sub):
                print(f"Subfolder: {item} → {os.listdir(sub)}")
                if os.path.exists(os.path.join(sub, "spam_model.pkl")):
                    model_file  = os.path.join(sub, "spam_model.pkl")
                    scaler_file = os.path.join(sub, "scaler.pkl")
                    print(f"Found in subfolder: {item}")
                    break

    model  = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    print("✅ Model + scaler loaded!")

def run(raw_data):
    data         = json.loads(raw_data)
    input_data   = np.array(data["data"])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)
    probability  = model.predict_proba(input_scaled)

    results = []
    for i in range(len(prediction)):
        pred      = prediction[i]
        prob      = probability[i][1]
        label     = "SPAM" if pred == 1 else "NOT SPAM"
        spam_prob = round(float(prob) * 100, 2)
        results.append({
            "prediction":       label,
            "spam_probability": spam_prob
        })

    return {"results": results}