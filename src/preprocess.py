import pandas as pd
import numpy as np
import os
import argparse

# ================================================================
# ARGUMENT PARSER
# ================================================================

parser = argparse.ArgumentParser()

parser.add_argument("--output_data", type=str, help="Path to save clean data")

# NEW ARGUMENTS 
parser.add_argument("--normal_count", type=int, default=700)
parser.add_argument("--spam_count", type=int, default=300)

args = parser.parse_args()

NORMAL_COUNT = args.normal_count
SPAM_COUNT = args.spam_count


print(f"Normal transactions: {NORMAL_COUNT}")
print(f"Spam transactions: {SPAM_COUNT}")

# ================================================================
# GENERATE DATA
# ================================================================

np.random.seed(42)

normal = pd.DataFrame({
    "amount": np.random.uniform(10, 500, NORMAL_COUNT),
    "frequency": np.random.randint(1, 5, NORMAL_COUNT),
    "label": 0
})

spam = pd.DataFrame({
    "amount": np.random.uniform(800, 5000, SPAM_COUNT),
    "frequency": np.random.randint(10, 50, SPAM_COUNT),
    "label": 1
})

df = pd.concat([normal, spam])
df = df.sample(frac=1).reset_index(drop=True)

print(f"Raw data rows: {len(df)}")

# ================================================================
# CLEAN DATA
# ================================================================

df["amount"] = df["amount"].fillna(df["amount"].mean())
df["frequency"] = df["frequency"].fillna(df["frequency"].mean())

df = df.drop_duplicates()

df = df[df["amount"] > 0]
df = df[df["amount"] <= 10000]
df = df[df["frequency"] > 0]

print(f"Clean rows: {len(df)}")

# ================================================================
# SAVE OUTPUT
# ================================================================

output_path = args.output_data if args.output_data else "data"
os.makedirs(output_path, exist_ok=True)

save_path = os.path.join(output_path, "clean_data.csv")

df.to_csv(save_path, index=False)

print(f"Saved to: {save_path}")