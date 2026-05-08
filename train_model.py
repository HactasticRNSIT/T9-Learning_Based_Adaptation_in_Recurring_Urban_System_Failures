"""
train_model.py — RR Nagar Urban Resilience: Model Training Brain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run this FIRST before launching app.py

Usage:
    python train_model.py
    
Outputs:
    traffic_model.pkl   — trained RandomForestRegressor
    model_columns.pkl   — ordered feature column list
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CSV_PATH        = "rr_nagar_traffic_dataset.csv"
MODEL_OUT       = "traffic_model.pkl"
COLUMNS_OUT     = "model_columns.pkl"
TARGET_COL      = "congestion_level"
ZONE_COL        = "zone"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
N_ESTIMATORS    = 200
MAX_DEPTH       = 15
MIN_SAMPLES_LEAF = 2


def banner(text: str) -> None:
    width = 60
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD & INSPECT DATA
# ─────────────────────────────────────────────────────────────
banner("STEP 1 › Loading Dataset")

if not os.path.exists(CSV_PATH):
    print(f"[ERROR] '{CSV_PATH}' not found. Place the CSV in the same directory.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"  ✔  Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"  ✔  Columns  : {df.columns.tolist()}")
print(f"  ✔  Zones    : {sorted(df[ZONE_COL].unique().tolist())}")
print(f"  ✔  Nulls    : {df.isnull().sum().sum()} total missing values")
print(f"  ✔  Target   : congestion_level ∈ [{df[TARGET_COL].min():.3f}, {df[TARGET_COL].max():.3f}]")
print(f"             mean={df[TARGET_COL].mean():.3f}, std={df[TARGET_COL].std():.3f}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────────
banner("STEP 2 › Preprocessing")

# Drop duplicates if any
before = len(df)
df = df.drop_duplicates()
print(f"  ✔  Dropped {before - len(df)} duplicate rows")

# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# One-Hot Encode the zone column
X = pd.get_dummies(X, columns=[ZONE_COL], prefix="zone", drop_first=False)
print(f"  ✔  One-Hot Encoded '{ZONE_COL}' → {[c for c in X.columns if c.startswith('zone_')]}")
print(f"  ✔  Final feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

# Save the exact column order (critical for inference alignment)
model_columns = X.columns.tolist()
print(f"  ✔  Feature columns saved: {model_columns}")


# ─────────────────────────────────────────────────────────────
# STEP 3 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
banner("STEP 3 › Train / Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"  ✔  Train set : {len(X_train):,} samples ({100*(1-TEST_SIZE):.0f}%)")
print(f"  ✔  Test  set : {len(X_test):,} samples ({100*TEST_SIZE:.0f}%)")


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────
banner("STEP 4 › Training RandomForestRegressor")

model = RandomForestRegressor(
    n_estimators    = N_ESTIMATORS,
    max_depth       = MAX_DEPTH,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    n_jobs          = -1,           # use all CPU cores
    random_state    = RANDOM_STATE,
)

print(f"  ⏳ Training with {N_ESTIMATORS} trees (max_depth={MAX_DEPTH})...")
model.fit(X_train, y_train)
print("  ✔  Training complete!")


# ─────────────────────────────────────────────────────────────
# STEP 5 — EVALUATE MODEL
# ─────────────────────────────────────────────────────────────
banner("STEP 5 › Model Evaluation")

y_pred = model.predict(X_test)
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)
rmse  = np.sqrt(np.mean((y_test - y_pred) ** 2))

print(f"  ✔  R²  Score  : {r2:.4f}  {'(Excellent)' if r2 > 0.85 else '(Good)' if r2 > 0.7 else '(Fair)'}")
print(f"  ✔  MAE        : {mae:.4f}  (mean absolute error in congestion units)")
print(f"  ✔  RMSE       : {rmse:.4f}")

# Breakdown by congestion severity bucket
y_test_arr = np.array(y_test)
buckets = {
    "Low  (0.0–0.4)":   (y_test_arr < 0.4),
    "Med  (0.4–0.7)":   (y_test_arr >= 0.4) & (y_test_arr < 0.7),
    "High (0.7–1.0)":   (y_test_arr >= 0.7),
}
print("\n  Accuracy by severity bucket:")
for label, mask in buckets.items():
    if mask.sum() > 0:
        bucket_mae = mean_absolute_error(y_test_arr[mask], y_pred[mask])
        print(f"    {label} : {mask.sum():>4} samples | MAE = {bucket_mae:.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 6 — FEATURE IMPORTANCE (the "why failures happen" intel)
# ─────────────────────────────────────────────────────────────
banner("STEP 6 › Feature Importance — Why Failures Happen")

importances = model.feature_importances_
fi_df = (
    pd.DataFrame({"feature": model_columns, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print(f"\n  {'Rank':<5} {'Feature':<35} {'Importance':>12}  Visual")
print("  " + "─" * 65)
for i, row in fi_df.iterrows():
    bar   = "█" * int(row["importance"] * 60)
    rank  = i + 1
    print(f"  {rank:<5} {row['feature']:<35} {row['importance']:>10.4f}  {bar}")

print("\n  ► Insight: Features at the top are the STRONGEST predictors of")
print("    urban congestion failure. City planners should monitor these first.")


# ─────────────────────────────────────────────────────────────
# STEP 7 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
banner("STEP 7 › Saving Model Artifacts")

joblib.dump(model,         MODEL_OUT)
joblib.dump(model_columns, COLUMNS_OUT)

print(f"  ✔  Model saved   → {MODEL_OUT}  ({os.path.getsize(MODEL_OUT)/1024:.1f} KB)")
print(f"  ✔  Columns saved → {COLUMNS_OUT}  ({os.path.getsize(COLUMNS_OUT)/1024:.1f} KB)")

banner("TRAINING COMPLETE — Ready to launch app.py")
print("  Run:  streamlit run app.py\n")
