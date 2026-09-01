"""
FinSentinel Final Test Evaluation

Evaluates the saved production model on the untouched
future test period.

Training:  step 1 -> 323
Final test: step 324 -> 743

This gives us a genuine out-of-sample evaluation.
"""

import os
import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"

MODEL_PATH = "models/finsentinel_model.joblib"

PREPROCESSOR_PATH = "models/finsentinel_preprocessor.joblib"

OUTPUT_PATH = "reports/final_test_results.csv"

TEST_START = 324


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total rows: {len(df):,}")


# ============================================================
# FINAL TEST SPLIT
# ============================================================

test_df = df[df["step"] >= TEST_START].copy()

print("")
print("=" * 60)
print("FINAL TEST SET")
print("=" * 60)

print(f"Test rows: {len(test_df):,}")

print(
    f"Test time: "
    f"{test_df['step'].min()} → "
    f"{test_df['step'].max()}"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]

TARGET = "isFraud"

X_test = test_df[FEATURES]

y_test = test_df[TARGET]


# ============================================================
# LOAD SAVED MODEL
# ============================================================

print("")
print("Loading production model...")

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

print("Model loaded successfully.")


# ============================================================
# ENCODE TEST DATA
# ============================================================

print("")
print("Encoding final test data...")

X_test_encoded = preprocessor.transform(
    X_test
)


# ============================================================
# PREDICT
# ============================================================

print("")
print("Generating fraud probabilities...")

probabilities = model.predict_proba(
    X_test_encoded
)[:, 1]


# ============================================================
# FINAL TEST METRICS
# ============================================================

pr_auc = average_precision_score(
    y_test,
    probabilities
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# Use the money-optimized threshold
THRESHOLD = 0.006

predictions = (
    probabilities >= THRESHOLD
).astype(int)


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()


# ============================================================
# PRINT RESULTS
# ============================================================

print("")
print("=" * 60)
print("FINSENTINEL FINAL TEST RESULTS")
print("=" * 60)

print(f"PR-AUC:              {pr_auc:.6f}")
print(f"ROC-AUC:             {roc_auc:.6f}")
print(f"Threshold:           {THRESHOLD:.4f}")
print(f"Precision:           {precision:.6f}")
print(f"Recall:              {recall:.6f}")
print(f"F1 score:            {f1:.6f}")

print("")
print("CONFUSION MATRIX")
print("-" * 60)

print(f"True negatives:      {tn:,}")
print(f"False positives:     {fp:,}")
print(f"False negatives:     {fn:,}")
print(f"True positives:      {tp:,}")


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "test_rows": len(test_df),
            "test_start_step": test_df["step"].min(),
            "test_end_step": test_df["step"].max(),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "threshold": THRESHOLD,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        }
    ]
)

os.makedirs(
    "reports",
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("")
print("=" * 60)
print("FINAL EVALUATION COMPLETE")
print("=" * 60)

print("")
print("Saved:")
print(OUTPUT_PATH)