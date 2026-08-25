import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
)

from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/PS_20174392719_1491204439457_log.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)


# ============================================================
# 2. KEEP TRANSACTION TYPES WHERE FRAUD EXISTS
# ============================================================

# Our exploration showed that all fraud cases occur
# in TRANSFER and CASH_OUT transactions.

df = df[
    df["type"].isin(["TRANSFER", "CASH_OUT"])
].copy()


# ============================================================
# 3. SORT BY TIME
# ============================================================

# "step" represents time.
#
# Sorting ensures our evaluation follows the real
# direction of time.

df = df.sort_values(
    "step"
).reset_index(drop=True)


# ============================================================
# 4. DEFINE TARGET AND FEATURES
# ============================================================

target = "isFraud"

y = df[target]

X = df.drop(
    columns=[target]
)


# ============================================================
# 5. REMOVE ACCOUNT IDENTIFIERS
# ============================================================

# We don't want the model memorizing individual
# account IDs.

X = X.drop(
    columns=[
        "nameOrig",
        "nameDest",
    ]
)


# ============================================================
# 6. CREATE TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# We now need THREE chronological periods:
#
# EARLY 60%  -> model training
# NEXT 10%   -> probability calibration
# FINAL 30%  -> completely untouched test
#
# We must NOT use the final test period to choose
# calibration or the money threshold.

train_cutoff = X["step"].quantile(0.60)

validation_cutoff = X["step"].quantile(0.70)


# Training period.

train_mask = (
    X["step"] <= train_cutoff
)


# Calibration/validation period.

validation_mask = (
    (X["step"] > train_cutoff)
    & (X["step"] <= validation_cutoff)
)


# Final future test period.

test_mask = (
    X["step"] > validation_cutoff
)


# Create training data.

X_train = X[
    train_mask
].copy()

y_train = y[
    train_mask
].copy()


# Create validation data.

X_validation = X[
    validation_mask
].copy()

y_validation = y[
    validation_mask
].copy()


# Create final test data.

X_test = X[
    test_mask
].copy()

y_test = y[
    test_mask
].copy()


# ============================================================
# 7. DISPLAY THE TIME SPLIT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TIME SPLIT"
)

print(
    "=" * 60
)

print(
    "Training:",
    len(X_train),
    "rows"
)

print(
    "Validation:",
    len(X_validation),
    "rows"
)

print(
    "Final test:",
    len(X_test),
    "rows"
)


print(
    "\nTraining time:",
    X_train["step"].min(),
    "→",
    X_train["step"].max()
)

print(
    "Validation time:",
    X_validation["step"].min(),
    "→",
    X_validation["step"].max()
)

print(
    "Test time:",
    X_test["step"].min(),
    "→",
    X_test["step"].max()
)


# ============================================================
# 8. DEFINE FEATURE TYPES
# ============================================================

categorical_features = [
    "type"
]


numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]


# ============================================================
# 9. CREATE PREPROCESSOR
# ============================================================

# "type" is categorical text.
#
# OneHotEncoder converts:
#
# TRANSFER -> numeric representation
# CASH_OUT -> numeric representation
#
# The encoder will be fitted ONLY on training data.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# 10. FIT PREPROCESSING ONLY ON TRAINING DATA
# ============================================================

X_train_encoded = preprocessor.fit_transform(
    X_train
)


# Apply the already-fitted encoder to validation data.

X_validation_encoded = preprocessor.transform(
    X_validation
)


# Apply the already-fitted encoder to final test data.

X_test_encoded = preprocessor.transform(
    X_test
)


# ============================================================
# 11. APPLY SMOTE ONLY TO TRAINING DATA
# ============================================================

print(
    "\nApplying SMOTE to training data..."
)

# IMPORTANT:
#
# We NEVER apply SMOTE to validation or test data.
#
# Those datasets must represent real transactions.

smote = SMOTE(
    random_state=42,
    sampling_strategy=0.1,
)


X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_encoded,
        y_train
    )
)


print(
    "Original training rows:",
    len(X_train)
)

print(
    "SMOTE training rows:",
    len(X_train_smote)
)


print(
    "\nTraining class distribution after SMOTE:"
)

print(
    pd.Series(
        y_train_smote
    ).value_counts()
)


# ============================================================
# 12. TRAIN LIGHTGBM
# ============================================================

print(
    "\nTraining LightGBM..."
)

base_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)


base_model.fit(
    X_train_smote,
    y_train_smote
)


# ============================================================
# 13. CALIBRATE THE ALREADY-TRAINED MODEL
# ============================================================

print(
    "\nCalibrating probabilities..."
)


# Modern scikit-learn uses FrozenEstimator instead of
# the older cv="prefit" approach.
#
# FrozenEstimator tells sklearn:
#
# "This model is already trained.
#  Do not retrain it.
#  Only learn the probability calibration layer."

calibrated_model = CalibratedClassifierCV(
    FrozenEstimator(
        base_model
    ),
    method="sigmoid",
)


# IMPORTANT:
#
# Calibration is learned using the validation period only.
#
# The final test period remains untouched.

calibrated_model.fit(
    X_validation_encoded,
    y_validation
)


# ============================================================
# 14. GENERATE CALIBRATED TEST PROBABILITIES
# ============================================================

test_probs = calibrated_model.predict_proba(
    X_test_encoded
)[:, 1]


# ============================================================
# 15. EVALUATE CALIBRATED MODEL
# ============================================================

pr_auc = average_precision_score(
    y_test,
    test_probs
)


brier = brier_score_loss(
    y_test,
    test_probs
)


print(
    "\n" + "=" * 60
)

print(
    "CALIBRATED MODEL"
)

print(
    "=" * 60
)

print(
    f"PR-AUC: {pr_auc:.6f}"
)

print(
    f"Brier score: {brier:.6f}"
)


# ============================================================
# 16. CALCULATE BUSINESS COST ASSUMPTIONS
# ============================================================

# A missed fraud costs the transaction amount.
#
# A false alarm costs ₹50 of customer-support time.

average_fraud_amount = df.loc[
    df["isFraud"] == 1,
    "amount"
].mean()


false_alarm_cost = 50.0


print(
    "\n" + "=" * 60
)

print(
    "BUSINESS COST ASSUMPTIONS"
)

print(
    "=" * 60
)

print(
    f"Average fraud amount: "
    f"₹{average_fraud_amount:,.2f}"
)

print(
    f"False-alarm cost: "
    f"₹{false_alarm_cost:,.2f}"
)


# ============================================================
# 17. SWEEP POSSIBLE THRESHOLDS
# ============================================================

# We test many possible probability thresholds.
#
# Example:
#
# threshold = 0.10
# threshold = 0.20
# threshold = 0.30
# ...
#
# For each threshold we calculate the expected business cost.

thresholds = np.linspace(
    0.001,
    0.999,
    200
)


results = []


# ============================================================
# 18. EVALUATE EACH THRESHOLD
# ============================================================

for threshold in thresholds:

    # Flag a transaction if its fraud probability
    # is greater than or equal to the threshold.

    predictions = (
        test_probs >= threshold
    ).astype(int)


    # Create an evaluation dataframe.

    evaluation = X_test.copy()

    evaluation["actual"] = (
        y_test.values
    )

    evaluation["predicted"] = (
        predictions
    )


    # --------------------------------------------------------
    # MISSED FRAUD
    # --------------------------------------------------------

    # Fraud that the model failed to flag.

    missed_fraud = evaluation[
        (evaluation["actual"] == 1)
        & (evaluation["predicted"] == 0)
    ]


    # --------------------------------------------------------
    # FALSE ALARMS
    # --------------------------------------------------------

    # Legitimate transactions incorrectly flagged.

    false_alarms = evaluation[
        (evaluation["actual"] == 0)
        & (evaluation["predicted"] == 1)
    ]


    # --------------------------------------------------------
    # FRAUD LOSS
    # --------------------------------------------------------

    # We assume the amount of each missed fraud
    # represents the financial loss.

    fraud_loss = missed_fraud[
        "amount"
    ].sum()


    # --------------------------------------------------------
    # CUSTOMER SUPPORT COST
    # --------------------------------------------------------

    support_cost = (
        len(false_alarms)
        * false_alarm_cost
    )


    # --------------------------------------------------------
    # TOTAL EXPECTED COST
    # --------------------------------------------------------

    total_cost = (
        fraud_loss
        + support_cost
    )


    # --------------------------------------------------------
    # COST PER 1,000 TRANSACTIONS
    # --------------------------------------------------------

    transaction_count = len(
        evaluation
    )


    cost_per_1000 = (
        total_cost
        / transaction_count
        * 1000
    )


    # --------------------------------------------------------
    # FRAUD RECALL
    # --------------------------------------------------------

    actual_fraud = (
        evaluation["actual"] == 1
    ).sum()


    caught_fraud = (
        (evaluation["actual"] == 1)
        & (evaluation["predicted"] == 1)
    ).sum()


    if actual_fraud > 0:

        recall = (
            caught_fraud
            / actual_fraud
        )

    else:

        recall = 0.0


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append(
        {
            "threshold": threshold,
            "cost_per_1000": cost_per_1000,
            "recall": recall,
            "false_alarms": len(
                false_alarms
            ),
            "missed_fraud": len(
                missed_fraud
            ),
        }
    )


# ============================================================
# 19. CONVERT RESULTS TO DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 20. FIND LOWEST-COST THRESHOLD
# ============================================================

best_index = results_df[
    "cost_per_1000"
].idxmin()


best = results_df.loc[
    best_index
]


best_threshold = best[
    "threshold"
]

best_cost = best[
    "cost_per_1000"
]

best_recall = best[
    "recall"
]

best_false_alarms = best[
    "false_alarms"
]

best_missed_fraud = best[
    "missed_fraud"
]


# ============================================================
# 21. PRINT MONEY-OPTIMIZED THRESHOLD
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MONEY-OPTIMIZED THRESHOLD"
)

print(
    "=" * 60
)

print(
    f"Best threshold: "
    f"{best_threshold:.4f}"
)

print(
    f"Expected cost per 1,000: "
    f"₹{best_cost:,.2f}"
)

print(
    f"Fraud recall: "
    f"{best_recall:.4%}"
)

print(
    f"False alarms: "
    f"{int(best_false_alarms):,}"
)

print(
    f"Missed fraud: "
    f"{int(best_missed_fraud):,}"
)


# ============================================================
# 22. SAVE THRESHOLD RESULTS
# ============================================================

output_file = (
    "reports/money_threshold_results.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


print(
    "\nResults saved to:"
)

print(
    output_file
)


# ============================================================
# 23. FINISHED
# ============================================================

print(
    "\nCalibration + money threshold complete."
)