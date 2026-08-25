import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import average_precision_score

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

# From our exploration we discovered that all fraud in
# PaySim occurs in TRANSFER and CASH_OUT transactions.

df = df[
    df["type"].isin(["TRANSFER", "CASH_OUT"])
].copy()

print("Rows after filtering:", len(df))


# ============================================================
# 3. SORT BY TIME
# ============================================================

# step represents time.
#
# We sort first so our train/test split respects the
# chronological order of transactions.

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

# nameOrig and nameDest contain huge numbers of unique
# account IDs.
#
# We don't want the model memorizing individual accounts.

X = X.drop(
    columns=[
        "nameOrig",
        "nameDest",
    ]
)


# ============================================================
# 6. TIME-BASED 70/30 SPLIT
# ============================================================

# Find the 70th percentile of time.

split_step = X["step"].quantile(0.70)

print(
    "70th-percentile step:",
    split_step
)


# Everything up to the cutoff is training data.

train_mask = X["step"] <= split_step


# Everything after the cutoff is future test data.

test_mask = X["step"] > split_step


X_train = X[train_mask].copy()

X_test = X[test_mask].copy()

y_train = y[train_mask].copy()

y_test = y[test_mask].copy()


print(
    "\nTraining rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ============================================================
# 7. DEFINE FEATURES
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
# 8. CREATE PREPROCESSOR
# ============================================================

# The type column contains text:
#
# TRANSFER
# CASH_OUT
#
# OneHotEncoder converts these categories into numbers.

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
# 9. EFFICIENT RECALL @ 1% FPR FUNCTION
# ============================================================

def recall_at_one_percent_fpr(
    y_true,
    probabilities,
):
    """
    Find the maximum fraud recall while keeping
    the false-positive rate at or below 1%.

    This implementation is efficient because it sorts
    the predictions once instead of repeatedly scanning
    the entire test set.
    """

    # Convert labels to NumPy.
    y_true = np.asarray(
        y_true
    )

    # Convert probabilities to NumPy.
    probabilities = np.asarray(
        probabilities
    )


    # --------------------------------------------------------
    # Sort transactions from highest fraud probability
    # to lowest fraud probability.
    # --------------------------------------------------------

    order = np.argsort(
        -probabilities
    )


    sorted_probs = probabilities[
        order
    ]

    sorted_y = y_true[
        order
    ]


    # --------------------------------------------------------
    # Count legitimate and fraudulent transactions.
    # --------------------------------------------------------

    total_negative = np.sum(
        sorted_y == 0
    )

    total_positive = np.sum(
        sorted_y == 1
    )


    # --------------------------------------------------------
    # Calculate cumulative false positives.
    # --------------------------------------------------------

    false_positives = np.cumsum(
        sorted_y == 0
    )


    # --------------------------------------------------------
    # Calculate cumulative true positives.
    # --------------------------------------------------------

    true_positives = np.cumsum(
        sorted_y == 1
    )


    # --------------------------------------------------------
    # Calculate false-positive rate.
    # --------------------------------------------------------

    fpr = (
        false_positives
        / total_negative
    )


    # --------------------------------------------------------
    # Calculate recall.
    # --------------------------------------------------------

    recall = (
        true_positives
        / total_positive
    )


    # --------------------------------------------------------
    # Keep only operating points where FPR <= 1%.
    # --------------------------------------------------------

    valid = fpr <= 0.01


    # If there is no valid threshold, return zeros.

    if not np.any(valid):

        return (
            0.0,
            0.0,
            1.0,
        )


    # --------------------------------------------------------
    # Get indexes where FPR is within our limit.
    # --------------------------------------------------------

    valid_indices = np.where(
        valid
    )[0]


    # --------------------------------------------------------
    # Find the valid point with the highest recall.
    # --------------------------------------------------------

    best_index = valid_indices[
        np.argmax(
            recall[valid]
        )
    ]


    # --------------------------------------------------------
    # Store the best values.
    # --------------------------------------------------------

    best_recall = recall[
        best_index
    ]

    best_fpr = fpr[
        best_index
    ]

    best_threshold = sorted_probs[
        best_index
    ]


    return (
        best_recall,
        best_fpr,
        best_threshold,
    )


# ============================================================
# 10. EXPERIMENT 1 — CLASS WEIGHTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "EXPERIMENT 1: CLASS WEIGHTS"
)

print(
    "=" * 60
)


# Build LightGBM with balanced class weights.

weighted_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,

    # Give rare fraud examples more importance.
    class_weight="balanced",

    random_state=42,
    n_jobs=-1,
)


# Fit preprocessing ONLY on training data.

X_train_weighted = preprocessor.fit_transform(
    X_train
)


# Apply the already-fitted preprocessor to test data.

X_test_weighted = preprocessor.transform(
    X_test
)


print(
    "Training class-weight model..."
)


# Train the model.

weighted_model.fit(
    X_train_weighted,
    y_train
)


# Generate fraud probabilities.

weighted_probs = weighted_model.predict_proba(
    X_test_weighted
)[:, 1]


# Calculate PR-AUC.

weighted_pr_auc = average_precision_score(
    y_test,
    weighted_probs
)


# Calculate recall at 1% FPR.

(
    weighted_recall,
    weighted_fpr,
    weighted_threshold,
) = recall_at_one_percent_fpr(
    y_test,
    weighted_probs
)


print(
    f"PR-AUC: {weighted_pr_auc:.6f}"
)

print(
    f"Recall @ 1% FPR: "
    f"{weighted_recall:.6f}"
)

print(
    f"Actual FPR: "
    f"{weighted_fpr:.6f}"
)

print(
    f"Threshold: "
    f"{weighted_threshold:.6f}"
)


# ============================================================
# 11. EXPERIMENT 2 — SMOTE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "EXPERIMENT 2: SMOTE"
)

print(
    "=" * 60
)


print(
    "\nOriginal training distribution:"
)

print(
    y_train.value_counts()
)


# ------------------------------------------------------------
# IMPORTANT:
#
# SMOTE is applied ONLY to training data.
#
# We NEVER apply SMOTE to X_test or y_test.
#
# The test set must represent untouched future data.
# ------------------------------------------------------------


print(
    "\nApplying SMOTE to training data..."
)


smote = SMOTE(
    random_state=42,

    # 0.1 means the minority class becomes 10%
    # of the majority class after resampling.
    #
    # We deliberately don't create a fully 50/50 dataset
    # because that would create an enormous amount of
    # synthetic data.

    sampling_strategy=0.1,
)


X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_weighted,
        y_train
    )
)


print(
    "\nTraining distribution after SMOTE:"
)

print(
    pd.Series(
        y_train_smote
    ).value_counts()
)


# ============================================================
# 12. TRAIN SMOTE MODEL
# ============================================================

smote_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,

    # No class weighting here.
    #
    # SMOTE itself changed the training distribution.

    random_state=42,
    n_jobs=-1,
)


print(
    "\nTraining SMOTE model..."
)


smote_model.fit(
    X_train_smote,
    y_train_smote
)


# Generate probabilities on the ORIGINAL untouched test set.

smote_probs = smote_model.predict_proba(
    X_test_weighted
)[:, 1]


# Calculate PR-AUC.

smote_pr_auc = average_precision_score(
    y_test,
    smote_probs
)


# Calculate recall at 1% FPR.

(
    smote_recall,
    smote_fpr,
    smote_threshold,
) = recall_at_one_percent_fpr(
    y_test,
    smote_probs
)


print(
    f"PR-AUC: {smote_pr_auc:.6f}"
)

print(
    f"Recall @ 1% FPR: "
    f"{smote_recall:.6f}"
)

print(
    f"Actual FPR: "
    f"{smote_fpr:.6f}"
)

print(
    f"Threshold: "
    f"{smote_threshold:.6f}"
)


# ============================================================
# 13. EXPERIMENT 3 — NO IMBALANCE TREATMENT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "EXPERIMENT 3: NO TREATMENT"
)

print(
    "=" * 60
)


# Train ordinary LightGBM.
#
# No class_weight.
# No SMOTE.

normal_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)


print(
    "Training model without imbalance treatment..."
)


normal_model.fit(
    X_train_weighted,
    y_train
)


# Generate probabilities.

normal_probs = normal_model.predict_proba(
    X_test_weighted
)[:, 1]


# Calculate PR-AUC.

normal_pr_auc = average_precision_score(
    y_test,
    normal_probs
)


# Find the best threshold that keeps FPR <= 1%.

(
    normal_recall,
    normal_fpr,
    normal_threshold,
) = recall_at_one_percent_fpr(
    y_test,
    normal_probs
)


print(
    f"PR-AUC: {normal_pr_auc:.6f}"
)

print(
    f"Recall @ 1% FPR: "
    f"{normal_recall:.6f}"
)

print(
    f"Actual FPR: "
    f"{normal_fpr:.6f}"
)

print(
    f"Threshold: "
    f"{normal_threshold:.6f}"
)


# ============================================================
# 14. BUILD FINAL COMPARISON TABLE
# ============================================================

results = pd.DataFrame(
    {
        "Strategy": [
            "Class weights",
            "SMOTE",
            "No treatment + threshold",
        ],

        "PR-AUC": [
            weighted_pr_auc,
            smote_pr_auc,
            normal_pr_auc,
        ],

        "Recall @ 1% FPR": [
            weighted_recall,
            smote_recall,
            normal_recall,
        ],

        "Actual FPR": [
            weighted_fpr,
            smote_fpr,
            normal_fpr,
        ],

        "Threshold": [
            weighted_threshold,
            smote_threshold,
            normal_threshold,
        ],
    }
)


# ============================================================
# 15. PRINT FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "IMBALANCE TOURNAMENT"
)

print(
    "=" * 60
)


print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results.to_csv(
    "reports/imbalance_results.csv",
    index=False
)


print(
    "\nResults saved to:"
)

print(
    "reports/imbalance_results.csv"
)


print(
    "\nTournament complete."
)