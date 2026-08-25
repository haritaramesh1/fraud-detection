import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)


# ============================================================
# 2. KEEP TRANSFER AND CASH_OUT
# ============================================================

df = df[
    df["type"].isin(["TRANSFER", "CASH_OUT"])
].copy()


# ============================================================
# 3. SORT BY TIME
# ============================================================

df = df.sort_values("step").reset_index(drop=True)


# ============================================================
# 4. DEFINE TARGET
# ============================================================

target = "isFraud"

y = df[target]

X = df.drop(columns=[target])


# ============================================================
# 5. REMOVE ACCOUNT IDs
# ============================================================

X = X.drop(
    columns=["nameOrig", "nameDest"]
)


# ============================================================
# 6. TIME-BASED SPLIT
# ============================================================

split_step = X["step"].quantile(0.70)

train_mask = X["step"] <= split_step
test_mask = X["step"] > split_step

X_train = X[train_mask].copy()
X_test = X[test_mask].copy()

y_train = y[train_mask].copy()
y_test = y[test_mask].copy()


# ============================================================
# 7. DEFINE FEATURES
# ============================================================

categorical_features = ["type"]

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
# 8. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
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
# 9. LOGISTIC REGRESSION
# ============================================================

# class_weight="balanced" tells the model that fraud
# is much more important because it is extremely rare.

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# 10. TRAIN
# ============================================================

print("Training Logistic Regression...")

model.fit(
    X_train,
    y_train,
)


# ============================================================
# 11. GET FRAUD PROBABILITIES
# ============================================================

# We want probabilities rather than just 0/1 predictions.
#
# Example:
#
# 0.002 → probably legitimate
# 0.30  → somewhat suspicious
# 0.95  → highly suspicious

probs = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 12. CALCULATE PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    probs,
)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"PR-AUC: {pr_auc:.6f}")

print("\nBaseline complete.")