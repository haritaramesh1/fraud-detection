import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from sklearn.metrics import average_precision_score


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)


# ============================================================
# 2. KEEP TRANSACTION TYPES WHERE FRAUD EXISTS
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
# 5. REMOVE ACCOUNT IDENTIFIERS
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
# 7. FEATURES
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
# 9. LIGHTGBM MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,

                # Fraud is extremely rare.
                # This makes the model pay more attention
                # to fraudulent transactions.

                class_weight="balanced",

                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# 10. TRAIN
# ============================================================

print("Training LightGBM...")

model.fit(
    X_train,
    y_train,
)


# ============================================================
# 11. FRAUD PROBABILITIES
# ============================================================

probs = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 12. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    probs,
)


print("\n" + "=" * 60)
print("LIGHTGBM RESULTS")
print("=" * 60)

print(f"PR-AUC: {pr_auc:.6f}")


# ============================================================
# 13. COMPARE AGAINST OUR BASELINE
# ============================================================

logistic_pr_auc = 0.660463

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"Logistic Regression: {logistic_pr_auc:.6f}"
)

print(
    f"LightGBM:            {pr_auc:.6f}"
)

print(
    f"Improvement:         "
    f"{pr_auc - logistic_pr_auc:+.6f}"
)