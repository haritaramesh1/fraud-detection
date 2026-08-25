import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True,
)


# ============================================================
# 2. LOAD DATA
# ============================================================

file_path = "data/PS_20174392719_1491204439457_log.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)


# ============================================================
# 3. KEEP TRANSFER AND CASH_OUT
# ============================================================

df = df[
    df["type"].isin(
        ["TRANSFER", "CASH_OUT"]
    )
].copy()


# ============================================================
# 4. SORT BY TIME
# ============================================================

df = df.sort_values(
    "step"
).reset_index(drop=True)


# ============================================================
# 5. CREATE FEATURES AND TARGET
# ============================================================

y = df["isFraud"]

X = df.drop(
    columns=["isFraud"]
)


# Remove account identifiers.

X = X.drop(
    columns=[
        "nameOrig",
        "nameDest",
    ]
)


# ============================================================
# 6. TIME-BASED TRAINING DATA
# ============================================================

# Use the first 70% of time for training.
#
# The final 30% remains future/test data and is not
# used to train the production model.

split_step = X[
    "step"
].quantile(0.70)


train_mask = (
    X["step"] <= split_step
)


X_train = X[
    train_mask
].copy()

y_train = y[
    train_mask
].copy()


print(
    "\nTraining rows:",
    len(X_train)
)

print(
    "Training time:",
    X_train["step"].min(),
    "→",
    X_train["step"].max()
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
# 9. ENCODE TRAINING DATA
# ============================================================

print(
    "\nEncoding training data..."
)

X_train_encoded = (
    preprocessor.fit_transform(
        X_train
    )
)


# ============================================================
# 10. APPLY SMOTE
# ============================================================

print(
    "Applying SMOTE..."
)

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
    "Original rows:",
    len(X_train)
)

print(
    "SMOTE rows:",
    len(X_train_smote)
)


# ============================================================
# 11. TRAIN LIGHTGBM
# ============================================================

print(
    "\nTraining production LightGBM..."
)

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)


model.fit(
    X_train_smote,
    y_train_smote
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

model_path = (
    "models/finsentinel_model.joblib"
)

preprocessor_path = (
    "models/finsentinel_preprocessor.joblib"
)


joblib.dump(
    model,
    model_path,
)


joblib.dump(
    preprocessor,
    preprocessor_path,
)


print(
    "\n" + "=" * 60
)

print(
    "MODEL SAVED"
)

print(
    "=" * 60
)

print(
    "Model:"
)

print(
    model_path
)

print(
    "\nPreprocessor:"
)

print(
    preprocessor_path
)

print(
    "\nFinSentinel production model ready."
)