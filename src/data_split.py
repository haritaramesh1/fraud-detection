import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. LOAD THE DATA
# ============================================================

file_path = "data/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. KEEP ONLY TRANSFER AND CASH_OUT
# ============================================================

# Our exploration showed that every fraud case belongs
# to these two transaction types.

df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

print("After filtering:", df.shape)


# ============================================================
# 3. SORT BY TIME
# ============================================================

# "step" represents time.
#
# Sorting ensures that earlier transactions appear before
# later transactions.

df = df.sort_values("step").reset_index(drop=True)


# ============================================================
# 4. FIND THE 70TH-PERCENTILE TIME CUTOFF
# ============================================================

# We use TIME rather than randomly selecting rows.
#
# This means:
#
#     EARLY 70% → TRAIN
#     LATE 30%  → TEST

split_step = df["step"].quantile(0.70)

print("\n70th-percentile step:", split_step)


# ============================================================
# 5. CREATE TRAINING DATA
# ============================================================

train_df = df[df["step"] <= split_step].copy()


# ============================================================
# 6. CREATE TEST DATA
# ============================================================

test_df = df[df["step"] > split_step].copy()


# ============================================================
# 7. CHECK THE SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TIME-BASED SPLIT")
print("=" * 60)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "\nTraining time:",
    train_df["step"].min(),
    "→",
    train_df["step"].max(),
)

print(
    "Testing time:",
    test_df["step"].min(),
    "→",
    test_df["step"].max(),
)


# ============================================================
# 8. CHECK FRAUD RATES
# ============================================================

print("\n" + "=" * 60)
print("FRAUD RATE")
print("=" * 60)

print(
    "Training fraud rate:",
    f"{train_df['isFraud'].mean() * 100:.4f}%"
)

print(
    "Testing fraud rate:",
    f"{test_df['isFraud'].mean() * 100:.4f}%"
)


# ============================================================
# 9. VERIFY THAT TIME DOES NOT OVERLAP
# ============================================================

print("\n" + "=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)

print(
    "Latest training step:",
    train_df["step"].max()
)

print(
    "Earliest testing step:",
    test_df["step"].min()
)

if train_df["step"].max() < test_df["step"].min():
    print("PASS: Training occurs before testing.")
else:
    print("WARNING: Time overlap detected!")


# ============================================================
# 10. SEPARATE FEATURES AND TARGET
# ============================================================

target = "isFraud"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]


# ============================================================
# 11. REMOVE ACCOUNT IDs
# ============================================================

# nameOrig and nameDest are account identifiers.
#
# We don't want the model memorizing millions of unique IDs.

X_train = X_train.drop(
    columns=["nameOrig", "nameDest"]
)

X_test = X_test.drop(
    columns=["nameOrig", "nameDest"]
)


# ============================================================
# 12. CHECK THE FINAL DATA
# ============================================================

print("\n" + "=" * 60)
print("FINAL TRAIN / TEST DATA")
print("=" * 60)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nFeature columns:")
print(X_train.columns.tolist())

# ============================================================
# 13. ONE-HOT ENCODE THE TRANSACTION TYPE
# ============================================================

# "type" is text, such as:
#
# TRANSFER
# CASH_OUT
#
# Machine-learning models need numerical values.
#
# OneHotEncoder converts:
#
# TRANSFER  -> [1, 0]
# CASH_OUT  -> [0, 1]
#
# IMPORTANT:
# We fit the encoder ONLY on X_train.
# We then use that already-fitted encoder on X_test.
#
# This prevents information from the future test data
# leaking into our training process.

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
# 14. FIT ONLY ON TRAINING DATA
# ============================================================

preprocessor.fit(X_train)


# ============================================================
# 15. TRANSFORM TRAIN AND TEST
# ============================================================

X_train_encoded = preprocessor.transform(X_train)

X_test_encoded = preprocessor.transform(X_test)


# ============================================================
# 16. CHECK THE RESULT
# ============================================================

print("\n" + "=" * 60)
print("ONE-HOT ENCODING")
print("=" * 60)

print(
    "Original training features:",
    X_train.shape[1]
)

print(
    "Encoded training features:",
    X_train_encoded.shape[1]
)

print(
    "Encoded test features:",
    X_test_encoded.shape[1]
)

print("\nEncoding complete.")