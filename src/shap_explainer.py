import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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

# From Stage 1 we discovered that fraud occurs in
# TRANSFER and CASH_OUT transactions.

df = df[
    df["type"].isin(["TRANSFER", "CASH_OUT"])
].copy()


# ============================================================
# 3. SORT BY TIME
# ============================================================

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

# Account IDs are identifiers rather than useful
# generalizable transaction features.

X = X.drop(
    columns=[
        "nameOrig",
        "nameDest",
    ]
)


# ============================================================
# 6. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# Same 70/30 chronological split used in our
# original LightGBM and imbalance experiments.

split_step = X[
    "step"
].quantile(0.70)


train_mask = (
    X["step"] <= split_step
)

test_mask = (
    X["step"] > split_step
)


X_train = X[
    train_mask
].copy()

y_train = y[
    train_mask
].copy()

X_test = X[
    test_mask
].copy()

y_test = y[
    test_mask
].copy()


print(
    "\nTraining rows:",
    len(X_train)
)

print(
    "Test rows:",
    len(X_test)
)

print(
    "Training time:",
    X_train["step"].min(),
    "→",
    X_train["step"].max()
)

print(
    "Test time:",
    X_test["step"].min(),
    "→",
    X_test["step"].max()
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
# 9. ENCODE DATA
# ============================================================

print(
    "\nEncoding training data..."
)

# Fit the encoder ONLY on training data.

X_train_encoded = (
    preprocessor.fit_transform(
        X_train
    )
)


# Apply the same encoder to test data.

X_test_encoded = (
    preprocessor.transform(
        X_test
    )
)


# ============================================================
# 10. APPLY SMOTE ONLY TO TRAINING DATA
# ============================================================

print(
    "Applying SMOTE..."
)

# SMOTE must NEVER be applied to the test set.
#
# The test set must remain representative of real
# future transactions.

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


# ============================================================
# 11. TRAIN LIGHTGBM
# ============================================================

print(
    "\nTraining LightGBM..."
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
# 12. GENERATE TEST PROBABILITIES
# ============================================================

print(
    "\nGenerating fraud probabilities..."
)

test_probs = (
    model.predict_proba(
        X_test_encoded
    )[:, 1]
)


# ============================================================
# 13. APPLY MONEY-OPTIMIZED THRESHOLD
# ============================================================

# Stage 3 found approximately 0.0060 as the
# money-optimized threshold.

money_threshold = 0.0060


test_predictions = (
    test_probs >= money_threshold
).astype(int)


# ============================================================
# 14. GET FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


print(
    "\nNumber of model features:",
    len(feature_names)
)


# ============================================================
# 15. SELECT SHAP SAMPLE
# ============================================================

# Explaining every one of the 818k test rows would be
# unnecessarily expensive.
#
# We explain 2,000 representative future transactions.

sample_size = min(
    2000,
    len(X_test_encoded)
)


rng = np.random.RandomState(
    42
)


sample_indices = rng.choice(
    len(X_test_encoded),
    size=sample_size,
    replace=False,
)


X_shap = X_test_encoded[
    sample_indices
]


print(
    "\nCalculating SHAP values for",
    sample_size,
    "test transactions..."
)


# ============================================================
# 16. CREATE SHAP EXPLAINER
# ============================================================

# TreeExplainer is optimized for tree-based models
# such as LightGBM.

explainer = shap.TreeExplainer(
    model
)


# Calculate SHAP explanations.

shap_values = explainer(
    X_shap
)


# ============================================================
# 17. SAVE SHAP SUMMARY PLOT
# ============================================================

print(
    "\nCreating SHAP summary plot..."
)


shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    show=False,
)


plt.tight_layout()


plt.savefig(
    "reports/shap_summary.png",
    dpi=200,
    bbox_inches="tight",
)


plt.close()


print(
    "Saved:"
)

print(
    "reports/shap_summary.png"
)


# ============================================================
# 18. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

# Mean absolute SHAP value measures how strongly
# each feature affects predictions overall.

mean_abs_shap = (
    np.abs(
        shap_values.values
    ).mean(axis=0)
)


feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    "\n" + "=" * 60
)

print(
    "TOP SHAP FEATURES"
)

print(
    "=" * 60
)


print(
    feature_importance.head(
        10
    ).to_string(
        index=False
    )
)


# Save feature importance.

feature_importance.to_csv(
    "reports/shap_feature_importance.csv",
    index=False,
)


# ============================================================
# 19. FIND FALSE POSITIVES
# ============================================================

# False positive:
#
# Actual transaction = legitimate
# Model prediction = fraud

false_positive_mask = (
    (y_test.values == 0)
    & (test_predictions == 1)
)


false_positive_indices = np.where(
    false_positive_mask
)[0]


print(
    "\n" + "=" * 60
)

print(
    "FALSE POSITIVES"
)

print(
    "=" * 60
)

print(
    "Total false positives:",
    len(false_positive_indices)
)


# ============================================================
# 20. SORT FALSE POSITIVES BY RISK
# ============================================================

# The highest-probability false positives are the
# transactions where the model was most confident
# and still wrong.

false_positive_indices = sorted(
    false_positive_indices,
    key=lambda i: test_probs[i],
    reverse=True,
)


# Keep the 10 worst false positives.

worst_false_positives = (
    false_positive_indices[:10]
)


# ============================================================
# 21. PRINT TOP 10 FALSE POSITIVES
# ============================================================

print(
    "\nTop 10 false positives:"
)


for rank, index in enumerate(
    worst_false_positives,
    start=1,
):

    original_row = X_test.iloc[
        index
    ]

    probability = test_probs[
        index
    ]


    print(
        "\n" + "-" * 60
    )

    print(
        f"#{rank}"
    )

    print(
        f"Fraud probability: "
        f"{probability:.6f}"
    )

    print(
        f"Transaction type: "
        f"{original_row['type']}"
    )

    print(
        f"Amount: "
        f"₹{original_row['amount']:,.2f}"
    )

    print(
        f"Old origin balance: "
        f"₹{original_row['oldbalanceOrg']:,.2f}"
    )

    print(
        f"New origin balance: "
        f"₹{original_row['newbalanceOrig']:,.2f}"
    )

    print(
        f"Step: "
        f"{original_row['step']}"
    )


# ============================================================
# 22. EXPLAIN THE WORST FALSE POSITIVE
# ============================================================

if len(worst_false_positives) > 0:

    # Select the highest-confidence false positive.

    worst_index = (
        worst_false_positives[0]
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "SHAP EXPLANATION: WORST FALSE POSITIVE"
    )

    print(
        "=" * 60
    )


    # IMPORTANT:
    #
    # Keep this as a 2D matrix.
    #
    # LightGBM expects:
    #
    # (rows, features)
    #
    # rather than:
    #
    # (features,)

    worst_row = X_test_encoded[
        worst_index:worst_index + 1
    ]


    # Calculate SHAP explanation.

    worst_explanation = explainer(
        worst_row
    )


    # Get the SHAP values for the single row.

    single_shap_values = (
        worst_explanation
        .values[0]
    )


    # Create a readable table.

    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": single_shap_values,
        }
    )


    # Absolute value tells us the strength
    # of the feature's influence.

    contributions[
        "absolute_value"
    ] = np.abs(
        contributions[
            "shap_value"
        ]
    )


    # Strongest contributors first.

    contributions = (
        contributions
        .sort_values(
            "absolute_value",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    print(
        "\nTop factors affecting this prediction:"
    )


    print(
        contributions.head(
            10
        ).to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Print plain-English interpretation
    # --------------------------------------------------------

    print(
        "\nInterpretation:"
    )


    for _, row in contributions.head(
        5
    ).iterrows():

        direction = (
            "increased"
            if row["shap_value"] > 0
            else "decreased"
        )


        print(
            f"- {row['feature']} "
            f"{direction} the fraud score "
            f"(SHAP {row['shap_value']:.4f})"
        )


    # Save the complete explanation.

    contributions.to_csv(
        "reports/worst_false_positive_shap.csv",
        index=False,
    )


    print(
        "\nSaved:"
    )

    print(
        "reports/worst_false_positive_shap.csv"
    )


# ============================================================
# 23. SAVE TOP FALSE POSITIVES
# ============================================================

if len(worst_false_positives) > 0:

    false_positive_rows = (
        X_test.iloc[
            worst_false_positives
        ].copy()
    )


    false_positive_rows[
        "fraud_probability"
    ] = test_probs[
        worst_false_positives
    ]


    false_positive_rows[
        "actual_fraud"
    ] = y_test.iloc[
        worst_false_positives
    ].values


    false_positive_rows.to_csv(
        "reports/top_false_positives.csv",
        index=False,
    )


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "SHAP ANALYSIS COMPLETE"
)

print(
    "=" * 60
)

print(
    "Saved files:"
)

print(
    "reports/shap_summary.png"
)

print(
    "reports/shap_feature_importance.csv"
)

print(
    "reports/top_false_positives.csv"
)

print(
    "reports/worst_false_positive_shap.csv"
)