import pandas as pd

from evidently import Report
from evidently import Dataset
from evidently import DataDefinition
from evidently.presets import DataDriftPreset


# ============================================================
# 1. LOAD PAYSim DATA
# ============================================================

file_path = "data/PS_20174392719_1491204439457_log.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)


# ============================================================
# 2. KEEP THE SAME TRANSACTION TYPES
# ============================================================

# We keep the same filtering used throughout FinSentinel.

df = df[
    df["type"].isin(["TRANSFER", "CASH_OUT"])
].copy()


# ============================================================
# 3. REMOVE IDENTIFIER COLUMNS
# ============================================================

# Account IDs are not model features and are not useful
# for our drift demonstration.

df = df.drop(
    columns=[
        "nameOrig",
        "nameDest",
        "isFraud",
        "isFlaggedFraud",
    ]
)


# ============================================================
# 4. SORT BY TIME
# ============================================================

df = df.sort_values(
    "step"
).reset_index(drop=True)


# ============================================================
# 5. CREATE REFERENCE DATA
# ============================================================

# The reference dataset represents the environment
# the model learned from.
#
# We use the first 70% of the available time period.

split_step = df[
    "step"
].quantile(0.70)


reference_data = df[
    df["step"] <= split_step
].copy()


# ============================================================
# 6. CREATE CURRENT DATA
# ============================================================

# The final 30% represents new transactions arriving
# after the model was trained.

current_data = df[
    df["step"] > split_step
].copy()


print(
    "\n" + "=" * 60
)

print(
    "REFERENCE VS CURRENT DATA"
)

print(
    "=" * 60
)

print(
    "Reference rows:",
    len(reference_data)
)

print(
    "Current rows:",
    len(current_data)
)

print(
    "Reference time:",
    reference_data["step"].min(),
    "→",
    reference_data["step"].max()
)

print(
    "Current time:",
    current_data["step"].min(),
    "→",
    current_data["step"].max()
)


# ============================================================
# 7. CREATE DELIBERATELY DRIFTED DATA
# ============================================================

# This is our demo.
#
# Imagine that transaction behavior suddenly changes.
#
# Fraudsters might start moving much larger amounts.
#
# To simulate this, we multiply transaction amounts
# in the current data by 3.

drifted_data = current_data.copy()


drifted_data["amount"] = (
    drifted_data["amount"] * 3
)


print(
    "\n" + "=" * 60
)

print(
    "DRIFT SIMULATION"
)

print(
    "=" * 60
)

print(
    "Transaction amounts multiplied by 3x."
)


# ============================================================
# 8. DEFINE EVIDENTLY DATA SCHEMA
# ============================================================

# Explicitly tell Evidently which columns are numerical
# and which are categorical.

numerical_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


categorical_columns = [
    "type"
]


schema = DataDefinition(
    numerical_columns=numerical_columns,
    categorical_columns=categorical_columns,
)


# ============================================================
# 9. CONVERT PANDAS DATAFRAMES TO EVIDENTLY DATASETS
# ============================================================

reference_dataset = Dataset.from_pandas(
    reference_data,
    data_definition=schema,
)


current_dataset = Dataset.from_pandas(
    drifted_data,
    data_definition=schema,
)


# ============================================================
# 10. CREATE DRIFT REPORT
# ============================================================

# DataDriftPreset compares the distribution of each
# feature between the reference and current datasets.
#
# Evidently automatically selects appropriate drift
# detection methods based on the column type.

report = Report(
    [
        DataDriftPreset()
    ]
)


# ============================================================
# 11. RUN DRIFT ANALYSIS
# ============================================================

print(
    "\nRunning Evidently drift analysis..."
)


evaluation = report.run(
    current_dataset,
    reference_dataset,
)


# ============================================================
# 12. SAVE HTML REPORT
# ============================================================

output_file = (
    "reports/drift_report.html"
)


evaluation.save_html(
    output_file
)


print(
    "\nDrift report saved to:"
)

print(
    output_file
)


# ============================================================
# 13. GET REPORT RESULTS
# ============================================================

# Convert the Evidently evaluation to a dictionary
# so we can inspect the calculated metrics.

results = evaluation.dict()


# ============================================================
# 14. PRINT BASIC RESULT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "DRIFT ANALYSIS COMPLETE"
)

print(
    "=" * 60
)


print(
    "Reference dataset:",
    len(reference_data),
    "rows"
)

print(
    "Current dataset:",
    len(drifted_data),
    "rows"
)

print(
    "Amount change:",
    "3x"
)


# ============================================================
# 15. SAVE A SMALL SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "dataset": [
            "reference",
            "current_drifted",
        ],
        "rows": [
            len(reference_data),
            len(drifted_data),
        ],
        "mean_amount": [
            reference_data[
                "amount"
            ].mean(),
            drifted_data[
                "amount"
            ].mean(),
        ],
    }
)


summary.to_csv(
    "reports/drift_summary.csv",
    index=False,
)


print(
    "\nSummary saved to:"
)

print(
    "reports/drift_summary.csv"
)


# ============================================================
# 16. FINISHED
# ============================================================

print(
    "\nOpen this file in your browser:"
)

print(
    "reports/drift_report.html"
)

print(
    "\nFinSentinel drift monitoring complete."
)