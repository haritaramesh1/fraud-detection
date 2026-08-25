import pandas as pd


# ---------------------------------------------------------
# 1. Load the complete PaySim dataset
# ---------------------------------------------------------

file_path = "data/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)


# ---------------------------------------------------------
# 2. Basic information
# ---------------------------------------------------------

print("=" * 60)
print("DATASET SIZE")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ---------------------------------------------------------
# 3. Column names
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

print(df.columns.tolist())


# ---------------------------------------------------------
# 4. Check missing values
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ---------------------------------------------------------
# 5. Fraud count
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FRAUD COUNTS")
print("=" * 60)

print(df["isFraud"].value_counts())


# ---------------------------------------------------------
# 6. Fraud percentage
# ---------------------------------------------------------

fraud_percentage = df["isFraud"].mean() * 100

print("\nFraud percentage:")
print(f"{fraud_percentage:.4f}%")


# ---------------------------------------------------------
# 7. Transaction types
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRANSACTION TYPES")
print("=" * 60)

print(df["type"].value_counts())


# ---------------------------------------------------------
# 8. Fraud rate by transaction type
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FRAUD RATE BY TRANSACTION TYPE")
print("=" * 60)

fraud_by_type = (
    df.groupby("type")["isFraud"]
    .mean()
    .sort_values(ascending=False)
)

print(fraud_by_type * 100)


# ---------------------------------------------------------
# 9. Fraud count by transaction type
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FRAUD COUNT BY TRANSACTION TYPE")
print("=" * 60)

fraud_count_by_type = (
    df.groupby("type")["isFraud"]
    .sum()
    .sort_values(ascending=False)
)

print(fraud_count_by_type)


# ---------------------------------------------------------
# 10. Time range
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TIME RANGE")
print("=" * 60)

print("Minimum step:", df["step"].min())
print("Maximum step:", df["step"].max())