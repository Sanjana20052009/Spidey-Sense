import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
DATA_PATH = "data/raw/credit_card_fraud_10k.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)
print("Duplicate transaction IDs:",
      df["transaction_id"].duplicated().sum())
print("\nFraud distribution:")
print(df["is_fraud"].value_counts())

print("\nFraud percentage:")
print(df["is_fraud"].value_counts(normalize=True) * 100)
print("\nUnique values:")

for column in df.columns:
    print("\n", column)
    print(df[column].unique()[:20])
transaction_ids = df["transaction_id"].copy()

df = df.drop(columns=["transaction_id"])
numeric_columns = [
    "amount",
    "transaction_hour",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())
df["merchant_category"] = df["merchant_category"].fillna("Unknown")
df["log_amount"] = np.log1p(df["amount"])
df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
df["is_late_night"] = (
    (df["transaction_hour"] >= 0) &
    (df["transaction_hour"] <= 5)
).astype(int)
df["high_velocity"] = (
    df["velocity_last_24h"] >= df["velocity_last_24h"].quantile(0.90)
).astype(int)
df["low_device_trust"] = (
    df["device_trust_score"] < 40
).astype(int)
df["risk_signal_count"] = (
    df["foreign_transaction"]
    + df["location_mismatch"]
    + df["high_velocity"]
    + df["low_device_trust"]
    + df["is_late_night"]
)
y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
X_train = pd.get_dummies(
    X_train,
    columns=["merchant_category"],
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=["merchant_category"],
    dtype=int
)
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)
X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)
with open("data/processed/features.txt", "w") as f:
    for column in X_train.columns:
        f.write(column + "\n")
dashboard_df = pd.read_csv(DATA_PATH)

dashboard_df.to_csv(
    "data/processed/dashboard_transactions.csv",
    index=False
)
            