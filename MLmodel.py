import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    auc, 
    f1_score
)
from xgboost import XGBClassifier

# ==========================================
# 1. FILE PATH CONFIGURATION
# ==========================================
# CHANGE THIS PATH: Replace 'credit_card_fraud_10k.csv' with your local CSV path if it's in a different directory.
DATASET_PATH = "credit_card_fraud_10k.csv"

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at '{file_path}'. Please check the path.")
    
    df = pd.read_csv(file_path)
    print(f"Data successfully loaded. Shape: {df.shape}")
    return df

# ==========================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ==========================================
def preprocess_data(df):
    # CHANGE THIS NAME: Replace 'is_fraud' or 'Class' with the actual target column name in your CSV.
    TARGET_COLUMN = "is_fraud" 
    
    if TARGET_COLUMN not in df.columns:
        # Fallback search for common fraud target column names
        possible_targets = [col for col in df.columns if col.lower() in ['class', 'is_fraud', 'fraud', 'target']]
        if possible_targets:
            TARGET_COLUMN = possible_targets[0]
            print(f"Target column auto-detected as: '{TARGET_COLUMN}'")
        else:
            raise KeyError(f"Target column '{TARGET_COLUMN}' not found. Check your CSV column headers.")

    # Drop non-predictive identifier columns if present
    # CHANGE THIS LIST: Add or remove column names that are unique IDs (e.g., 'transaction_id', 'user_id', 'name')
    ignore_cols = [col for col in ['transaction_id', 'user_id', 'customer_id', 'timestamp'] if col in df.columns]
    
    X = df.drop(columns=[TARGET_COLUMN] + ignore_cols)
    y = df[TARGET_COLUMN]

    # Handle categorical variables (One-Hot Encoding)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"Encoding categorical features: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y

# ==========================================
# 3. MODEL TRAINING & EVALUATION
# ==========================================
def main():
    # Load dataset
    df = load_data(DATASET_PATH)
    X, y = preprocess_data(df)

    # Train-Test Split (80% Train, 20% Test) with Stratification for class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight to handle imbalanced fraud data
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_weight = num_neg / max(num_pos, 1)

    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")

    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)

    print("\n" + "="*40)
    print("           EVALUATION METRICS          ")
    print("="*40)
    print(f"PR-AUC Score: {pr_auc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Feature Importance Extraction for AI Agent Explanations
    importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\nTop 5 Predictive Signals (Spider-Sense Evidence):")
    print(importance_df.head(5).to_string(index=False))

    # Save trained model for FastAPI / Agent inference
    model.save_model("xgboost_fraud_model.json")
    print("\nModel saved to 'xgboost_fraud_model.json'.")

if __name__ == "__main__":
    main()