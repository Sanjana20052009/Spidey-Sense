import os
import json
import pandas as pd
from xgboost import XGBClassifier
from groq import Groq

# 1. Load Trained Model
MODEL_FILE = "xgboost_fraud_model.json"
xgb_agent = XGBClassifier()

if os.path.exists(MODEL_FILE):
    xgb_agent.load_model(MODEL_FILE)
else:
    raise FileNotFoundError(f"'{MODEL_FILE}' not found. Please train the model first.")

# 2. Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

def evaluate_transaction(transaction_data: dict) -> dict:
    df_input = pd.DataFrame([transaction_data])
    
    # Calculate Fraud Probability
    risk_score = float(xgb_agent.predict_proba(df_input)[0][1])
    is_flagged = risk_score >= 0.50

    transaction_summary = {col: val for col, val in zip(df_input.columns, df_input.values[0])}
    alert_summary = ""

    if is_flagged:
        # Define prompt BEFORE calling the client
        prompt = f"""
        You are the 'Spider-Sense' Security AI Agent. 
        A transaction was just flagged as POTENTIAL FRAUD with a risk score of {risk_score:.2%}.

        Transaction Details:
        {json.dumps(transaction_summary, indent=2)}

        Provide a concise, 2-3 sentence 'Spider-Sense Alert' explaining why this transaction triggered the alert and what action the user or fraud analyst should take immediately.
        """
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",  # Active Groq model string
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            alert_summary = response.choices[0].message.content.strip()
        except Exception as e:
            alert_summary = f"[Spider-Sense Alert] Risk Score: {risk_score:.2%}. (Error: {str(e)})"
        alert_summary = f"Transaction verified safe. Risk score is low ({risk_score:.2%})."

    return {
        "risk_score": round(risk_score, 4),
        "flagged": is_flagged,
        "spider_sense_alert": alert_summary
    }

if __name__ == "__main__":
    sample_transaction = {
        "amount": 450.00,
        "transaction_hour": 3,
        "foreign_transaction": 1,
        "location_mismatch": 1,
        "device_trust_score": 0.2,
        "velocity_last_24h": 8,
        "cardholder_age": 29,
        "merchant_category_Electronics": 1,
        "merchant_category_Food": 0,
        "merchant_category_Grocery": 0,
        "merchant_category_Travel": 0
    }

    result = evaluate_transaction(sample_transaction)
    print("\n--- Agent Output ---")
    print(result["spider_sense_alert"])