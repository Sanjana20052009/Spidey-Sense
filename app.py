"""
SPIDEY-SENSE
AI Fraud Detection Web Application

Backend:
    FastAPI
    XGBoost
    Pandas

This server:
    1. Serves the existing Spidey-Sense frontend.
    2. Loads the existing XGBoost model.
    3. Converts website inputs into the model's expected features.
    4. Performs fraud prediction.
    5. Calculates a user-facing risk score.
    6. Generates explainable risk signals.
    7. Returns the result to script.js.

Run:
    pip install fastapi uvicorn pandas numpy xgboost scikit-learn

Then:
    python app.py

Open:
    http://127.0.0.1:8000
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from xgboost import XGBClassifier
from aiagent import evaluate_transaction


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "xgboost_fraud_model.json"
DATASET_PATH = BASE_DIR / "credit_card_fraud_10k.csv"

INDEX_PATH = BASE_DIR / "index.html"
SCRIPT_PATH = BASE_DIR / "script.js"
STYLE_PATH = BASE_DIR / "style.css"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Spidey-Sense Fraud Detection API",
    description="AI-powered financial transaction fraud detection.",
    version="1.0.0",
)


# Allow frontend requests.
# Since FastAPI also serves the frontend, this is mainly useful
# if you later deploy frontend and backend separately.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

model: Optional[XGBClassifier] = None

try:
    if MODEL_PATH.exists():
        model = XGBClassifier()
        model.load_model(str(MODEL_PATH))
        print("==============================================")
        print(" SPIDEY-SENSE MODEL")
        print("==============================================")
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print("WARNING: xgboost_fraud_model.json was not found.")

except Exception as exc:
    print(f"WARNING: Could not load XGBoost model: {exc}")
    model = None


# ============================================================
# DATASET / FEATURE INFORMATION
# ============================================================

dataset_columns = []

if DATASET_PATH.exists():
    try:
        dataset_preview = pd.read_csv(DATASET_PATH, nrows=5)
        dataset_columns = list(dataset_preview.columns)
        print(f"Dataset loaded: {DATASET_PATH}")
        print(f"Dataset columns: {dataset_columns}")
    except Exception as exc:
        print(f"WARNING: Could not inspect dataset: {exc}")


# ============================================================
# REQUEST MODEL
# ============================================================

class TransactionRequest(BaseModel):
    """
    Fields displayed by the current Spidey-Sense website.
    """

    amount: float = Field(..., ge=0)

    location: str = Field(
        default="Chennai",
        min_length=1,
        max_length=100
    )

    device: str = Field(
        default="Known device",
        max_length=100
    )

    merchant: str = Field(
        default="Familiar merchant",
        max_length=150
    )

    velocity: int = Field(
        default=1,
        ge=0
    )

    time: str = Field(
        default="10:42",
        max_length=10
    )

    # Optional advanced fields.
    #
    # These are not currently displayed in index.html,
    # but accepting them makes the API compatible with the
    # underlying dataset/model.

    merchant_category: Optional[str] = None

    foreign_transaction: Optional[int] = Field(
        default=None,
        ge=0,
        le=1
    )

    location_mismatch: Optional[int] = Field(
        default=None,
        ge=0,
        le=1
    )

    device_trust_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100
    )

    cardholder_age: Optional[int] = Field(
        default=35,
        ge=18,
        le=120
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def parse_hour(time_string: str) -> int:
    """
    Convert HH:MM into the integer transaction_hour
    expected by the model.
    """

    try:
        parts = time_string.split(":")

        hour = int(parts[0])

        if hour < 0 or hour > 23:
            raise ValueError

        return hour

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid transaction time. Use HH:MM format."
        )


def normalize_text(value: str) -> str:
    """
    Normalize user input for comparisons.
    """

    if value is None:
        return ""

    return str(value).strip().lower()


def infer_merchant_category(
    merchant: str,
    explicit_category: Optional[str] = None
) -> str:
    """
    Convert the website's free-text merchant field into one of
    the categories represented in the training dataset.

    Dataset categories include:
        Electronics
        Travel
        Grocery
        Food
        Clothing

    If the user explicitly supplies merchant_category, use it.

    Otherwise infer a category from the merchant description.
    """

    if explicit_category:
        return explicit_category

    text = normalize_text(merchant)

    category_keywords = {
        "travel": [
            "travel",
            "flight",
            "airline",
            "hotel",
            "booking",
            "uber",
            "ola",
            "train",
            "bus"
        ],

        "food": [
            "food",
            "restaurant",
            "cafe",
            "coffee",
            "pizza",
            "swiggy",
            "zomato",
            "dining"
        ],

        "grocery": [
            "grocery",
            "supermarket",
            "market",
            "vegetable",
            "fruit",
            "dmart",
            "reliance"
        ],

        "clothing": [
            "clothing",
            "shirt",
            "dress",
            "fashion",
            "shoe",
            "shoes",
            "apparel"
        ],

        "electronics": [
            "electronics",
            "phone",
            "mobile",
            "laptop",
            "computer",
            "tablet",
            "camera",
            "amazon",
            "flipkart"
        ],
    }

    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return category.title()

    # The original frontend describes "Familiar merchant".
    # Electronics is used as a neutral dataset category when
    # no better category can be inferred.
    return "Electronics"


def infer_location_mismatch(
    location: str,
    explicit_value: Optional[int]
) -> int:
    """
    The original frontend considers Chennai, Madurai and
    Coimbatore normal locations.

    The dataset itself contains location_mismatch as a binary
    feature.

    If explicitly provided, use it.
    Otherwise infer it from the website location.
    """

    if explicit_value is not None:
        return int(explicit_value)

    normal_locations = {
        "chennai",
        "madurai",
        "coimbatore"
    }

    return 0 if normalize_text(location) in normal_locations else 1


def infer_foreign_transaction(
    location: str,
    explicit_value: Optional[int]
) -> int:
    """
    Infer foreign_transaction.

    Since the current UI only asks for a location and not country,
    this defaults to 0 unless explicitly supplied.

    This avoids incorrectly treating cities such as Mumbai
    as foreign transactions.
    """

    if explicit_value is not None:
        return int(explicit_value)

    return 0


def infer_device_trust_score(
    device: str,
    explicit_value: Optional[float]
) -> float:
    """
    Convert the current UI's Known/New device choice into the
    numerical device_trust_score used by the ML model.
    """

    if explicit_value is not None:
        return float(explicit_value)

    device_text = normalize_text(device)

    if "new" in device_text:
        return 25.0

    return 90.0


def build_model_dataframe(
    transaction: TransactionRequest
) -> pd.DataFrame:
    """
    Convert the website transaction into a dataframe matching
    the original training pipeline.

    Original training pipeline:
        - removes transaction_id
        - removes target is_fraud
        - one-hot encodes categorical variables
    """

    transaction_hour = parse_hour(transaction.time)

    merchant_category = infer_merchant_category(
        transaction.merchant,
        transaction.merchant_category
    )

    location_mismatch = infer_location_mismatch(
        transaction.location,
        transaction.location_mismatch
    )

    foreign_transaction = infer_foreign_transaction(
        transaction.location,
        transaction.foreign_transaction
    )

    device_trust_score = infer_device_trust_score(
        transaction.device,
        transaction.device_trust_score
    )

    row = {
        "amount": float(transaction.amount),

        "transaction_hour": int(transaction_hour),

        "merchant_category": merchant_category,

        "foreign_transaction": int(foreign_transaction),

        "location_mismatch": int(location_mismatch),

        "device_trust_score": float(device_trust_score),

        "velocity_last_24h": int(transaction.velocity),

        "cardholder_age": int(transaction.cardholder_age or 35),
    }

    df = pd.DataFrame([row])

    # Same preprocessing used in MLmodel.py.
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_cols) > 0:
        df = pd.get_dummies(
            df,
            columns=categorical_cols,
            drop_first=True
        )

    return df


def align_features_with_model(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Align the generated dataframe with the exact feature ordering
    stored in the trained XGBoost model.

    This is extremely important because the training code used
    pandas.get_dummies() and the model therefore expects the
    resulting feature layout.
    """

    if model is None:
        raise RuntimeError("ML model is not loaded.")

    try:
        expected_features = model.get_booster().feature_names
    except Exception:
        expected_features = None

    if not expected_features:
        return df

    aligned = df.reindex(
        columns=expected_features,
        fill_value=0
    )

    # Make sure all values are numeric.
    aligned = aligned.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    return aligned


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_rule_risk(
    transaction: TransactionRequest
) -> tuple[int, list[str]]:
    """
    Deterministic risk signals.

    These are intentionally separate from the ML probability.

    The final score combines:
        ML probability
        behavioral/rule signals
    """

    score = 0
    signals = []

    amount = float(transaction.amount)
    velocity = int(transaction.velocity)

    device = normalize_text(transaction.device)
    location = normalize_text(transaction.location)

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    if amount >= 50000:
        score += 25

        signals.append(
            "Unusually large transaction amount"
        )

    elif amount >= 10000:
        score += 12

        signals.append(
            "Amount above normal baseline"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if "new" in device:
        score += 20

        signals.append(
            "New device detected"
        )

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    if velocity >= 6:
        score += 20

        signals.append(
            "Very high transaction velocity"
        )

    elif velocity >= 5:
        score += 15

        signals.append(
            "High transaction velocity"
        )

    elif velocity >= 3:
        score += 5

        signals.append(
            "Elevated transaction velocity"
        )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    normal_locations = {
        "chennai",
        "madurai",
        "coimbatore"
    }

    if location not in normal_locations:
        score += 15

        signals.append(
            "Unusual location"
        )

    # --------------------------------------------------------
    # Merchant
    # --------------------------------------------------------

    merchant = normalize_text(transaction.merchant)

    if (
        "unfamiliar" in merchant
        or "unknown" in merchant
        or "suspicious" in merchant
    ):
        score += 10

        signals.append(
            "Unfamiliar merchant"
        )

    # --------------------------------------------------------
    # Foreign transaction
    # --------------------------------------------------------

    if transaction.foreign_transaction == 1:
        score += 15

        signals.append(
            "Foreign transaction detected"
        )

    # --------------------------------------------------------
    # Explicit location mismatch
    # --------------------------------------------------------

    if transaction.location_mismatch == 1:
        score += 15

        signals.append(
            "Location mismatch detected"
        )

    return min(score, 100), signals


def calculate_final_risk_score(
    ml_probability: float,
    rule_score: int
) -> int:
    """
    Combine ML probability and deterministic signals.

    70% ML
    30% rules

    The resulting value is bounded between 0 and 100.
    """

    score = (
        ml_probability * 0.70
        + rule_score * 0.30
    )

    return int(round(
        max(0, min(100, score))
    ))


def get_risk_level(score: int) -> str:
    """
    Map score to the same levels used by the original frontend.
    """

    if score <= 30:
        return "LOW RISK"

    if score <= 60:
        return "MEDIUM RISK"

    if score <= 80:
        return "HIGH RISK"

    return "CRITICAL RISK"


def get_recommendation(score: int) -> str:
    """
    Decision-support recommendation.
    """

    if score <= 30:
        return "✓ APPROVE — Transaction appears safe."

    if score <= 60:
        return "⚠ MONITOR — Flag transaction for review."

    if score <= 80:
        return (
            "🔐 ADDITIONAL VERIFICATION — "
            "Request OTP/device confirmation."
        )

    return (
        "🚨 HOLD AND REVIEW — "
        "Escalate to fraud analyst."
    )


def get_explanation(
    score: int,
    signals: list[str]
) -> str:
    """
    Generate the natural-language explanation shown in
    the Spidey-Sense Agent section.
    """

    if not signals:
        return (
            "The transaction closely matches the available "
            "behavioral signals and does not show significant "
            "anomalies."
        )

    signal_text = ", ".join(
        signal.lower()
        for signal in signals[:4]
    )

    if score <= 30:
        return (
            "The transaction appears consistent with normal "
            f"behavior. Observed signals: {signal_text}."
        )

    if score <= 60:
        return (
            "The transaction shows some unusual behavior "
            f"that should be monitored. Signals detected: "
            f"{signal_text}."
        )

    if score <= 80:
        return (
            "The transaction contains multiple behavioral "
            "anomalies requiring additional verification. "
            f"Signals detected: {signal_text}."
        )

    return (
        "The transaction strongly deviates from the expected "
        "behavior across multiple risk signals. "
        f"Signals detected: {signal_text}."
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/api/health")
def health():
    """
    Check whether the backend and ML model are available.
    """

    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
        "dataset_available": DATASET_PATH.exists(),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/model-info")
def model_info():
    """
    Return basic information about the model.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded."
        )

    try:
        features = model.get_booster().feature_names
    except Exception:
        features = []

    return {
        "model": "XGBoost",
        "model_file": MODEL_PATH.name,
        "feature_count": len(features),
        "features": features,
    }


# ============================================================
# MAIN PREDICTION ENDPOINT
# ============================================================

@app.post("/api/predict")
def predict_transaction(
    transaction: TransactionRequest
):
    """
    Main Spidey-Sense prediction endpoint.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "XGBoost model is unavailable. "
                "Make sure xgboost_fraud_model.json "
                "exists in the project root."
            )
        )

    try:
        # ----------------------------------------------------
        # Convert website input to model features
        # ----------------------------------------------------

        raw_features = build_model_dataframe(
            transaction
        )

        model_features = align_features_with_model(
            raw_features
        )

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        fraud_probability = float(
            model.predict_proba(
                model_features
            )[0][1]
        )

        ml_probability_percent = (
            fraud_probability * 100
        )

        ml_prediction = int(
            model.predict(
                model_features
            )[0]
        )

        # ----------------------------------------------------
        # Rule engine
        # ----------------------------------------------------

        rule_score, signals = calculate_rule_risk(
            transaction
        )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        risk_score = calculate_final_risk_score(
            ml_probability_percent,
            rule_score
        )

        risk_level = get_risk_level(
    risk_score
)
        # Recommendation based on risk score
        if risk_score <= 30:
            recommendation = "APPROVE — Transaction appears safe."

        elif risk_score <= 60:
            recommendation = "MONITOR — Flag transaction for review."

        elif risk_score <= 80:
            recommendation = "ADDITIONAL VERIFICATION — Request OTP/device confirmation."

        else:
            recommendation = "HOLD AND REVIEW — Escalate to fraud analyst."
        # Explanation for the prediction
        explanation = (
            "Transaction risk score: "
            + str(risk_score)
            + "/100. Risk level: "
            + str(risk_level)
            + ". The decision is based on the ML model and risk signals."
)
        agent_result = evaluate_transaction(
    transaction_data={
        "amount": transaction.amount,
        "merchant": transaction.merchant,
        "location": transaction.location,
        "device": transaction.device,
        "time": transaction.time,
        "velocity": transaction.velocity,
    },
    fraud_probability=ml_probability_percent,
    risk_score=risk_score,
    signals=signals
)  

# ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "success": True,

            "prediction": {
                "is_fraud": bool(ml_prediction),

                "fraud_probability": round(
                    ml_probability_percent,
                    2
                ),

                "risk_score": risk_score,

                "risk_level": risk_level,

                "recommendation": recommendation,
            },

            "signals": signals,

            "explanation": explanation,

            # AI Agent result
            "ai_agent": {
                "generated": agent_result["ai_generated"],
                "response": agent_result["spider_sense_alert"]
            },

            "features": {
                "amount": transaction.amount,

                "transaction_hour": parse_hour(
                    transaction.time
                ),

                "merchant_category":
                    infer_merchant_category(
                        transaction.merchant,
                        transaction.merchant_category
                    ),

                "foreign_transaction":
                    infer_foreign_transaction(
                        transaction.location,
                        transaction.foreign_transaction
                    ),

                "location_mismatch":
                    infer_location_mismatch(
                        transaction.location,
                        transaction.location_mismatch
                    ),

                "device_trust_score":
                    infer_device_trust_score(
                        transaction.device,
                        transaction.device_trust_score
                    ),

                "velocity_last_24h":
                    transaction.velocity,

                "cardholder_age":
                    transaction.cardholder_age,
            },

            "engine": {
                "ml_probability":
                    round(
                        ml_probability_percent,
                        2
                    ),

                "rule_score":
                    rule_score,

                "final_risk_score":
                    risk_score,
            }
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Prediction error:", exc)

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )

# ============================================================
# SERVE EXISTING WEBSITE
# ============================================================

@app.get("/")
def serve_index():
    """
    Serve the existing index.html.
    """

    if not INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(
        str(INDEX_PATH)
    )


@app.get("/index.html")
def serve_index_html():
    return serve_index()


@app.get("/script.js")
def serve_script():
    """
    Serve frontend JavaScript.
    """

    if not SCRIPT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="script.js not found."
        )

    return FileResponse(
        str(SCRIPT_PATH),
        media_type="application/javascript"
    )


@app.get("/style.css")
def serve_style():
    """
    Serve frontend CSS.
    """

    if not STYLE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="style.css not found."
        )

    return FileResponse(
        str(STYLE_PATH),
        media_type="text/css"
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print()
    print("==============================================")
    print("        SPIDEY-SENSE SERVER STARTING")
    print("==============================================")
    print()
    print("Website:")
    print("http://127.0.0.1:8000")
    print()
    print("API:")
    print("POST http://127.0.0.1:8000/api/predict")
    print()
    print("Health:")
    print("http://127.0.0.1:8000/api/health")
    print()
    print("==============================================")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
