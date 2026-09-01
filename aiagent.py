import os
import json
from groq import Groq


# ============================================================
# GROQ CLIENT
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )

client = Groq(api_key=api_key)


# ============================================================
# AI DECISION INTELLIGENCE
# ============================================================

def evaluate_transaction(
    transaction_data: dict,
    fraud_probability: float,
    risk_score: int,
    signals: list[str]
) -> dict:
    """
    Generate an AI-powered explanation and recommendation.

    XGBoost is responsible for the fraud prediction.
    This function uses the prediction + transaction evidence
    to generate human-readable decision intelligence.
    """

    transaction_summary = {
        key: value
        for key, value in transaction_data.items()
        if value is not None
    }

    prompt = f"""
You are Spidey-Sense, a financial fraud decision-intelligence AI.

Analyze the transaction using ONLY the evidence provided below.

TRANSACTION:
{json.dumps(transaction_summary, indent=2)}

MACHINE LEARNING FRAUD PROBABILITY:
{fraud_probability:.2f}%

FINAL RISK SCORE:
{risk_score}/100

DETECTED RISK SIGNALS:
{json.dumps(signals, indent=2)}

Your task is to provide a concise decision-support response.

Return EXACTLY this structure:

ASSESSMENT:
<2-3 sentences explaining the risk using the supplied evidence>

ACTION:
<one clear recommended action>

Do not invent transaction information.
Do not change the ML probability or risk score.
Do not claim certainty that a transaction is fraudulent.
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.2
        )

        ai_response = (
            response.choices[0]
            .message
            .content
            .strip()
        )

    except Exception as exc:

        print("AI Agent error:", exc)

        ai_response = (
            "ASSESSMENT:\n"
            "The transaction requires review based on "
            "the available machine-learning and behavioral signals.\n\n"
            "ACTION:\n"
            "Review the transaction and apply additional "
            "authentication if required."
        )

    return {
        "spider_sense_alert": ai_response,
        "ai_generated": True
    }