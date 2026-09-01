def calculate_risk(fraud_probability):

    risk_score = fraud_probability * 100

    if risk_score <= 30:
        level = "LOW"
        action = "APPROVE AUTOMATICALLY"

    elif risk_score <= 60:
        level = "MEDIUM"
        action = "MONITOR"

    elif risk_score <= 80:
        level = "HIGH"
        action = "ADDITIONAL VERIFICATION"

    else:
        level = "CRITICAL"
        action = "HOLD AND REVIEW"

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": level,
        "recommended_action": action
    }