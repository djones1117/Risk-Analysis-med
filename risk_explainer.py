# risk_explainer.py

from typing import Dict, Any, List

def explain_risk(features: Dict[str, Any]) -> List[str]:
    """
    Build a list of human-readable reasons based on raw feature values.

    Expected keys (matching FEATURE_COLS):
      male, age, education, currentSmoker, cigsPerDay, BPMeds,
      prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
      diaBP, BMI, heartRate, glucose
    """
    reasons = []

    age = features.get("age")
    if age is not None:
        if age >= 60:
            reasons.append("Age over 60")
        elif age >= 45:
            reasons.append("Age over 45")

    sys_bp = features.get("sysBP")
    if sys_bp is not None:
        if sys_bp >= 140:
            reasons.append("High systolic blood pressure (≥ 140)")
        elif sys_bp >= 130:
            reasons.append("Elevated systolic blood pressure (≥ 130)")

    dia_bp = features.get("diaBP")
    if dia_bp is not None and dia_bp >= 90:
        reasons.append("High diastolic blood pressure (≥ 90)")

    bmi = features.get("BMI")
    if bmi is not None:
        if bmi >= 30:
            reasons.append("BMI in obese range (≥ 30)")
        elif bmi >= 25:
            reasons.append("BMI in overweight range (≥ 25)")

    smoker = features.get("currentSmoker")
    cigs_per_day = features.get("cigsPerDay")
    if smoker == 1 or smoker is True:
        reasons.append("Current smoker")
        if cigs_per_day is not None and cigs_per_day >= 10:
            reasons.append("Smokes ≥ 10 cigarettes per day")

    diabetes = features.get("diabetes")
    if diabetes == 1 or diabetes is True:
        reasons.append("Diabetes")

    tot_chol = features.get("totChol")
    if tot_chol is not None and tot_chol >= 240:
        reasons.append("High total cholesterol (≥ 240)")

    glucose = features.get("glucose")
    if glucose is not None and glucose >= 126:
        reasons.append("Elevated fasting glucose (≥ 126)")

    # Only show top few
    return reasons[:3]