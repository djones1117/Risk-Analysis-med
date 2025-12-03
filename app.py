from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from risk_explainer import explain_risk


MODEL_PATH = Path("models") / "risk_model.joblib"

# Load the bundle (pipeline + metadata) at startup
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_cols = bundle["feature_cols"]

app = FastAPI(
    title="Cardiovascular Risk Prediction API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


origins = [
    "http://127.0.0.1:5501",
    "http://localhost:5501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    # Match FEATURE_COLS, types based on the dataset
    male: int = Field(..., ge=0, le=1)
    age: int = Field(..., ge=18, le=100)
    education: Optional[float] = Field(None, ge=1, le=4)
    currentSmoker: int = Field(..., ge=0, le=1)
    cigsPerDay: Optional[float] = Field(None, ge=0)
    BPMeds: Optional[float] = Field(None, ge=0, le=1)
    prevalentStroke: int = Field(..., ge=0, le=1)
    prevalentHyp: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    totChol: Optional[float] = Field(None, ge=50)
    sysBP: float = Field(..., ge=50)
    diaBP: float = Field(..., ge=30)
    BMI: Optional[float] = Field(None, ge=10)
    heartRate: Optional[float] = Field(None, ge=20)
    glucose: Optional[float] = Field(None, ge=40)

class RiskOutput(BaseModel):
    risk_score: float
    risk_level: str
    top_factors: List[str]

@app.get("/")
def root():
    return {"status": "ok", "message": "Cardiovascular Risk Prediction API"}

@app.post("/predict", response_model=RiskOutput)
def predict_risk(patient: PatientInput):
    # Convert request to dict
    patient_dict = patient.model_dump()

     # Build feature vector as DataFrame with proper column names
    X = pd.DataFrame([patient_dict])[feature_cols]

    # Probability of positive class (TenYearCHD = 1)
    proba = float(pipeline.predict_proba(X)[0][1])

    # Map probability to risk band
    if proba < 0.10:
        level = "Low"
    elif proba < 0.20:
        level = "Moderate"
        # small gap
    else:
        level = "High"

    top_factors = explain_risk(patient_dict)

    return RiskOutput(
        risk_score=round(proba, 3),
        risk_level=level,
        top_factors=top_factors,
    )
