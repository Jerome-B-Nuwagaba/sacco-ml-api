from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="SACCO ML Service")

# ── Load models once at startup ───────────────────────────────────────────────
BASE = os.path.dirname(__file__)

try:
    gb_cs    = joblib.load(os.path.join(BASE, "models/model_credit_score_v3.pkl"))
    rf_clf   = joblib.load(os.path.join(BASE, "models/model_risk_classifier_v3.pkl"))
    gb_loan  = joblib.load(os.path.join(BASE, "models/model_loan_amount_v3.pkl"))
    encoders = joblib.load(os.path.join(BASE, "models/label_encoders_v3.pkl"))
    feat_med = pd.read_csv(os.path.join(BASE, "models/feature_medians.csv"), index_col=0).squeeze()
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

SACCO_MAX_LOAN = 15_000_000
SACCO_MIN_LOAN = 100_000
SACCO_MAX_DTI  = 0.40

FEATURE_COLS = [
    'rainfall_2022_mm','rainfall_2023_mm','avg_rainfall_mm',
    'profitability_2022','profitability_2023','profitability_trend',
    'soil_acidity','temperature_c','humidity','wind_speed',
    'primary_crop_enc','weather_desc_enc',
    'age','emp_length_years','annual_income_ugx','loan_amount_ugx',
    'interest_rate_pct','loan_pct_income','credit_hist_years',
    'prior_default_num','high_interest_flag',
    'loan_grade_enc','loan_purpose_enc','home_ownership_enc',
    'total_rainfall_mm','avg_temp_c','drought_flag',
    'flood_risk_flag','weather_risk_score',
    'membership_years','land_size_acres','num_guarantors',
    'annual_savings_ugx','collateral_ugx',
    'debt_to_income','income_per_acre','savings_to_income',
    'collateral_coverage','experienced_member','multi_guarantors',
    'rainfall_trend','region_enc','district_enc',
]

# ── Request schema ─────────────────────────────────────────────────────────────
class LoanRequest(BaseModel):
    # Member profile
    age: int                        = Field(..., ge=18, le=80)
    membership_years: int           = Field(..., ge=0, le=50)
    annual_income_ugx: int          = Field(..., ge=0)
    annual_savings_ugx: int         = Field(..., ge=0)
    land_size_acres: float          = Field(..., ge=0)
    collateral_ugx: int             = Field(..., ge=0)
    num_guarantors: int             = Field(..., ge=0, le=10)
    emp_length_years: float         = Field(..., ge=0)
    primary_crop: str               = "Maize"
    home_ownership: str             = "OWN"

    # Loan request details
    loan_amount_ugx: int            = Field(..., ge=0)
    loan_grade: str                 = "C"
    loan_purpose: str               = "PERSONAL"
    interest_rate_pct: float        = Field(..., ge=0)
    loan_pct_income: float          = Field(..., ge=0)
    credit_hist_years: int          = Field(..., ge=0)
    prior_default_num: int          = Field(0, ge=0, le=1)

    # Farm data
    profitability_2022: int         = Field(..., ge=1, le=5)
    profitability_2023: int         = Field(..., ge=1, le=5)
    soil_acidity: int               = Field(..., ge=1, le=5)
    rainfall_2022_mm: float         = Field(..., ge=0)
    rainfall_2023_mm: float         = Field(..., ge=0)
    temperature_c: float            = Field(..., ge=0)
    humidity: float                 = Field(..., ge=0, le=100)
    wind_speed: float               = Field(..., ge=0)
    weather_desc: str               = "Clear"

    # District / weather
    district: str                   = "Kampala"
    region: str                     = "Central"
    total_rainfall_mm: float        = Field(..., ge=0)
    avg_temp_c: float               = Field(..., ge=0)
    drought_flag: int               = Field(0, ge=0, le=1)
    flood_risk_flag: int            = Field(0, ge=0, le=1)
    weather_risk_score: float       = Field(0.0, ge=0, le=10)

def safe_encode(encoder, value):
    try:
        return int(encoder.transform([str(value)])[0])
    except ValueError:
        return 0

# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.post("/predict")
def predict(req: LoanRequest):
    row = {col: float(feat_med.get(col, 0)) for col in FEATURE_COLS}

    # Numeric fields
    numeric = [
        'age','emp_length_years','annual_income_ugx','loan_amount_ugx',
        'annual_savings_ugx','collateral_ugx','interest_rate_pct',
        'loan_pct_income','credit_hist_years','prior_default_num',
        'membership_years','land_size_acres','num_guarantors',
        'rainfall_2022_mm','rainfall_2023_mm','profitability_2022',
        'profitability_2023','soil_acidity','temperature_c','humidity',
        'wind_speed','total_rainfall_mm','avg_temp_c',
        'drought_flag','flood_risk_flag','weather_risk_score',
    ]
    for f in numeric:
        if hasattr(req, f):
            row[f] = getattr(req, f)

    # Categorical fields
    for cat in ['primary_crop','home_ownership','loan_purpose','loan_grade',
                'region','district','weather_desc']:
        if cat in encoders:
            row[cat + '_enc'] = safe_encode(encoders[cat], getattr(req, cat))

    # Derived features
    row['avg_rainfall_mm']     = (row['rainfall_2022_mm'] + row['rainfall_2023_mm']) / 2
    row['profitability_trend'] = row['profitability_2023'] - row['profitability_2022']
    row['debt_to_income']      = row['loan_amount_ugx'] / max(row['annual_income_ugx'], 1)
    row['income_per_acre']     = row['annual_income_ugx'] / max(row['land_size_acres'], 0.1)
    row['savings_to_income']   = row['annual_savings_ugx'] / max(row['annual_income_ugx'], 1)
    row['collateral_coverage'] = row['collateral_ugx'] / max(row['loan_amount_ugx'], 1)
    row['high_interest_flag']  = int(row['interest_rate_pct'] > 15)
    row['experienced_member']  = int(row['membership_years'] >= 5)
    row['multi_guarantors']    = int(row['num_guarantors'] >= 3)
    row['rainfall_trend']      = row['rainfall_2023_mm'] - row['rainfall_2022_mm']

    X = pd.DataFrame([row])[FEATURE_COLS].fillna(0)

    # Run models
    credit_score      = round(float(gb_cs.predict(X)[0]), 1)
    risk_label        = rf_clf.predict(X)[0]
    risk_proba_arr    = rf_clf.predict_proba(X)[0]
    risk_proba        = dict(zip(rf_clf.classes_, [round(float(p), 3) for p in risk_proba_arr]))
    loan_raw          = float(gb_loan.predict(X)[0])
    loan_rec          = int(round(np.clip(loan_raw, SACCO_MIN_LOAN, SACCO_MAX_LOAN), -4))

    dti       = row['debt_to_income']
    top_prob  = float(max(risk_proba_arr))

    policy_flags = []
    if not dti <= SACCO_MAX_DTI:
        policy_flags.append(f"DTI {dti:.1%} exceeds 40% limit — officer review required")
    if row['prior_default_num'] == 1:
        policy_flags.append("Prior default on record — credit committee approval required")
    if row['drought_flag'] == 1:
        policy_flags.append("Drought season flagged — consider reduced disbursement")

    return {
        "credit_score":          credit_score,
        "risk_level":            risk_label,
        "risk_probabilities":    risk_proba,
        "uncertainty_warning":   f"Low confidence ({top_prob:.0%}) — manual review recommended" if top_prob < 0.60 else None,
        "recommended_loan_ugx":  loan_rec,
        "loan_within_dti_limit": dti <= SACCO_MAX_DTI,
        "policy_flags":          policy_flags,
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}