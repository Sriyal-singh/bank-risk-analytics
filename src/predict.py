"""Thin inference layer used by the Streamlit app (and easy to unit-test)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import churn as churn_mod
from . import credit_scoring, delinquency, fraud
from .utils import MODELS, local_reasons

_CACHE: dict = {}


def _load(name: str):
    if name not in _CACHE:
        path = MODELS / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python run_all.py` first.")
        _CACHE[name] = joblib.load(path)
    return _CACHE[name]


CUSTOMER_DEFAULTS = {
    "age": 35, "gender": "M", "city_tier": "Tier1", "employment": "Salaried", "income": 600_000.0,
    "tenure_months": 48, "credit_history_years": 8.0, "credit_limit": 200_000.0, "utilization": 0.4,
    "balance": 80_000.0, "late_12m": 0, "payment_ratio": 0.6, "debt_to_income": 0.25, "num_products": 2,
    "monthly_txns": 22, "app_logins_30d": 9, "complaints_12m": 0, "inactive_months": 0,
    "salary_credited": 1, "avg_account_balance": 60_000.0, "loan_amount": 500_000.0,
    "loan_term_months": 36, "loan_to_income": 0.8, "annual_revenue": 10_000.0,
}


def _customer_frame(values: dict) -> pd.DataFrame:
    row = {**CUSTOMER_DEFAULTS, **values}
    row["balance"] = values.get("balance", row["utilization"] * row["credit_limit"])
    row["loan_to_income"] = values.get("loan_to_income", row["loan_amount"] / row["income"])
    return pd.DataFrame([row])


def delinquency_risk(values: dict) -> dict:
    art = _load("delinquency")
    df = delinquency.engineer(_customer_frame(values))
    X = df[art["num"] + art["cat"]]
    p = float(art["pipe"].predict_proba(X)[:, 1][0])
    reasons = local_reasons(art["pipe"], X, art["ref"], art["num"] + art["cat"])[0]
    expected_loss_avoided = p * float(df["balance"].iloc[0]) * delinquency.SAVE_RATE
    return {"probability": p, "reasons": reasons,
            "contact": expected_loss_avoided > delinquency.CONTACT_COST}


def credit_decision(values: dict) -> dict:
    art = _load("scorecard")
    sc = art["scorecard"]
    df = _customer_frame(values)
    X = df[credit_scoring.NUM + credit_scoring.CAT].astype({c: float for c in credit_scoring.NUM})
    score = float(sc.score(X)[0])
    return {"score": score, "prob_default": float(sc.prob_bad(X)[0]),
            "cutoff": art["cutoff"], "approved": score >= art["cutoff"]}


def fraud_risk(values: dict) -> dict:
    art = _load("fraud")
    row = {"amount": 2500.0, "hour": 14, "merchant_category": "grocery", "is_foreign": 0,
           "card_present": 1, "txn_count_1h": 1, "distance_from_home_km": 5.0, "new_device": 0,
           "amount_ratio": 1.0, **values}
    df = fraud.engineer(pd.DataFrame([row]))
    p = float(art["pipe"].predict_proba(df[art["num"] + art["cat"]])[:, 1][0])
    return {"score": p, "alert": p >= art["threshold"], "threshold": art["threshold"]}


def churn_risk(values: dict) -> dict:
    art = _load("churn")
    df = _customer_frame(values)
    X = df[art["num"] + art["cat"]].astype({c: float for c in art["num"]})
    p = float(art["pipe"].predict_proba(X)[:, 1][0])
    rev = float(df["annual_revenue"].iloc[0])
    ev = p * churn_mod.EFFECTIVENESS * rev * churn_mod.VALUE_YEARS - churn_mod.OFFER_COST
    return {"probability": p, "expected_net_value": ev, "make_offer": ev > 0,
            "action": churn_mod.next_best_action(df.iloc[0]) if ev > 0 else "No offer - not cost-effective"}
