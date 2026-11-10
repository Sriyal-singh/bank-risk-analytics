"""Synthetic (but realistic) bank data.

Why synthetic? It makes the repo fully reproducible with one command and avoids licence
issues. The generators embed realistic relationships (utilisation, late payments, income,
hidden "financial stress" noise, class imbalance, missing values, a gender-income proxy for
the fairness audit). To use real data, pass your own CSVs to run_all.py (see README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate(z, rate: float) -> float:
    """Find the intercept shift so that mean(sigmoid(z + shift)) == rate."""
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _draw(rng, p):
    return (rng.random(len(p)) < p).astype(int)


def make_customers(n: int = 25_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    gender = rng.choice(["M", "F"], n, p=[0.55, 0.45])
    city_tier = rng.choice(["Tier1", "Tier2", "Tier3"], n, p=[0.45, 0.35, 0.20])
    employment = rng.choice(
        ["Salaried", "Self-employed", "Business", "Other"], n, p=[0.55, 0.20, 0.15, 0.10]
    )
    stress = rng.normal(0, 1, n)  # hidden financial stress (never given to the model)
    disengage = rng.normal(0, 1, n)  # hidden disengagement from the bank

    income = (
        rng.lognormal(12.85, 0.55, n)
        * (1 + 0.012 * (age - 21))
        * np.where(gender == "F", 0.90, 1.0)  # proxy: lets the fairness audit find something
    )
    income = np.clip(income, 150_000, 6_000_000).round(-3)
    tenure_months = np.minimum(rng.integers(3, 200, n), (age - 18) * 12)
    credit_history_years = np.clip(np.minimum(age - 18, rng.gamma(2.0, 3.5, n)), 0.5, None).round(1)
    credit_limit = (income * rng.uniform(0.12, 0.5, n)).round(-3)
    utilization = np.clip(rng.beta(2, 4, n) + 0.07 * stress, 0.0, 1.0).round(3)
    balance = (utilization * credit_limit).round(0)
    late_12m = rng.poisson(np.exp(-1.1 + 0.55 * stress + 1.2 * utilization))
    payment_ratio = np.clip(
        0.8 - 0.12 * stress - 0.35 * utilization + rng.normal(0, 0.15, n), 0.02, 1.0
    ).round(3)
    debt_to_income = np.clip(rng.beta(2, 5, n) * 0.9 + 0.04 * stress, 0.0, 0.9).round(3)
    num_products = 1 + rng.poisson(1.1, n)
    monthly_txns = rng.poisson(22 * np.exp(-0.25 * disengage))
    app_logins_30d = rng.poisson(9 * np.exp(-0.35 * disengage))
    complaints_12m = rng.poisson(np.exp(-1.6 + 0.5 * disengage))
    inactive_months = np.minimum(rng.poisson(np.exp(-0.6 + 0.6 * disengage)), 6)
    salary_credited = _draw(rng, _sigmoid(1.0 - 0.9 * disengage))
    avg_account_balance = (rng.lognormal(10.8, 1.0, n) * np.exp(-0.15 * disengage)).round(0)
    loan_amount = (income * rng.uniform(0.3, 2.0, n)).round(-3)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 84], n, p=[0.1, 0.2, 0.3, 0.15, 0.2, 0.05])
    loan_to_income = (loan_amount / income).round(3)

    # ---- targets -------------------------------------------------------------------------
    z = (
        0.55 * late_12m + 2.0 * utilization - 2.2 * payment_ratio + 1.8 * debt_to_income
        - 0.25 * np.log(income / 4e5) + 0.5 * stress + rng.normal(0, 0.4, n)
    )
    delinquent = _draw(rng, _sigmoid(z + _calibrate(z, 0.12)))

    zl = (
        2.4 * debt_to_income + 0.35 * late_12m - 0.30 * np.log(income / 4e5)
        - 0.05 * credit_history_years + 0.9 * loan_to_income + 0.02 * loan_term_months / 12
        + 0.5 * (employment == "Other") + 0.25 * (employment == "Self-employed")
        + 0.1 * (city_tier == "Tier3") + 0.5 * stress + rng.normal(0, 0.5, n)
    )
    loan_default = _draw(rng, _sigmoid(zl + _calibrate(zl, 0.14)))

    zc = (
        0.5 * complaints_12m + 0.45 * inactive_months - 0.10 * app_logins_30d
        - 0.8 * salary_credited - 0.30 * (num_products - 1) + 0.5 * disengage
        + rng.normal(0, 0.5, n)
    )
    churned = _draw(rng, _sigmoid(zc + _calibrate(zc, 0.18)))

    annual_revenue = np.clip(
        1_200 + 0.02 * avg_account_balance + 1_100 * num_products + 0.18 * balance * payment_ratio,
        1_000, None,
    ).round(0)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "age": age, "gender": gender, "city_tier": city_tier, "employment": employment,
            "income": income, "tenure_months": tenure_months,
            "credit_history_years": credit_history_years, "credit_limit": credit_limit,
            "utilization": utilization, "balance": balance, "late_12m": late_12m,
            "payment_ratio": payment_ratio, "debt_to_income": debt_to_income,
            "num_products": num_products, "monthly_txns": monthly_txns,
            "app_logins_30d": app_logins_30d, "complaints_12m": complaints_12m,
            "inactive_months": inactive_months, "salary_credited": salary_credited,
            "avg_account_balance": avg_account_balance, "loan_amount": loan_amount,
            "loan_term_months": loan_term_months, "loan_to_income": loan_to_income,
            "annual_revenue": annual_revenue,
            "delinquent_90d": delinquent, "loan_default": loan_default, "churned": churned,
        }
    )
    # realistic missing data
    df.loc[rng.random(n) < 0.04, "credit_history_years"] = np.nan
    df.loc[rng.random(n) < 0.03, "debt_to_income"] = np.nan
    return df


def make_transactions(n: int = 120_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="4min")
    hour = ts.hour.to_numpy()
    cats = ["grocery", "fuel", "restaurant", "electronics", "travel", "online_retail", "atm", "jewellery"]
    merchant = rng.choice(cats, n, p=[0.22, 0.14, 0.16, 0.06, 0.07, 0.25, 0.07, 0.03])
    high_risk_cat = np.isin(merchant, ["electronics", "jewellery", "travel"])
    amount = rng.lognormal(6.9, 1.0, n) * np.where(high_risk_cat, 3.0, 1.0)
    is_foreign = rng.binomial(1, 0.04, n)
    card_present = np.where(merchant == "online_retail", 0, rng.binomial(1, 0.8, n))
    txn_count_1h = 1 + rng.poisson(0.25, n)
    distance = rng.exponential(8, n) + is_foreign * rng.uniform(500, 5000, n)
    new_device = (card_present == 0) * rng.binomial(1, 0.12, n)
    amount_ratio = amount / rng.lognormal(6.9, 0.35, n)
    night = (hour <= 5).astype(int)

    z = (
        2.4 * is_foreign + 2.2 * new_device + 1.3 * (txn_count_1h - 1) + 1.0 * (1 - card_present)
        + 1.4 * night + 1.0 * np.clip(np.log(amount_ratio), 0, None) + 0.8 * high_risk_cat
        + 1.8 * (high_risk_cat & (card_present == 0))      # interaction: risky merchant, card not present
        + 1.2 * (night * (txn_count_1h > 1))                # interaction: night-time bursts
        + rng.normal(0, 0.5, n)
    )
    is_fraud = _draw(rng, _sigmoid(z + _calibrate(z, 0.006)))

    return pd.DataFrame(
        {
            "txn_id": [f"T{i:07d}" for i in range(n)],
            "timestamp": ts, "amount": amount.round(2), "hour": hour,
            "merchant_category": merchant, "is_foreign": is_foreign,
            "card_present": card_present, "txn_count_1h": txn_count_1h,
            "distance_from_home_km": distance.round(1), "new_device": new_device,
            "amount_ratio": amount_ratio.round(3), "is_fraud": is_fraud,
        }
    )
