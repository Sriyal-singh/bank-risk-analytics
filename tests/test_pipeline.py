"""Smoke + sanity tests.  Run:  python -m pytest -q   (after `python run_all.py --quick`)"""
import numpy as np
import pandas as pd
import pytest

from src import churn, credit_scoring, data, delinquency, fraud, predict


@pytest.fixture(scope="module")
def customers():
    return data.make_customers(6000, seed=1)


@pytest.fixture(scope="module")
def tx():
    return data.make_transactions(30000, seed=2)


def test_data_shapes_and_rates(customers, tx):
    assert customers["customer_id"].is_unique
    assert 0.08 < customers["delinquent_90d"].mean() < 0.16
    assert 0.003 < tx["is_fraud"].mean() < 0.01
    assert customers["credit_history_years"].isna().any()  # missing values are present on purpose


def test_delinquency_beats_random(customers):
    r = delinquency.run(customers, save=True)
    assert r["champion_calibrated"]["roc_auc"] > 0.75
    assert r["top_decile_lift"] > 2.5
    cal = r["champion_calibrated"]
    assert abs(cal["mean_predicted_rate"] - cal["actual_rate"]) < 0.02  # calibrated


def test_scorecard_points_and_fairness(customers):
    r = credit_scoring.run(customers, save=True)
    assert r["models"]["WOE scorecard"]["roc_auc"] > 0.70
    # scorecard should be close to the black-box model (interpretability at small cost)
    assert r["models"]["Gradient boosting"]["roc_auc"] - r["models"]["WOE scorecard"]["roc_auc"] < 0.03
    assert 0.65 < r["approval_policy"]["scorecard"]["approval_rate"] < 0.75
    # approved book must be safer than the whole book
    assert r["approval_policy"]["scorecard"]["bad_rate_of_approved"] < r["approval_policy"]["portfolio_bad_rate_if_approve_all"]


def test_scorecard_never_uses_protected_attributes():
    assert not set(credit_scoring.PROTECTED) & set(credit_scoring.NUM + credit_scoring.CAT)
    assert not {"gender", "age"} & set(delinquency.NUM + delinquency.CAT)


def test_fraud_cost_saving(tx):
    r = fraud.run(tx, save=True)
    best_pr = max(v["pr_auc"] for k, v in r["models_test"].items() if "pr_auc" in v)
    assert best_pr > 5 * r["fraud_rate_test"]  # far better than no-skill
    assert r["business"]["saving_pct"] > 0


def test_churn_targeting_beats_random(customers):
    r = churn.run(customers, save=True)
    s = r["strategy"]
    parse = lambda x: float(x.replace("Rs", "").replace(",", "").strip())
    assert parse(s["net_value_model_targeting"]) > parse(s["net_value_random_same_size"])


def test_prediction_layer_directionality():
    low = predict.delinquency_risk({"utilization": 0.1, "late_12m": 0, "payment_ratio": 0.95, "debt_to_income": 0.1})
    high = predict.delinquency_risk({"utilization": 0.95, "late_12m": 6, "payment_ratio": 0.05, "debt_to_income": 0.7})
    assert high["probability"] > low["probability"]

    good = predict.credit_decision({"debt_to_income": 0.1, "late_12m": 0, "income": 1_200_000})
    bad = predict.credit_decision({"debt_to_income": 0.8, "late_12m": 6, "income": 200_000})
    assert good["score"] > bad["score"]

    safe = predict.fraud_risk({})
    risky = predict.fraud_risk({"is_foreign": 1, "new_device": 1, "card_present": 0, "hour": 3,
                                "txn_count_1h": 4, "merchant_category": "electronics", "amount_ratio": 8.0})
    assert risky["score"] > safe["score"]

    stay = predict.churn_risk({"inactive_months": 0, "app_logins_30d": 20, "salary_credited": 1})
    leave = predict.churn_risk({"inactive_months": 5, "app_logins_30d": 0, "salary_credited": 0, "complaints_12m": 3})
    assert leave["probability"] > stay["probability"]
